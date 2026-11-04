"""Forward-pass timing component.

Implements ``docs/BENCHMARK_PROTOCOL.md`` section 3, steps 4, 6 and 7 (input
preparation, warm-up, timed trials) and the sample validation of step 8. Metrics,
the memory profile, environment metadata and orchestration live elsewhere (M2.2+).

Timing mechanism (methodology: synchronized host wall clock, not CUDA events):

* CPU:  ``t0 = perf_counter_ns(); forward; t1 = perf_counter_ns()``
* CUDA: ``synchronize(device); t0 = perf_counter_ns(); forward;
  synchronize(device); t1 = perf_counter_ns()``

Elapsed milliseconds are ``(t1 - t0) / 1_000_000``. Because CUDA launches are
asynchronous, the synchronize before ``t0`` drains earlier work and the one before
``t1`` waits for this forward to finish; without them the host would time only
kernel *dispatch*. Nothing else (tokenizing, slicing, device transfers, reading
logits, ``.item()``, printing, statistics) happens between the two timestamps;
all inputs are assembled and placed on the device beforehand by ``prepare_inputs``.
"""

import gc
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from inference_lab import models
from inference_lab.config import BenchmarkConfig
from inference_lab.dataset import (
    MEASURED_POSITIONS,
    WARMUP_POSITIONS,
    batch_indices,
)
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.precision import PrecisionSettings, apply_fp32_baseline
from inference_lab.tokenization import TokenizedFixture

# Positions 0-9 (warm-up) and 10-59 (measured) are all assembled up front.
PREPARED_POSITIONS = range(WARMUP_POSITIONS.start, MEASURED_POSITIONS.stop)
NS_PER_MS = 1_000_000
TIMING_MECHANISM_CPU = "perf_counter_ns"
TIMING_MECHANISM_CUDA = "cuda_synchronized_perf_counter_ns"


@dataclass(frozen=True, eq=False)
class PreparedBatch:
    """One batch, already sliced and placed on the target device."""

    position: int
    fixture_indices: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True, eq=False)
class PreparedInputs:
    """All 60 batches (positions 0-59) for one device and batch size."""

    device: str
    batch_size: int
    batches: tuple[PreparedBatch, ...]


@dataclass(frozen=True)
class ThreadSettings:
    """CPU intra-op thread count requested by the config and the effective one."""

    requested: int
    effective: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingResult:
    """Raw timing output of one configuration (not the final M2 result contract)."""

    device: str
    batch_size: int
    timing_mechanism: str
    warmup_forwards: int
    warmup_positions: list[int]
    measured_positions: list[int]
    latency_samples_ms: list[list[float]]
    thread_settings: ThreadSettings
    precision_settings: PrecisionSettings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- thin seams (separate so tests can run without CUDA and with a fake clock) ----


def _to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device)


def _synchronize(device: str) -> None:
    torch.cuda.synchronize(device)


def _clock() -> int:
    return time.perf_counter_ns()


def _model_device(model: Any) -> str:
    return str(next(model.parameters()).device)


# --- execution controls ------------------------------------------------------------


def apply_cpu_threads(config: BenchmarkConfig) -> ThreadSettings:
    """Set ``torch.set_num_threads(config.cpu_threads)`` and read it back.

    The count is never capped or altered. A mismatch between requested and
    effective values fails the run (``CONFIG_INVALID``); inter-op threads, CPU
    affinity and power settings are not touched.
    """
    requested = config.cpu_threads
    try:
        torch.set_num_threads(requested)
        effective = int(torch.get_num_threads())
    except (RuntimeError, ValueError) as exc:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Cannot set the CPU intra-op thread count to {requested}: {exc}",
        ) from None
    if effective != requested:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"CPU intra-op threads: requested {requested} but PyTorch reports "
            f"{effective}.",
            {"requested": requested, "effective": effective},
        )
    return ThreadSettings(requested=requested, effective=effective)


# --- input preparation (untimed) -----------------------------------------------------


def prepare_inputs(
    config: BenchmarkConfig, fixture: TokenizedFixture
) -> PreparedInputs:
    """Assemble batch positions 0-59 and place them on ``config.device``.

    Uses ``batch_indices`` so CPU and CUDA configurations see identical fixture
    indices in identical order. For CUDA the copies are synchronized before
    returning so no transfer is still in flight when warm-up starts. Never
    switches device: CUDA that is not available raises ``CUDA_UNAVAILABLE``.
    """
    if config.device != "cpu":
        models.require_cuda_available(config.device)
    expected = (len(fixture.input_ids), config.sequence_length)
    if (
        tuple(fixture.input_ids.shape) != expected
        or tuple(fixture.attention_mask.shape) != expected
    ):
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Fixture tensors must have shape {expected}; got "
            f"{tuple(fixture.input_ids.shape)} and {tuple(fixture.attention_mask.shape)}.",
        )
    batches = []
    position = None
    try:
        for position in PREPARED_POSITIONS:
            indices = batch_indices(config.batch_size, position)
            batches.append(
                PreparedBatch(
                    position=position,
                    fixture_indices=indices,
                    input_ids=_to_device(fixture.input_ids[indices], config.device),
                    attention_mask=_to_device(
                        fixture.attention_mask[indices], config.device
                    ),
                )
            )
        position = None  # all transfers done; only the final synchronize remains
        if config.device != "cpu":
            _synchronize(config.device)
    except RuntimeError as exc:
        if config.device == "cpu":
            raise
        stage = (
            f"transferring batch position {position}"
            if position is not None
            else "the final synchronization"
        )
        details = {
            "device": config.device,
            "batch_size": config.batch_size,
            "stage": stage,
            "batches_prepared": len(batches),
        }
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise InferenceLabError(
                ErrorCode.CUDA_OOM,
                f"Out of GPU memory while preparing inputs ({stage}): {exc}",
                details,
            ) from None
        raise InferenceLabError(
            ErrorCode.CUDA_RUNTIME_ERROR,
            f"CUDA error while preparing inputs ({stage}): {exc}",
            details,
        ) from None
    return PreparedInputs(config.device, config.batch_size, tuple(batches))


# --- sample validation ------------------------------------------------------------------


def validate_samples(samples: list[list[float]], trials: int, per_trial: int) -> None:
    """Raise ``INVALID_MEASUREMENT`` unless there are ``trials`` x ``per_trial``
    finite, strictly positive samples."""
    problems = []
    if len(samples) != trials:
        problems.append(f"expected {trials} trials, got {len(samples)}")
    for t, trial in enumerate(samples):
        if len(trial) != per_trial:
            problems.append(f"trial {t} has {len(trial)} samples, expected {per_trial}")
        for i, value in enumerate(trial):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"trial {t} sample {i} is not a number: {value!r}")
            elif not math.isfinite(value) or value <= 0:
                problems.append(
                    f"trial {t} sample {i} is not finite and strictly positive: {value!r}"
                )
    if problems:
        shown = problems[:10]
        raise InferenceLabError(
            ErrorCode.INVALID_MEASUREMENT,
            "Invalid latency samples: "
            + "; ".join(shown)
            + (
                f"; ... ({len(problems)} problems in total)"
                if len(problems) > 10
                else ""
            ),
            {"problems": problems},
        )


# --- timed loops ---------------------------------------------------------------------------


def _warmup(model: Any, inputs: PreparedInputs, cuda: bool) -> None:
    for position in WARMUP_POSITIONS:
        batch = inputs.batches[position]
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    if cuda:
        _synchronize(inputs.device)


def _trial_cpu(model: Any, batches: tuple[PreparedBatch, ...]) -> list[float]:
    clock = _clock
    elapsed = []
    for batch in batches:
        input_ids, attention_mask = batch.input_ids, batch.attention_mask
        start = clock()
        model(input_ids=input_ids, attention_mask=attention_mask)
        end = clock()
        elapsed.append((end - start) / NS_PER_MS)
    return elapsed


def _trial_cuda(
    model: Any, batches: tuple[PreparedBatch, ...], device: str
) -> list[float]:
    clock = _clock
    synchronize = _synchronize
    elapsed = []
    for batch in batches:
        input_ids, attention_mask = batch.input_ids, batch.attention_mask
        synchronize(device)
        start = clock()
        model(input_ids=input_ids, attention_mask=attention_mask)
        synchronize(device)
        end = clock()
        elapsed.append((end - start) / NS_PER_MS)
    return elapsed


def _run_trial(model: Any, inputs: PreparedInputs, cuda: bool) -> list[float]:
    """One trial: GC collected outside timing, cyclic GC off for the trial."""
    batches = inputs.batches[MEASURED_POSITIONS.start : MEASURED_POSITIONS.stop]
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        if cuda:
            return _trial_cuda(model, batches, inputs.device)
        return _trial_cpu(model, batches)
    finally:
        if gc_was_enabled:
            gc.enable()


# --- public entry point ---------------------------------------------------------------------


def _check_inputs(config: BenchmarkConfig, model: Any, inputs: PreparedInputs) -> None:
    def invalid(message: str) -> InferenceLabError:
        return InferenceLabError(ErrorCode.CONFIG_INVALID, message)

    if config.device != "cpu":
        models.require_cuda_available(config.device)
    if inputs.device != config.device:
        raise invalid(
            f"Inputs were prepared for {inputs.device!r} but the config requests "
            f"{config.device!r}."
        )
    if inputs.batch_size != config.batch_size:
        raise invalid(
            f"Inputs were prepared for batch size {inputs.batch_size} but the config "
            f"requests {config.batch_size}."
        )
    if [b.position for b in inputs.batches] != list(PREPARED_POSITIONS):
        raise invalid(
            f"Inputs must hold exactly batch positions 0-{PREPARED_POSITIONS.stop - 1}."
        )
    if model.training:
        raise invalid("The model must be in evaluation mode.")
    model_device = _model_device(model)
    if model_device != config.device:
        raise invalid(
            f"The model is on {model_device!r} but the config requests {config.device!r}."
        )


def run_timing(
    config: BenchmarkConfig, model: Any, inputs: PreparedInputs
) -> TimingResult:
    """Warm up, then time 3 x 50 forwards of ``model`` over the prepared batches.

    ``model`` must already be loaded, in eval mode and on ``config.device``;
    ``inputs`` come from ``prepare_inputs``. Applies and verifies the FP32 baseline
    and the CPU thread count first. Any failure raises and no partial result is
    returned; samples are never retried or discarded.
    """
    _check_inputs(config, model, inputs)
    precision_settings = apply_fp32_baseline()
    thread_settings = apply_cpu_threads(config)
    cuda = config.device != "cpu"

    samples: list[list[float]] = []
    try:
        with torch.inference_mode():
            _warmup(model, inputs, cuda)
            for _ in range(config.trials):
                samples.append(_run_trial(model, inputs, cuda))
    except torch.cuda.OutOfMemoryError as exc:
        raise InferenceLabError(
            ErrorCode.CUDA_OOM, f"Out of GPU memory during timing: {exc}"
        ) from None
    except RuntimeError as exc:
        code = ErrorCode.CUDA_RUNTIME_ERROR if cuda else ErrorCode.INVALID_MEASUREMENT
        raise InferenceLabError(
            code, f"Forward pass failed during timing: {exc}"
        ) from None

    validate_samples(samples, config.trials, config.measured_iterations_per_trial)
    return TimingResult(
        device=config.device,
        batch_size=config.batch_size,
        timing_mechanism=TIMING_MECHANISM_CUDA if cuda else TIMING_MECHANISM_CPU,
        warmup_forwards=len(WARMUP_POSITIONS),
        warmup_positions=list(WARMUP_POSITIONS),
        measured_positions=list(MEASURED_POSITIONS),
        latency_samples_ms=samples,
        thread_settings=thread_settings,
        precision_settings=precision_settings,
    )
