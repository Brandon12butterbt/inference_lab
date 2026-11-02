"""Untimed CPU/CUDA output parity check.

Implements ``docs/BENCHMARK_PROTOCOL.md`` section 3, step 5: the same pinned
model must produce numerically equivalent logits on the CPU (the reference) and
on CUDA for identical inputs, before any timing begins. This is a correctness
check only; nothing here is timed or measured for performance.

Pass criteria (initial, owner-approved; never loosened automatically):

* both outputs have shape ``(batch_size, 2)``, dtype float32 and only finite values;
* elementwise ``|cuda - cpu| <= atol + rtol * |cpu|`` with ``atol = rtol = 1e-3``,
  evaluated on every element (an aggregate difference would not be a valid test);
* the argmax label of every row is identical.

``max_abs_diff`` and ``max_rel_diff`` are diagnostics. The relative difference is
``|cuda - cpu| / max(|cpu|, atol)``: flooring the reference magnitude at ``atol``
keeps it finite when a CPU value is zero or tiny, where it measures the difference
on the scale of the absolute tolerance instead of exploding.
"""

import enum
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from inference_lab import models
from inference_lab.config import BenchmarkConfig
from inference_lab.dataset import PARITY_POSITION, batch_indices
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.precision import PrecisionSettings, apply_fp32_baseline
from inference_lab.tokenization import TokenizedFixture

ATOL = 1e-3
RTOL = 1e-3
NUM_LABELS = 2


class ParityStatus(enum.StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ParityResult:
    """Structured, JSON-serializable outcome of one parity check."""

    status: ParityStatus
    atol: float
    rtol: float
    max_abs_diff: float | None
    max_rel_diff: float | None
    labels_match: bool | None
    batch_position: int
    batch_size: int
    reference_device: str
    compared_device: str
    compared_device_name: str | None = None
    fixture_indices: list[int] | None = None
    num_elements: int | None = None
    num_violations: int | None = None
    worst_element: dict[str, Any] | None = None
    reference_labels: list[int] | None = None
    candidate_labels: list[int] | None = None
    reference_logits: list[list[float]] | None = None
    candidate_logits: list[list[float]] | None = None
    failure_reasons: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParityResult":
        return cls(**{**data, "status": ParityStatus(data["status"])})

    def raise_if_failed(self) -> None:
        """Raise ``PARITY_FAILED`` with the full result as ``details``."""
        if self.status is not ParityStatus.FAILED:
            return
        parts = [f"batch position {self.batch_position}, batch size {self.batch_size}"]
        if self.max_abs_diff is not None:
            parts.append(
                f"max_abs_diff={self.max_abs_diff:.3e}, "
                f"max_rel_diff={self.max_rel_diff:.3e}, "
                f"{self.num_violations}/{self.num_elements} elements outside "
                f"|cuda-cpu| <= {self.atol} + {self.rtol}*|cpu|"
            )
        if self.labels_match is not None:
            parts.append(f"labels_match={self.labels_match}")
        parts.extend(self.failure_reasons or [])
        raise InferenceLabError(
            ErrorCode.PARITY_FAILED,
            f"CPU/{self.compared_device} output parity failed: "
            + "; ".join(parts)
            + ".",
            {"parity_result": self.to_dict()},
        )


def not_applicable_result(
    config: BenchmarkConfig, batch_position: int = PARITY_POSITION
) -> ParityResult:
    """Parity record for a CPU configuration (the CPU is the reference)."""
    return ParityResult(
        status=ParityStatus.NOT_APPLICABLE,
        atol=ATOL,
        rtol=RTOL,
        max_abs_diff=None,
        max_rel_diff=None,
        labels_match=None,
        batch_position=batch_position,
        batch_size=config.batch_size,
        reference_device="cpu",
        compared_device=config.device,
    )


def _json_list(tensor: torch.Tensor) -> list[Any]:
    """Tensor to nested lists, mapping non-finite floats to None (valid JSON)."""
    return [[v if math.isfinite(v) else None for v in row] for row in tensor.tolist()]


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    batch_size: int,
    batch_position: int = PARITY_POSITION,
    reference_device: str = "cpu",
    compared_device: str = "cuda:0",
    compared_device_name: str | None = None,
    fixture_indices: list[int] | None = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> ParityResult:
    """Compare ``candidate`` logits to the CPU ``reference``; never raises on a mismatch.

    Returns a ``ParityResult`` with status ``passed`` or ``failed``; call
    ``raise_if_failed()`` to turn a failure into ``PARITY_FAILED``.
    """
    base = {
        "atol": atol,
        "rtol": rtol,
        "batch_position": batch_position,
        "batch_size": batch_size,
        "reference_device": reference_device,
        "compared_device": compared_device,
        "compared_device_name": compared_device_name,
        "fixture_indices": fixture_indices,
    }
    expected_shape = (batch_size, NUM_LABELS)
    reasons: list[str] = []
    for name, tensor in (("reference", reference), ("candidate", candidate)):
        if tuple(tensor.shape) != expected_shape:
            reasons.append(
                f"{name} logits have shape {tuple(tensor.shape)}; expected {expected_shape}"
            )
        if tensor.dtype != torch.float32:
            reasons.append(
                f"{name} logits have dtype {tensor.dtype}; expected torch.float32"
            )
    if not reasons:
        for name, tensor in (("reference", reference), ("candidate", candidate)):
            if not bool(torch.isfinite(tensor).all()):
                reasons.append(f"{name} logits contain NaN or infinity")
    if reasons:
        return ParityResult(
            status=ParityStatus.FAILED,
            max_abs_diff=None,
            max_rel_diff=None,
            labels_match=None,
            failure_reasons=reasons,
            **base,
        )

    ref = reference.detach().to("cpu").double()
    cand = candidate.detach().to("cpu").double()
    diff = (cand - ref).abs()
    violations = diff > (atol + rtol * ref.abs())
    rel = diff / ref.abs().clamp_min(atol)
    ref_labels = ref.argmax(dim=1)
    cand_labels = cand.argmax(dim=1)
    labels_match = bool((ref_labels == cand_labels).all())
    worst = int(diff.argmax())
    row, col = divmod(worst, NUM_LABELS)
    num_violations = int(violations.sum())

    reasons = []
    if num_violations:
        reasons.append(f"{num_violations} element(s) exceed the elementwise tolerance")
    if not labels_match:
        reasons.append("predicted labels differ between CPU and CUDA")
    return ParityResult(
        status=ParityStatus.FAILED if reasons else ParityStatus.PASSED,
        max_abs_diff=float(diff.max()),
        max_rel_diff=float(rel.max()),
        labels_match=labels_match,
        num_elements=diff.numel(),
        num_violations=num_violations,
        worst_element={
            "row": row,
            "column": col,
            "cpu": float(ref[row, col]),
            "cuda": float(cand[row, col]),
            "abs_diff": float(diff[row, col]),
        },
        reference_labels=ref_labels.tolist(),
        candidate_labels=cand_labels.tolist(),
        reference_logits=_json_list(reference.detach().to("cpu")),
        candidate_logits=_json_list(candidate.detach().to("cpu")),
        failure_reasons=reasons or None,
        **base,
    )


# --- thin device helpers (separate so tests can run without CUDA) ---------------


def _to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device)


def _synchronize(device: str) -> None:
    torch.cuda.synchronize(device)


def _device_name(device: str) -> str | None:
    try:
        return torch.cuda.get_device_name(device)
    except (RuntimeError, AssertionError, ValueError):
        return None


def _forward_logits(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.inference_mode():
        return model(input_ids=input_ids, attention_mask=attention_mask).logits


def run_parity_check(
    config: BenchmarkConfig,
    model: Any,
    fixture: TokenizedFixture,
    batch_position: int = PARITY_POSITION,
) -> tuple[ParityResult, PrecisionSettings | None]:
    """Compare CPU and CUDA logits of ``model`` for one deterministic fixture batch.

    ``model`` must be the pinned model on the CPU in eval mode (as returned by
    ``models.load_model``) and ``fixture`` the verified tokenized fixture. For a
    CPU configuration this returns a ``not_applicable`` result. For ``cuda:0`` it
    applies and verifies the FP32 baseline, computes the CPU reference, moves the
    *same* model and the *same* batch to CUDA, synchronizes, and compares; the
    model is left on CUDA. A failed comparison raises ``PARITY_FAILED`` (the full
    result is in ``details["parity_result"]``); unavailable CUDA raises
    ``CUDA_UNAVAILABLE`` before any computation. Returns the result and the
    effective precision settings (``None`` when not applicable).
    """
    if config.device == "cpu":
        return not_applicable_result(config, batch_position), None
    models.require_cuda_available(config.device)

    first = next(model.parameters())
    if first.device.type != "cpu" or model.training:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            "run_parity_check expects the model on the CPU in evaluation mode "
            f"(found device {first.device}, training={model.training}).",
        )
    precision_settings = apply_fp32_baseline()

    indices = batch_indices(config.batch_size, batch_position)
    input_ids = fixture.input_ids[indices]
    attention_mask = fixture.attention_mask[indices]

    reference = _forward_logits(model, input_ids, attention_mask).detach().clone()

    model = models.place_on_device(model, config.device)
    try:
        candidate = _forward_logits(
            model,
            _to_device(input_ids, config.device),
            _to_device(attention_mask, config.device),
        )
        _synchronize(config.device)
        candidate = candidate.detach().to("cpu")
    except torch.cuda.OutOfMemoryError as exc:
        raise InferenceLabError(
            ErrorCode.CUDA_OOM, f"Out of GPU memory during the parity forward: {exc}"
        ) from None
    except RuntimeError as exc:
        raise InferenceLabError(
            ErrorCode.CUDA_RUNTIME_ERROR, f"CUDA error during the parity forward: {exc}"
        ) from None

    result = compare_logits(
        reference,
        candidate,
        batch_size=config.batch_size,
        batch_position=batch_position,
        reference_device="cpu",
        compared_device=config.device,
        compared_device_name=_device_name(config.device),
        fixture_indices=indices,
    )
    result.raise_if_failed()
    return result, precision_settings
