"""Performance metrics and stability assessment for raw latency samples.

Implements ``docs/BENCHMARK_PROTOCOL.md`` section 4 (metric definitions) and
section 6 (stability assessment). This module only analyses samples that were
already collected (see ``timing.py``); it measures nothing and never alters,
rounds, trims or discards a sample. Rounding and formatting belong to the UI.

Formulas (``B`` = batch size, ``s_i`` = one batch latency in ms):

* ``sum_batch_execution_seconds`` (per trial) = ``sum(s_i) / 1000``
* ``total_processed_inputs`` (per trial) = ``B * 50``
* ``throughput_inputs_per_sec`` (per trial) = inputs / ``sum_batch_execution_seconds``
* ``trial_median_ms`` = median of that trial's 50 samples
* ``p50_batch_latency_ms`` / ``p95_batch_latency_ms`` = ``numpy.percentile`` (default
  linear interpolation) over all 150 samples; min / max over the same samples
* ``aggregate_throughput_inputs_per_sec`` = total inputs of all trials / sum of the
  three ``sum_batch_execution_seconds`` (never a mean of per-trial rates)
* ``trial_median_spread_pct`` = ``(max(medians) - min(medians)) / median(medians) * 100``
* ``stability_flag`` = ``unstable`` if the spread is strictly greater than
  ``STABILITY_SPREAD_THRESHOLD_PCT``, otherwise ``stable``

Sums use ``math.fsum`` (exactly rounded) so the totals do not depend on summation
order. An ``unstable`` run is still a valid measurement.
"""

import enum
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from inference_lab.config import (
    MEASURED_ITERATIONS_PER_TRIAL,
    SUPPORTED_BATCH_SIZES,
    TRIALS,
)
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.timing import TimingResult, validate_samples

# Initial, unvalidated protocol value (section 6); changing it needs an
# owner-approved protocol revision.
STABILITY_SPREAD_THRESHOLD_PCT = 10.0
MS_PER_SECOND = 1000.0


class StabilityFlag(enum.StrEnum):
    STABLE = "stable"
    UNSTABLE = "unstable"


@dataclass(frozen=True)
class TrialMetrics:
    """Metrics of one trial (50 batch latencies)."""

    trial_index: int
    sum_batch_execution_seconds: float
    total_processed_inputs: int
    throughput_inputs_per_sec: float
    trial_median_ms: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Metrics across all trials (150 batch latencies)."""

    p50_batch_latency_ms: float
    p95_batch_latency_ms: float
    min_batch_latency_ms: float
    max_batch_latency_ms: float
    total_processed_inputs: int
    sum_batch_execution_seconds: float
    aggregate_throughput_inputs_per_sec: float
    trial_median_spread_pct: float
    stability_threshold_pct: float
    stability_flag: StabilityFlag


@dataclass(frozen=True)
class MetricsResult:
    """Trial and aggregate metrics plus the unchanged raw samples."""

    batch_size: int
    trials: list[TrialMetrics]
    aggregate: AggregateMetrics
    latency_samples_ms: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aggregate"]["stability_flag"] = self.aggregate.stability_flag.value
        return data


def _check_batch_size(batch_size: Any) -> None:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size not in SUPPORTED_BATCH_SIZES
    ):
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"batch_size must be one of {SUPPORTED_BATCH_SIZES}; got {batch_size!r}.",
            {"field": "batch_size", "value": repr(batch_size)},
        )


def stability_flag(spread_pct: float) -> StabilityFlag:
    """``unstable`` only if the spread is strictly above the threshold."""
    if spread_pct > STABILITY_SPREAD_THRESHOLD_PCT:
        return StabilityFlag.UNSTABLE
    return StabilityFlag.STABLE


def compute_metrics(
    latency_samples_ms: list[list[float]], batch_size: int
) -> MetricsResult:
    """Compute the protocol metrics from 3 trials x 50 raw latencies (ms/batch).

    Raises ``INVALID_MEASUREMENT`` for any missing, malformed, non-finite or
    non-positive sample and ``CONFIG_INVALID`` for an unsupported batch size.
    The input is not modified; the result holds an unrounded copy.
    """
    _check_batch_size(batch_size)
    if not isinstance(latency_samples_ms, (list, tuple)) or not all(
        isinstance(trial, (list, tuple)) for trial in latency_samples_ms
    ):
        raise InferenceLabError(
            ErrorCode.INVALID_MEASUREMENT,
            "latency_samples_ms must be a list of per-trial lists of numbers.",
        )
    validate_samples(latency_samples_ms, TRIALS, MEASURED_ITERATIONS_PER_TRIAL)
    samples = [[float(s) for s in trial] for trial in latency_samples_ms]
    try:
        with np.errstate(all="ignore"):  # judged by the explicit checks below
            result = _calculate(samples, batch_size)
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise InferenceLabError(
            ErrorCode.INVALID_MEASUREMENT,
            "Latency samples are outside the range in which the metrics can be "
            f"calculated ({type(exc).__name__}: {exc}).",
        ) from None
    _require_finite_metrics(result)
    return result


def _require_finite_metrics(result: "MetricsResult") -> None:
    """Fail unless every derived value is finite and every time denominator > 0."""
    data = result.to_dict()
    problems = []
    seconds = [(f"trials[{t.trial_index}]", t) for t in result.trials]
    seconds.append(("aggregate", result.aggregate))
    for label, item in seconds:
        if not item.sum_batch_execution_seconds > 0:
            problems.append(f"{label}.sum_batch_execution_seconds is not positive")

    def walk(name: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{name}.{key}", inner)
        elif isinstance(value, list):
            for i, inner in enumerate(value):
                walk(f"{name}[{i}]", inner)
        elif isinstance(value, float) and not math.isfinite(value):
            problems.append(f"{name} is not finite ({value!r})")

    walk("metrics", data)
    if problems:
        raise InferenceLabError(
            ErrorCode.INVALID_MEASUREMENT,
            "Latency samples produce non-finite or degenerate metrics: "
            + "; ".join(problems[:5])
            + (f"; ... ({len(problems)} problems)" if len(problems) > 5 else ""),
            {"problems": problems},
        )


def _calculate(samples: list[list[float]], batch_size: int) -> "MetricsResult":
    trials = []
    for index, trial in enumerate(samples):
        seconds = math.fsum(trial) / MS_PER_SECOND
        inputs = batch_size * len(trial)
        trials.append(
            TrialMetrics(
                trial_index=index,
                sum_batch_execution_seconds=seconds,
                total_processed_inputs=inputs,
                throughput_inputs_per_sec=inputs / seconds,
                trial_median_ms=float(np.median(trial)),
            )
        )

    everything = np.array([s for trial in samples for s in trial], dtype=np.float64)
    total_inputs = sum(t.total_processed_inputs for t in trials)
    total_seconds = math.fsum(t.sum_batch_execution_seconds for t in trials)
    medians = [t.trial_median_ms for t in trials]
    spread = (max(medians) - min(medians)) / float(np.median(medians)) * 100

    aggregate = AggregateMetrics(
        p50_batch_latency_ms=float(np.percentile(everything, 50)),
        p95_batch_latency_ms=float(np.percentile(everything, 95)),
        min_batch_latency_ms=float(everything.min()),
        max_batch_latency_ms=float(everything.max()),
        total_processed_inputs=total_inputs,
        sum_batch_execution_seconds=total_seconds,
        aggregate_throughput_inputs_per_sec=total_inputs / total_seconds,
        trial_median_spread_pct=spread,
        stability_threshold_pct=STABILITY_SPREAD_THRESHOLD_PCT,
        stability_flag=stability_flag(spread),
    )
    return MetricsResult(batch_size, trials, aggregate, samples)


def compute_metrics_from_timing(result: TimingResult) -> MetricsResult:
    """Convenience wrapper for the output of ``timing.run_timing``."""
    return compute_metrics(result.latency_samples_ms, result.batch_size)
