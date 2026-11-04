"""Unit tests for metrics.py.

Every sample here is a SYNTHETIC test fixture with expected values worked out by
hand (or with exact ``fractions``); none are real measurements. No GPU, model,
network, clock or CPU speed is involved.
"""

import copy
import json
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from inference_lab import metrics
from inference_lab.config import SUPPORTED_BATCH_SIZES
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.metrics import StabilityFlag, compute_metrics


def const(value, n=50):
    return [float(value)] * n


def ramp():
    """SYNTHETIC: trial 0 = 1..50 ms, trial 1 = 51..100 ms, trial 2 = 101..150 ms."""
    return [[float(v) for v in range(s, s + 50)] for s in (1, 51, 101)]


def with_medians(*medians):
    """SYNTHETIC: three constant trials, so each trial median equals the given value."""
    return [const(m) for m in medians]


# --- per-trial metrics --------------------------------------------------------------


def test_per_trial_seconds_inputs_throughput_and_median():
    result = compute_metrics(ramp(), batch_size=4)
    first = result.trials[0]
    # sum(1..50) = 1275 ms -> 1.275 s; 4 * 50 = 200 inputs; median of 1..50 = 25.5
    assert first.sum_batch_execution_seconds == 1.275
    assert first.total_processed_inputs == 200
    assert first.throughput_inputs_per_sec == pytest.approx(
        float(Fraction(200) / Fraction(1275, 1000))
    )
    assert first.trial_median_ms == 25.5
    # trial 1: sum(51..100) = 3775 ms, median 75.5; trial 2: sum(101..150) = 6275 ms, median 125.5
    assert result.trials[1].sum_batch_execution_seconds == pytest.approx(3.775)
    assert result.trials[2].sum_batch_execution_seconds == pytest.approx(6.275)
    assert [t.trial_median_ms for t in result.trials] == [25.5, 75.5, 125.5]
    assert [t.trial_index for t in result.trials] == [0, 1, 2]


def test_sum_equals_sum_of_raw_samples_divided_by_1000():
    samples = ramp()
    result = compute_metrics(samples, 8)
    for trial, raw in zip(result.trials, samples, strict=True):
        assert trial.sum_batch_execution_seconds == pytest.approx(sum(raw) / 1000)


@pytest.mark.parametrize("batch_size", SUPPORTED_BATCH_SIZES)
def test_total_processed_inputs_is_batch_size_times_50(batch_size):
    result = compute_metrics(ramp(), batch_size)
    assert [t.total_processed_inputs for t in result.trials] == [batch_size * 50] * 3
    assert result.aggregate.total_processed_inputs == batch_size * 150


# --- aggregate latency percentiles -------------------------------------------------------


def test_p50_p95_use_numpy_linear_interpolation_over_all_150_samples():
    agg = compute_metrics(ramp(), 4).aggregate  # the 150 samples are exactly 1..150
    # linear rank = q * (n - 1): p50 -> 74.5 => between 75 and 76; p95 -> 141.55 => 142.55
    assert agg.p50_batch_latency_ms == 75.5
    assert agg.p95_batch_latency_ms == pytest.approx(142.55)
    assert agg.min_batch_latency_ms == 1.0
    assert agg.max_batch_latency_ms == 150.0


def test_percentile_of_two_level_data_interpolates_between_neighbours():
    # SYNTHETIC: 100 samples of 10 ms and 50 of 20 ms. Sorted index of p95 is
    # 0.95 * 149 = 141.55 -> both neighbours are 20, so p95 = 20; p50 index 74.5 -> 10.
    samples = [const(10), const(10), const(20)]
    agg = compute_metrics(samples, 1).aggregate
    assert agg.p50_batch_latency_ms == 10.0
    assert agg.p95_batch_latency_ms == 20.0


def test_p95_between_distinct_neighbours():
    # SYNTHETIC: sorted index 141.55 straddles sorted[141] = 3 and sorted[142] = 5.
    flat = [1.0] * 141 + [3.0] + [5.0] * 8
    samples = [flat[0:50], flat[50:100], flat[100:150]]
    agg = compute_metrics(samples, 1).aggregate
    assert agg.p95_batch_latency_ms == pytest.approx(3.0 + 0.55 * (5.0 - 3.0))


# --- aggregate throughput ----------------------------------------------------------------


def test_aggregate_throughput_is_total_inputs_over_total_seconds():
    result = compute_metrics(ramp(), 4)
    total_seconds = Fraction(1275 + 3775 + 6275, 1000)
    assert result.aggregate.sum_batch_execution_seconds == pytest.approx(
        float(total_seconds)
    )
    assert result.aggregate.aggregate_throughput_inputs_per_sec == pytest.approx(
        float(Fraction(600) / total_seconds)
    )


def test_aggregate_is_not_the_mean_of_trial_rates():
    # SYNTHETIC unequal speeds, batch size 8: 10 ms/batch, 20 ms/batch, 20 ms/batch.
    result = compute_metrics([const(10), const(20), const(20)], 8)
    rates = [t.throughput_inputs_per_sec for t in result.trials]
    assert rates == pytest.approx([800.0, 400.0, 400.0])
    mean_of_rates = sum(rates) / 3  # 533.33.. : the WRONG way to aggregate
    # right way: 1200 inputs / (0.5 + 1.0 + 1.0) s = 480 inputs/s
    assert result.aggregate.aggregate_throughput_inputs_per_sec == pytest.approx(480.0)
    assert result.aggregate.aggregate_throughput_inputs_per_sec != pytest.approx(
        mean_of_rates
    )


def test_equal_speed_trials_give_matching_trial_and_aggregate_throughput():
    result = compute_metrics([const(5), const(5), const(5)], 16)
    expected = 16 / 0.005  # 16 inputs per 5 ms
    assert result.aggregate.aggregate_throughput_inputs_per_sec == pytest.approx(
        expected
    )
    assert all(
        t.throughput_inputs_per_sec == pytest.approx(expected) for t in result.trials
    )


# --- stability ---------------------------------------------------------------------------


def test_spread_formula_and_flag_for_unstable_run():
    agg = compute_metrics(ramp(), 4).aggregate  # medians 25.5, 75.5, 125.5
    expected = float(Fraction(1255, 10) - Fraction(255, 10)) / 75.5 * 100
    assert agg.trial_median_spread_pct == pytest.approx(expected)
    assert agg.stability_flag is StabilityFlag.UNSTABLE


def test_identical_trials_are_stable_with_zero_spread():
    agg = compute_metrics(with_medians(12, 12, 12), 1).aggregate
    assert agg.trial_median_spread_pct == 0.0
    assert agg.stability_flag is StabilityFlag.STABLE


def test_spread_of_exactly_10_percent_is_stable():
    # medians 90, 100, 100: (100 - 90) / 100 * 100 = 10.0 exactly -> not "above" threshold
    agg = compute_metrics(with_medians(90, 100, 100), 1).aggregate
    assert agg.trial_median_spread_pct == 10.0
    assert agg.stability_flag is StabilityFlag.STABLE
    # same boundary with the largest trial as the outlier: (110 - 100) / 100 * 100
    agg = compute_metrics(with_medians(100, 100, 110), 1).aggregate
    assert agg.trial_median_spread_pct == 10.0
    assert agg.stability_flag is StabilityFlag.STABLE


def test_spread_just_above_10_percent_is_unstable():
    agg = compute_metrics(with_medians(100, 100, 110.01), 1).aggregate
    assert agg.trial_median_spread_pct > 10.0
    assert agg.stability_flag is StabilityFlag.UNSTABLE
    agg = compute_metrics(with_medians(89.9, 100, 100), 1).aggregate
    assert agg.stability_flag is StabilityFlag.UNSTABLE


def test_spread_just_below_10_percent_is_stable():
    agg = compute_metrics(with_medians(100, 100, 109.99), 1).aggregate
    assert agg.trial_median_spread_pct < 10.0
    assert agg.stability_flag is StabilityFlag.STABLE


def test_threshold_constant_and_helper_are_explicit():
    assert metrics.STABILITY_SPREAD_THRESHOLD_PCT == 10.0
    assert metrics.stability_flag(10.0) is StabilityFlag.STABLE
    assert metrics.stability_flag(math.nextafter(10.0, 11.0)) is StabilityFlag.UNSTABLE
    assert metrics.stability_flag(0.0) is StabilityFlag.STABLE
    assert compute_metrics(ramp(), 1).aggregate.stability_threshold_pct == 10.0


def test_unstable_run_keeps_every_sample():
    samples = ramp()
    result = compute_metrics(samples, 4)
    assert result.aggregate.stability_flag is StabilityFlag.UNSTABLE
    assert result.latency_samples_ms == samples
    assert len(result.trials) == 3  # nothing dropped or re-run


# --- raw samples, serialization and types --------------------------------------------------


def test_all_150_raw_samples_preserved_unrounded_and_input_untouched():
    samples = [
        [1.0000000000000002 + i * 1e-13 + t * 1e-12 for i in range(50)]
        for t in range(3)
    ]
    before = copy.deepcopy(samples)
    result = compute_metrics(samples, 4)
    assert samples == before  # input not mutated
    assert result.latency_samples_ms == before  # no rounding, no reordering
    assert sum(len(t) for t in result.latency_samples_ms) == 150
    assert result.latency_samples_ms is not samples  # an independent copy


def test_integer_samples_are_accepted_and_returned_as_floats():
    result = compute_metrics([[2] * 50, [2] * 50, [2] * 50], 1)
    assert all(type(s) is float for trial in result.latency_samples_ms for s in trial)


def test_result_is_json_serializable_with_plain_python_types():
    result = compute_metrics(ramp(), 4)
    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data  # exact round trip
    assert data["batch_size"] == 4
    assert data["aggregate"]["stability_flag"] == "unstable"
    for trial in data["trials"]:
        assert type(trial["total_processed_inputs"]) is int
        assert type(trial["trial_index"]) is int
        for key in (
            "sum_batch_execution_seconds",
            "throughput_inputs_per_sec",
            "trial_median_ms",
        ):
            assert type(trial[key]) is float
    for key, value in data["aggregate"].items():
        if key == "stability_flag":
            assert type(value) is str
        elif key == "total_processed_inputs":
            assert type(value) is int
        else:
            assert type(value) is float, key
    assert all(type(s) is float for t in data["latency_samples_ms"] for s in t)


def test_canonical_field_names_are_present():
    data = compute_metrics(ramp(), 1).to_dict()
    assert set(data["trials"][0]) == {
        "trial_index",
        "sum_batch_execution_seconds",
        "total_processed_inputs",
        "throughput_inputs_per_sec",
        "trial_median_ms",
    }
    assert {
        "p50_batch_latency_ms",
        "p95_batch_latency_ms",
        "min_batch_latency_ms",
        "max_batch_latency_ms",
        "aggregate_throughput_inputs_per_sec",
        "trial_median_spread_pct",
        "stability_flag",
    } <= set(data["aggregate"])


def test_very_small_positive_durations_are_valid():
    tiny = 1e-9  # ms; SYNTHETIC
    result = compute_metrics([const(tiny)] * 3, 4)
    assert result.trials[0].sum_batch_execution_seconds == pytest.approx(
        50 * tiny / 1000
    )
    assert result.trials[0].throughput_inputs_per_sec == pytest.approx(
        200 / (50 * tiny / 1000)
    )
    assert result.aggregate.min_batch_latency_ms == tiny
    assert result.aggregate.stability_flag is StabilityFlag.STABLE
    assert math.isfinite(result.aggregate.aggregate_throughput_inputs_per_sec)


def test_works_directly_from_a_timing_result_shape():
    timing_like = SimpleNamespace(latency_samples_ms=ramp(), batch_size=4)
    result = metrics.compute_metrics_from_timing(timing_like)
    assert result.batch_size == 4
    assert result.aggregate.p50_batch_latency_ms == 75.5


# --- invalid samples ------------------------------------------------------------------------


def assert_invalid_measurement(samples, batch_size=4):
    with pytest.raises(InferenceLabError) as err:
        compute_metrics(samples, batch_size)
    assert err.value.code is ErrorCode.INVALID_MEASUREMENT


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [const(1), const(1)],
        [const(1)] * 4,
        [const(1, 49), const(1), const(1)],
        [const(1), const(1), const(1, 51)],
        [[], [], []],
        None,
        "samples",
        [None, const(1), const(1)],
        [5, const(1), const(1)],
    ],
)
def test_wrong_structure_or_counts_are_rejected(samples):
    assert_invalid_measurement(samples)


@pytest.mark.parametrize(
    "bad",
    [0.0, 0, -1.0, -1e-12, math.nan, math.inf, -math.inf, None, "1.0", True, [1.0]],
)
def test_malformed_or_non_positive_values_are_rejected(bad):
    samples = ramp()
    samples[2][49] = bad
    assert_invalid_measurement(samples)


# --- numeric range of derived values --------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        5e-324,  # smallest subnormal: seconds underflow to 0 (ZeroDivisionError)
        1e-310,  # subnormal: finite seconds but infinite throughput
        1e308,  # sum of samples overflows (OverflowError in fsum)
        1.7976931348623157e308,  # largest finite double
    ],
)
@pytest.mark.parametrize("batch_size", [1, 32])
def test_extreme_finite_durations_raise_invalid_measurement(value, batch_size):
    with pytest.raises(InferenceLabError) as err:
        compute_metrics([const(value)] * 3, batch_size)
    assert err.value.code is ErrorCode.INVALID_MEASUREMENT


def test_one_huge_sample_with_finite_derived_metrics_is_kept_unclamped():
    # The exact sum stays below the double maximum, so every derived value is finite
    # and the result is valid: no clamping, substitution or discarding.
    samples = ramp()
    samples[1][10] = 1.7976931348623157e308
    result = compute_metrics(samples, 4)
    assert result.aggregate.max_batch_latency_ms == 1.7976931348623157e308
    assert result.latency_samples_ms[1][10] == 1.7976931348623157e308
    json.dumps(result.to_dict(), allow_nan=False)


def test_extreme_values_produce_no_numpy_warnings(recwarn):
    with pytest.raises(InferenceLabError):
        compute_metrics([const(1e308)] * 3, 4)
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("value", [1e-300, 1e-9, 1e-6, 1.0, 1e6, 1e12])
def test_returned_metrics_are_finite_json_safe_and_have_positive_denominators(value):
    result = compute_metrics([const(value)] * 3, 32)
    json.dumps(result.to_dict(), allow_nan=False)  # raises on NaN/inf
    assert all(t.sum_batch_execution_seconds > 0 for t in result.trials)
    assert result.aggregate.sum_batch_execution_seconds > 0
    assert math.isfinite(result.aggregate.aggregate_throughput_inputs_per_sec)


def test_normal_results_serialize_without_nan():
    result = compute_metrics(ramp(), 4)
    text = json.dumps(result.to_dict(), allow_nan=False)
    assert json.loads(text) == result.to_dict()


# --- batch size -------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 2, 3, 64, True, False, 4.0, "4", None])
def test_unsupported_batch_sizes_are_rejected(bad):
    with pytest.raises(InferenceLabError) as err:
        compute_metrics(ramp(), bad)
    assert err.value.code is ErrorCode.CONFIG_INVALID


# --- SYNTHETIC demonstration ---------------------------------------------------------------------


def test_synthetic_demo_throughput_scales_with_batch_size_and_inversely_with_time():
    """SYNTHETIC illustration only, not a measurement of any device.

    With identical 10 ms batch latency, throughput is proportional to batch size;
    doubling the batch latency halves it.
    """
    rates = {
        b: compute_metrics(
            [const(10)] * 3, b
        ).aggregate.aggregate_throughput_inputs_per_sec
        for b in SUPPORTED_BATCH_SIZES
    }
    assert rates == pytest.approx(
        {1: 100.0, 4: 400.0, 8: 800.0, 16: 1600.0, 32: 3200.0}
    )
    slower = compute_metrics([const(20)] * 3, 8).aggregate
    assert slower.aggregate_throughput_inputs_per_sec == pytest.approx(rates[8] / 2)
