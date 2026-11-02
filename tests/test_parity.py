"""Offline unit tests for ``inference_lab.parity``.

All logits and models here are synthetic test fixtures built from tiny tensors
and fakes; no CUDA hardware, model weights or network are used, and none of the
numbers are measurements.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from inference_lab import models, parity
from inference_lab.config import BenchmarkConfig
from inference_lab.dataset import PARITY_POSITION, batch_indices
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.parity import (
    ATOL,
    RTOL,
    ParityResult,
    ParityStatus,
    compare_logits,
    not_applicable_result,
    run_parity_check,
)
from inference_lab.precision import PrecisionSettings
from inference_lab.tokenization import TokenizedFixture


def t(rows, dtype=torch.float32):
    return torch.tensor(rows, dtype=dtype)


def compare(reference, candidate, **kwargs):
    batch_size = kwargs.pop("batch_size", len(reference))
    return compare_logits(reference, candidate, batch_size=batch_size, **kwargs)


# --- compare_logits: pass cases -------------------------------------------------


def test_exact_agreement_passes():
    ref = t([[1.0, -1.0], [0.5, 2.0]])
    result = compare(ref, ref.clone())
    assert result.status is ParityStatus.PASSED
    assert result.max_abs_diff == 0.0
    assert result.max_rel_diff == 0.0
    assert result.labels_match is True
    assert result.num_violations == 0
    assert result.num_elements == 4
    assert result.failure_reasons is None
    assert (result.atol, result.rtol) == (ATOL, RTOL) == (1e-3, 1e-3)
    assert result.reference_device == "cpu"
    assert result.compared_device == "cuda:0"


def test_difference_inside_tolerance_passes_and_is_reported():
    ref = t([[1.0, -1.0], [0.5, 2.0]])
    result = compare(ref, ref + 5e-4)  # allowed is at least 1e-3 + 1e-3*|cpu|
    assert result.status is ParityStatus.PASSED
    assert result.max_abs_diff == pytest.approx(5e-4, rel=1e-3)
    assert result.num_violations == 0


def test_tolerance_is_elementwise_not_an_aggregate_threshold():
    # A 0.05 difference on a magnitude-100 element is inside 1e-3 + 1e-3*100,
    # even though 0.05 is far above atol: the rule scales with |cpu| per element.
    ref = t([[100.0, -100.0]])
    cand = t([[100.05, -100.0]])
    result = compare(ref, cand)
    assert result.status is ParityStatus.PASSED
    assert result.max_abs_diff == pytest.approx(0.05, rel=1e-3)
    assert result.max_abs_diff > ATOL


def test_tolerance_boundary_on_a_magnitude_two_element():
    ref = t([[2.0, -2.0]])  # allowed difference: 1e-3 + 1e-3 * 2 = 3e-3
    assert compare(ref, t([[2.0029, -2.0]])).status is ParityStatus.PASSED
    assert compare(ref, t([[2.0031, -2.0]])).status is ParityStatus.FAILED


# --- compare_logits: failure cases ----------------------------------------------


def test_difference_outside_tolerance_fails_with_diagnostics():
    ref = t([[1.0, -1.0], [0.5, 2.0]])
    cand = t([[1.01, -1.0], [0.5, 2.0]])
    result = compare(ref, cand)
    assert result.status is ParityStatus.FAILED
    assert result.num_violations == 1
    assert result.labels_match is True
    assert result.max_abs_diff == pytest.approx(0.01, rel=1e-3)
    assert result.worst_element["row"] == 0
    assert result.worst_element["column"] == 0
    assert result.worst_element["cpu"] == pytest.approx(1.0)
    assert result.worst_element["cuda"] == pytest.approx(1.01)
    assert any("elementwise tolerance" in r for r in result.failure_reasons)


def test_label_mismatch_fails_even_when_numerically_tiny():
    ref = t([[1.0000, 1.0004], [3.0, -3.0]])
    cand = t([[1.0004, 1.0000], [3.0, -3.0]])  # 4e-4 differences: within tolerance
    result = compare(ref, cand)
    assert result.num_violations == 0
    assert result.max_abs_diff < ATOL
    assert result.labels_match is False
    assert result.status is ParityStatus.FAILED
    assert result.reference_labels == [1, 0]
    assert result.candidate_labels == [0, 0]
    assert any("labels differ" in r for r in result.failure_reasons)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_non_finite_values_fail(bad, side):
    good = t([[1.0, -1.0], [0.5, 2.0]])
    broken = good.clone()
    broken[1, 0] = bad
    ref, cand = (broken, good) if side == "reference" else (good, broken)
    result = compare(ref, cand)
    assert result.status is ParityStatus.FAILED
    assert result.max_abs_diff is None and result.max_rel_diff is None
    assert result.labels_match is None
    assert any("NaN or infinity" in r and side in r for r in result.failure_reasons)
    json.dumps(result.to_dict(), allow_nan=False)  # still valid JSON


@pytest.mark.parametrize(
    "candidate",
    [
        t([[1.0, -1.0]] * 3),
        t([[1.0, -1.0, 0.0]] * 2),
        t([1.0, -1.0]),
        t([[1.0, -1.0]] * 2).double(),
    ],
    ids=["wrong-batch", "wrong-classes", "1-d", "float64"],
)
def test_wrong_shape_or_dtype_fails(candidate):
    ref = t([[1.0, -1.0], [0.5, 2.0]])
    result = compare(ref, candidate)
    assert result.status is ParityStatus.FAILED
    assert result.max_abs_diff is None
    assert result.failure_reasons


def test_half_precision_output_is_rejected():
    ref = t([[1.0, -1.0]])
    result = compare(ref, ref.half())
    assert result.status is ParityStatus.FAILED
    assert any("dtype" in r for r in result.failure_reasons)


# --- diagnostics ------------------------------------------------------------------


def test_max_abs_and_relative_difference_values():
    ref = t([[2.0, 0.0], [1.0, -1.0]])
    cand = ref + t([[0.001, 0.0005], [0.0, 0.0]])
    result = compare(ref, cand)
    assert result.max_abs_diff == pytest.approx(0.001, rel=2e-3)
    # rel = |diff| / max(|cpu|, atol): 0.001/2 = 5e-4 and 0.0005/1e-3 = 0.5.
    assert result.max_rel_diff == pytest.approx(0.5, rel=1e-3)
    assert result.status is ParityStatus.PASSED  # 0.0005 <= 1e-3 + 0


def test_relative_difference_stays_finite_for_zero_reference_values():
    ref = t([[0.0, 0.0], [0.0, 1e-30]])
    cand = ref + 1e-3
    result = compare(ref, cand)
    assert result.max_rel_diff is not None
    assert result.max_rel_diff < float("inf")
    assert result.max_rel_diff == pytest.approx(1.0, rel=1e-3)  # 1e-3 / atol
    json.dumps(result.to_dict(), allow_nan=False)


def test_logits_and_labels_are_recorded_for_review():
    ref = t([[1.0, -1.0], [0.5, 2.0]])
    result = compare(
        ref, ref.clone(), fixture_indices=[40, 41], compared_device_name="Test GPU"
    )
    assert result.reference_logits == [[1.0, -1.0], [0.5, 2.0]]
    assert result.candidate_logits == result.reference_logits
    assert result.reference_labels == result.candidate_labels == [0, 1]
    assert result.fixture_indices == [40, 41]
    assert result.compared_device_name == "Test GPU"


# --- results: serialization, failure, not applicable -----------------------------


def test_results_are_json_serializable_and_round_trip():
    ref = t([[1.0, -1.0], [0.5, 2.0]])
    for result in (
        compare(ref, ref + 1e-4),
        compare(ref, ref + 1.0),
        not_applicable_result(BenchmarkConfig(device="cpu", batch_size=4)),
    ):
        data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert data["status"] in {"passed", "failed", "not_applicable"}
        assert ParityResult.from_dict(data) == result


def test_raise_if_failed_uses_parity_failed_with_full_result():
    ref = t([[1.0, -1.0]])
    failed = compare(ref, t([[1.5, -1.0]]))
    with pytest.raises(InferenceLabError) as excinfo:
        failed.raise_if_failed()
    err = excinfo.value
    assert err.code is ErrorCode.PARITY_FAILED
    assert "max_abs_diff=" in err.message
    assert "batch position 10" in err.message
    assert err.details["parity_result"]["status"] == "failed"
    assert err.details["parity_result"]["max_abs_diff"] == pytest.approx(0.5)
    assert compare(ref, ref.clone()).raise_if_failed() is None


def test_not_applicable_for_cpu_configuration():
    config = BenchmarkConfig(device="cpu", batch_size=8)
    result = not_applicable_result(config)
    assert result.status is ParityStatus.NOT_APPLICABLE
    assert result.compared_device == "cpu" and result.reference_device == "cpu"
    assert result.batch_size == 8 and result.batch_position == PARITY_POSITION
    assert result.max_abs_diff is None and result.labels_match is None
    assert result.raise_if_failed() is None


# --- run_parity_check orchestration (fakes; no CUDA) ------------------------------


def make_fixture() -> TokenizedFixture:
    ids = torch.arange(512 * 128, dtype=torch.int64).reshape(512, 128) % 1000
    return TokenizedFixture(
        input_ids=ids,
        attention_mask=torch.ones((512, 128), dtype=torch.int64),
        sequence_length=128,
        token_ids_sha256="0" * 64,
        attention_mask_sha256="0" * 64,
        truncated_count=0,
        real_length_min=1,
        real_length_median=1.0,
        real_length_max=1,
    )


class FakeModel(torch.nn.Module):
    """Logits are a pure function of the input ids; ``shift`` simulates GPU error."""

    def __init__(self, events, shift=0.0, error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.events, self.shift, self.error = events, shift, error
        self.on_device = False
        self.calls = []
        self.eval()

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        self.events.append("forward_device" if self.on_device else "forward_cpu")
        if self.on_device and self.error is not None:
            raise self.error
        mean = input_ids.float().mean(dim=1) / 1000.0
        logits = torch.stack([mean, -mean], dim=1)
        if self.on_device:
            logits = logits + self.shift
        return SimpleNamespace(logits=logits)


@pytest.fixture
def harness(monkeypatch):
    events = []

    def place(model, device):
        events.append(f"place:{device}")
        model.on_device = True
        return model

    def to_device(tensor, device):
        events.append(f"to_device:{device}")
        return tensor

    settings = PrecisionSettings(
        api="legacy",
        float32_matmul_precision="highest",
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        fp32_precision_readback={},
        torch_version="test",
    )

    def apply_precision():
        events.append("precision")
        return settings

    monkeypatch.setattr(models, "cuda_available", lambda: True)
    monkeypatch.setattr(models, "place_on_device", place)
    monkeypatch.setattr(parity, "_to_device", to_device)
    monkeypatch.setattr(parity, "_synchronize", lambda d: events.append(f"sync:{d}"))
    monkeypatch.setattr(parity, "_device_name", lambda d: "Test GPU")
    monkeypatch.setattr(parity, "apply_fp32_baseline", apply_precision)
    return SimpleNamespace(events=events, settings=settings)


def cuda_config(batch_size=4):
    return BenchmarkConfig(device="cuda:0", batch_size=batch_size)


def test_orchestration_order_and_identical_inputs(harness):
    model = FakeModel(harness.events)
    fixture = make_fixture()
    result, settings = run_parity_check(cuda_config(4), model, fixture)

    assert harness.events == [
        "precision",
        "forward_cpu",  # CPU reference first, on the CPU
        "place:cuda:0",  # the same model is then moved
        "to_device:cuda:0",
        "to_device:cuda:0",
        "forward_device",
        "sync:cuda:0",  # synchronized before the results are examined
    ]
    assert settings is harness.settings
    expected = batch_indices(4, 10)
    assert expected == [40, 41, 42, 43]
    cpu_call, device_call = model.calls
    assert torch.equal(cpu_call[0], fixture.input_ids[expected])
    assert torch.equal(cpu_call[0], device_call[0])  # identical batch on both
    assert torch.equal(cpu_call[1], device_call[1])
    assert result.status is ParityStatus.PASSED
    assert result.fixture_indices == expected
    assert result.batch_position == PARITY_POSITION == 10
    assert result.compared_device == "cuda:0"
    assert result.compared_device_name == "Test GPU"
    assert result.max_abs_diff == 0.0


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_uses_deterministic_position_10_for_every_batch_size(harness, batch_size):
    model = FakeModel(harness.events)
    result, _ = run_parity_check(cuda_config(batch_size), model, make_fixture())
    assert result.fixture_indices == batch_indices(batch_size, 10)
    assert len(result.reference_logits) == batch_size


def test_custom_batch_position(harness):
    result, _ = run_parity_check(
        cuda_config(4), FakeModel(harness.events), make_fixture(), batch_position=11
    )
    assert result.fixture_indices == [44, 45, 46, 47]
    assert result.batch_position == 11


def test_numerical_disagreement_raises_parity_failed(harness):
    model = FakeModel(harness.events, shift=0.5)
    with pytest.raises(InferenceLabError) as excinfo:
        run_parity_check(cuda_config(4), model, make_fixture())
    err = excinfo.value
    assert err.code is ErrorCode.PARITY_FAILED
    detail = err.details["parity_result"]
    assert detail["status"] == "failed"
    assert detail["max_abs_diff"] == pytest.approx(0.5, rel=1e-3)
    assert detail["compared_device_name"] == "Test GPU"


def test_cpu_configuration_is_not_applicable_and_touches_nothing(harness):
    model = FakeModel(harness.events)
    result, settings = run_parity_check(
        BenchmarkConfig(device="cpu", batch_size=4), model, make_fixture()
    )
    assert result.status is ParityStatus.NOT_APPLICABLE
    assert settings is None
    assert harness.events == [] and model.calls == []


def test_unavailable_cuda_raises_before_any_computation(harness, monkeypatch):
    monkeypatch.setattr(models, "cuda_available", lambda: False)
    model = FakeModel(harness.events)
    with pytest.raises(InferenceLabError) as excinfo:
        run_parity_check(cuda_config(4), model, make_fixture())
    assert excinfo.value.code is ErrorCode.CUDA_UNAVAILABLE
    assert "not switched to the CPU" in excinfo.value.message
    assert harness.events == [] and model.calls == []


def test_model_must_start_on_cpu_in_eval_mode(harness):
    model = FakeModel(harness.events)
    model.train()
    with pytest.raises(InferenceLabError) as excinfo:
        run_parity_check(cuda_config(4), model, make_fixture())
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert model.calls == []


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (torch.cuda.OutOfMemoryError("oom"), ErrorCode.CUDA_OOM),
        (RuntimeError("kernel launch failed"), ErrorCode.CUDA_RUNTIME_ERROR),
    ],
)
def test_cuda_forward_failures_use_specific_codes(harness, error, code):
    model = FakeModel(harness.events, error=error)
    with pytest.raises(InferenceLabError) as excinfo:
        run_parity_check(cuda_config(4), model, make_fixture())
    assert excinfo.value.code is code


def test_precision_failure_stops_before_any_forward(harness, monkeypatch):
    def failing():
        raise InferenceLabError(ErrorCode.CONFIG_INVALID, "TF32 is enabled")

    monkeypatch.setattr(parity, "apply_fp32_baseline", failing)
    model = FakeModel(harness.events)
    with pytest.raises(InferenceLabError) as excinfo:
        run_parity_check(cuda_config(4), model, make_fixture())
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert model.calls == []
