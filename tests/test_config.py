"""Unit tests for ``inference_lab.errors`` and ``inference_lab.config``.

Every value here is a synthetic unit-test fixture. ``SYNTHETIC_REVISION`` is a
made-up 40-hex string; it is NOT a downloaded model revision, and nothing in
this file represents a benchmark result.
"""

import json
from types import SimpleNamespace

import pytest

import inference_lab.config as config_module
from inference_lab.config import (
    MODEL_ID,
    PINNED_MODEL_REVISION,
    SUPPORTED_BATCH_SIZES,
    SUPPORTED_DEVICES,
    BenchmarkConfig,
    default_cpu_threads,
)
from inference_lab.errors import ErrorCode, InferenceLabError

SYNTHETIC_REVISION = "0123456789abcdef" * 2 + "01234567"


def fake_cpu_detection(monkeypatch, physical, logical):
    """Replace psutil inside config.py so tests never depend on the real CPU.

    ``physical``/``logical`` are returned for ``cpu_count(logical=False/True)``;
    an Exception instance is raised instead of returned.
    """

    def cpu_count(logical=True):
        value = logical_value if logical else physical_value
        if isinstance(value, Exception):
            raise value
        return value

    physical_value, logical_value = physical, logical
    monkeypatch.setattr(config_module, "psutil", SimpleNamespace(cpu_count=cpu_count))


@pytest.fixture(autouse=True)
def synthetic_cpu(monkeypatch):
    """Default fake machine: 8 physical / 16 logical cores (synthetic)."""
    fake_cpu_detection(monkeypatch, physical=8, logical=16)


def make(**overrides) -> BenchmarkConfig:
    values = {
        "model_revision": SYNTHETIC_REVISION,
        "device": "cpu",
        "batch_size": 8,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


# --- errors -------------------------------------------------------------------


def test_error_taxonomy_matches_architecture_doc():
    expected = {
        "CONFIG_INVALID",
        "CUDA_UNAVAILABLE",
        "MODEL_LOAD_FAILED",
        "ATTENTION_BACKEND_MISMATCH",
        "FIXTURE_INTEGRITY_FAILED",
        "PARITY_FAILED",
        "INVALID_MEASUREMENT",
        "CUDA_OOM",
        "CUDA_RUNTIME_ERROR",
        "WORKER_CRASHED",
        "STORAGE_ERROR",
        "INTERRUPTED",
    }
    assert {code.value for code in ErrorCode} == expected
    assert all(code.name == code.value for code in ErrorCode)


def test_error_carries_code_and_readable_message():
    err = InferenceLabError(ErrorCode.CUDA_UNAVAILABLE, "CUDA is not available.")
    assert err.code is ErrorCode.CUDA_UNAVAILABLE
    assert err.message == "CUDA is not available."
    assert str(err) == "[CUDA_UNAVAILABLE] CUDA is not available."
    assert isinstance(err, Exception)


def test_error_accepts_string_code_and_rejects_unknown_code():
    assert InferenceLabError("CUDA_OOM", "out of memory").code is ErrorCode.CUDA_OOM
    with pytest.raises(ValueError, match="Unknown error code"):
        InferenceLabError("NOT_A_CODE", "x")


def test_error_json_round_trip():
    err = InferenceLabError(ErrorCode.PARITY_FAILED, "mismatch", {"max_abs_diff": 0.5})
    payload = json.loads(json.dumps(err.to_dict()))
    assert payload == {
        "error_code": "PARITY_FAILED",
        "error_message": "mismatch",
        "details": {"max_abs_diff": 0.5},
    }
    rebuilt = InferenceLabError.from_dict(payload)
    assert rebuilt.code is ErrorCode.PARITY_FAILED
    assert rebuilt.message == "mismatch"
    assert rebuilt.details == {"max_abs_diff": 0.5}


# --- valid configurations -----------------------------------------------------


def test_valid_cpu_config_uses_protocol_defaults():
    cfg = make()
    assert cfg.device == "cpu"
    assert cfg.model_id == MODEL_ID
    assert cfg.sequence_length == 128
    assert cfg.precision == "fp32"
    assert cfg.matmul_precision == "highest"
    assert cfg.attn_implementation == "sdpa"
    assert cfg.warmup_iterations == 10
    assert cfg.measured_iterations_per_trial == 50
    assert cfg.trials == 3
    assert cfg.memory_profile_forwards == 10
    assert cfg.fixture_id == "synthetic-sentiment-v1"
    assert cfg.fixture_seed == 42
    assert cfg.cpu_threads == 8  # hardware-aware default from the fake machine


def test_valid_cuda_config():
    cfg = make(device="cuda:0", batch_size=32)
    assert cfg.device == "cuda:0"
    assert cfg.batch_size == 32


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
@pytest.mark.parametrize("batch_size", SUPPORTED_BATCH_SIZES)
def test_every_supported_device_and_batch_size_is_accepted(device, batch_size):
    assert make(device=device, batch_size=batch_size).batch_size == batch_size


@pytest.mark.parametrize("threads", [1, 12, 24, 32, 128])
def test_explicit_cpu_threads_accepted_unchanged(threads):
    # Even far above the fake machine's 8/16 cores: never capped or modified.
    assert make(cpu_threads=threads).cpu_threads == threads


# --- hardware-aware cpu_threads default -----------------------------------------


def test_default_prefers_physical_cores(monkeypatch):
    fake_cpu_detection(monkeypatch, physical=6, logical=12)
    assert default_cpu_threads() == 6
    cfg = make()
    assert cfg.cpu_threads == 6
    assert type(cfg.cpu_threads) is int


@pytest.mark.parametrize("physical", [None, 0, -2, "6", 6.0, ValueError("boom")])
def test_default_falls_back_to_logical_count(monkeypatch, physical):
    fake_cpu_detection(monkeypatch, physical=physical, logical=16)
    assert default_cpu_threads() == 16
    assert make().cpu_threads == 16


@pytest.mark.parametrize("bad", [None, 0, -1, "16", RuntimeError("no detection")])
def test_default_falls_back_to_one(monkeypatch, bad):
    fake_cpu_detection(monkeypatch, physical=bad, logical=bad)
    assert default_cpu_threads() == 1
    assert make().cpu_threads == 1


def test_explicit_threads_do_not_consult_hardware(monkeypatch):
    def forbidden(logical=True):
        raise AssertionError("hardware detection must not run")

    monkeypatch.setattr(config_module, "psutil", SimpleNamespace(cpu_count=forbidden))
    assert make(cpu_threads=4).cpu_threads == 4


def test_attention_backend_is_free_configuration():
    # Compatibility is verified at model load (later work package), not here.
    assert make(attn_implementation="eager").attn_implementation == "eager"


def test_config_is_immutable():
    cfg = make()
    with pytest.raises(AttributeError):
        cfg.batch_size = 4  # type: ignore[misc]


# --- invalid configurations ---------------------------------------------------


def assert_config_invalid(field: str, **overrides):
    with pytest.raises(InferenceLabError) as excinfo:
        make(**overrides)
    err = excinfo.value
    assert err.code is ErrorCode.CONFIG_INVALID
    assert field in err.message
    assert err.details["field"] == field
    return err


@pytest.mark.parametrize("batch_size", [0, 2, 3, 64, -1, 8.0, "8", True, None])
def test_invalid_batch_size(batch_size):
    assert_config_invalid("batch_size", batch_size=batch_size)


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "gpu", "CPU", "", None, 0])
def test_invalid_device(device):
    assert_config_invalid("device", device=device)


@pytest.mark.parametrize("length", [0, 64, 127, 129, 512, 128.0, "128", True])
def test_invalid_sequence_length(length):
    assert_config_invalid("sequence_length", sequence_length=length)


@pytest.mark.parametrize("precision", ["fp16", "bf16", "FP32", "", None])
def test_invalid_precision(precision):
    assert_config_invalid("precision", precision=precision)


@pytest.mark.parametrize("value", ["high", "medium", "", None])
def test_invalid_matmul_precision(value):
    assert_config_invalid("matmul_precision", matmul_precision=value)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("warmup_iterations", 0),
        ("warmup_iterations", 5),
        ("warmup_iterations", 10.0),
        ("measured_iterations_per_trial", 49),
        ("measured_iterations_per_trial", 100),
        ("trials", 1),
        ("trials", 5),
        ("trials", True),
        ("memory_profile_forwards", 1),
        ("memory_profile_forwards", 20),
    ],
)
def test_invalid_protocol_iteration_counts(field, bad_value):
    assert_config_invalid(field, **{field: bad_value})


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "main",
        "abc123",
        SYNTHETIC_REVISION[:-1],  # 39 characters
        SYNTHETIC_REVISION + "0",  # 41 characters
        SYNTHETIC_REVISION.upper(),  # uppercase hex
        "g" * 40,  # not hexadecimal
        " " + SYNTHETIC_REVISION[:-1],  # leading whitespace
        SYNTHETIC_REVISION + "\n",  # trailing newline
        None,
        1234,
    ],
)
def test_invalid_model_revision_format(revision):
    assert_config_invalid("model_revision", model_revision=revision)


def test_model_revision_defaults_to_the_pinned_revision():
    cfg = BenchmarkConfig(device="cpu", batch_size=8)
    assert cfg.model_revision == PINNED_MODEL_REVISION
    assert PINNED_MODEL_REVISION == "714eb0fa89d2f80546fda750413ed43d93601a13"
    assert make().model_revision == SYNTHETIC_REVISION  # explicit value wins


def test_explicit_revision_round_trips_unchanged():
    cfg = make()
    assert cfg.model_revision == SYNTHETIC_REVISION
    assert BenchmarkConfig.from_json(cfg.to_json()).model_revision == SYNTHETIC_REVISION
    pinned = BenchmarkConfig(device="cpu", batch_size=8)
    assert BenchmarkConfig.from_dict(pinned.to_dict()) == pinned


def test_from_dict_still_requires_model_revision():
    data = make().to_dict()
    del data["model_revision"]  # saved configs are never silently re-pinned
    with pytest.raises(InferenceLabError) as excinfo:
        BenchmarkConfig.from_dict(data)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert "model_revision" in excinfo.value.message


def test_required_fields_and_keyword_only_construction():
    with pytest.raises(TypeError):
        BenchmarkConfig(device="cpu")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BenchmarkConfig(batch_size=8)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BenchmarkConfig("cpu", 8)  # type: ignore[misc]


def test_unapproved_model_id_rejected():
    assert_config_invalid("model_id", model_id="some-org/other-model")


@pytest.mark.parametrize("attn", ["", " ", " sdpa", "sdpa ", None, 1])
def test_invalid_attention_request(attn):
    assert_config_invalid("attn_implementation", attn_implementation=attn)


def test_invalid_fixture_identity():
    assert_config_invalid("fixture_id", fixture_id="synthetic-sentiment-v2")
    assert_config_invalid("fixture_seed", fixture_seed=7)


@pytest.mark.parametrize("threads", [0, -1, -32, 12.0, 0.5, "12", True, False, None])
def test_invalid_cpu_threads(threads):
    assert_config_invalid("cpu_threads", cpu_threads=threads)


def test_invalid_config_message_is_human_readable():
    err = assert_config_invalid("batch_size", batch_size=3)
    assert "[1, 4, 8, 16, 32]" in err.message
    assert "got 3" in err.message
    assert str(err).startswith("[CONFIG_INVALID] ")


# --- serialization ------------------------------------------------------------


def test_to_dict_contains_every_architecture_field():
    expected = {
        "model_id",
        "model_revision",
        "device",
        "batch_size",
        "sequence_length",
        "precision",
        "matmul_precision",
        "attn_implementation",
        "warmup_iterations",
        "measured_iterations_per_trial",
        "trials",
        "memory_profile_forwards",
        "fixture_id",
        "fixture_seed",
        "cpu_threads",
    }
    assert set(make().to_dict()) == expected


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_dict_and_json_round_trip(device):
    cfg = make(device=device, batch_size=16, cpu_threads=8, attn_implementation="eager")
    assert BenchmarkConfig.from_dict(cfg.to_dict()) == cfg
    assert BenchmarkConfig.from_json(cfg.to_json()) == cfg
    assert json.loads(cfg.to_json()) == cfg.to_dict()


def test_saved_config_reloads_identically_on_a_different_machine(monkeypatch):
    # Saved on a synthetic 64-thread machine with an explicit 32 threads...
    fake_cpu_detection(monkeypatch, physical=32, logical=64)
    saved = make(cpu_threads=32).to_json()
    # ...reloaded on a synthetic 2-core machine: still exactly 32.
    fake_cpu_detection(monkeypatch, physical=2, logical=2)
    reloaded = BenchmarkConfig.from_json(saved)
    assert reloaded.cpu_threads == 32
    assert reloaded.to_json() == saved


def test_resolved_default_is_saved_and_not_rederived_on_reload(monkeypatch):
    fake_cpu_detection(monkeypatch, physical=6, logical=12)
    cfg = make()  # default resolved to a concrete 6
    saved = cfg.to_dict()
    assert saved["cpu_threads"] == 6
    fake_cpu_detection(monkeypatch, physical=24, logical=48)
    assert BenchmarkConfig.from_dict(saved).cpu_threads == 6
    assert make().cpu_threads == 24  # a *new* config uses the new machine


def test_from_dict_requires_cpu_threads():
    data = make().to_dict()
    del data["cpu_threads"]  # a saved config is never silently re-defaulted
    with pytest.raises(InferenceLabError) as excinfo:
        BenchmarkConfig.from_dict(data)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert "cpu_threads" in excinfo.value.message


def test_json_is_deterministic():
    assert make().to_json() == make().to_json()


def test_from_dict_rejects_missing_and_unknown_fields():
    data = make().to_dict()

    missing = {k: v for k, v in data.items() if k != "trials"}
    with pytest.raises(InferenceLabError) as excinfo:
        BenchmarkConfig.from_dict(missing)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert "trials" in excinfo.value.message

    with pytest.raises(InferenceLabError) as excinfo:
        BenchmarkConfig.from_dict({**data, "surprise": 1})
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert "surprise" in excinfo.value.message


def test_from_dict_revalidates_values():
    data = make().to_dict()
    data["batch_size"] = 3
    with pytest.raises(InferenceLabError) as excinfo:
        BenchmarkConfig.from_dict(data)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("text", ["not json", "", "[1, 2]", "null"])
def test_from_json_rejects_bad_input(text):
    with pytest.raises(InferenceLabError) as excinfo:
        BenchmarkConfig.from_json(text)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
