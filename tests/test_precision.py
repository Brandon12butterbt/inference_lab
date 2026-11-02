"""Offline unit tests for ``inference_lab.precision``.

The checks run against a fake ``torch`` namespace that mimics the PyTorch 2.11
defaults (cuDNN TF32 on, matmul TF32 off), so no CUDA hardware is needed. A few
tests also read/apply the real flags (restoring them afterwards).
"""

import json
from types import SimpleNamespace

import pytest
import torch

from inference_lab import precision
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.precision import (
    PrecisionSettings,
    apply_fp32_baseline,
    read_precision_settings,
    verify_fp32_baseline,
)


class FakeCudnn:
    """cuDNN flags; ``allow_tf32`` can be made to raise like a mixed-API state."""

    def __init__(self, allow_tf32=True, conv="none", rnn="none", mixed_error=False):
        self._allow_tf32 = allow_tf32
        self._mixed_error = mixed_error
        self.fp32_precision = "none"
        self.conv = SimpleNamespace(fp32_precision=conv)
        self.rnn = SimpleNamespace(fp32_precision=rnn)

    @property
    def allow_tf32(self):
        if self._mixed_error:
            raise RuntimeError("mix of the legacy and new APIs")
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value):
        self._allow_tf32 = value


class StubbornCudnn(FakeCudnn):
    """Ignores attempts to change ``allow_tf32``."""

    @property
    def allow_tf32(self):
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value):
        pass


def make_torch(
    matmul_precision="highest",
    matmul_tf32=False,
    cudnn=None,
    readbacks=None,
    set_error=None,
):
    """A fake ``torch`` module; defaults copy PyTorch 2.11 (cuDNN TF32 on)."""
    state = {"precision": matmul_precision}
    readbacks = readbacks or {}

    def set_precision(value):
        if set_error is not None:
            raise set_error
        state["precision"] = value

    return SimpleNamespace(
        __version__="2.11.0-fake",
        get_float32_matmul_precision=lambda: state["precision"],
        set_float32_matmul_precision=set_precision,
        backends=SimpleNamespace(
            fp32_precision=readbacks.get("backends", "none"),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(
                    allow_tf32=matmul_tf32,
                    fp32_precision=readbacks.get("cuda.matmul", "none"),
                )
            ),
            cudnn=cudnn if cudnn is not None else FakeCudnn(),
            mkldnn=SimpleNamespace(fp32_precision=readbacks.get("mkldnn", "none")),
        ),
    )


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        module = make_torch(**kwargs)
        monkeypatch.setattr(precision, "torch", module)
        return module

    return install


def expect_config_invalid(call, *args):
    with pytest.raises(InferenceLabError) as excinfo:
        call(*args)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    return excinfo.value


# --- verification ---------------------------------------------------------------


def test_pytorch_default_state_fails_because_cudnn_tf32_is_on(fake):
    fake()  # matmul highest/False already, but cuDNN TF32 defaults to True
    err = expect_config_invalid(verify_fp32_baseline)
    assert "cuDNN TF32 is enabled" in err.message
    assert any("cuDNN" in p for p in err.details["problems"])


def test_fully_baseline_state_passes(fake):
    fake(cudnn=FakeCudnn(allow_tf32=False))
    settings = verify_fp32_baseline()
    assert settings.float32_matmul_precision == "highest"
    assert settings.cuda_matmul_allow_tf32 is False
    assert settings.cudnn_allow_tf32 is False
    assert settings.api == "legacy"
    assert settings.torch_version == "2.11.0-fake"


def test_non_highest_matmul_precision_fails(fake):
    fake(matmul_precision="high", cudnn=FakeCudnn(allow_tf32=False))
    err = expect_config_invalid(verify_fp32_baseline)
    assert "'high'" in err.message and "'highest'" in err.message


def test_cuda_matmul_tf32_fails(fake):
    fake(matmul_tf32=True, cudnn=FakeCudnn(allow_tf32=False))
    err = expect_config_invalid(verify_fp32_baseline)
    assert "CUDA matmul TF32 is enabled" in err.message


@pytest.mark.parametrize("value", ["none", "ieee", "unavailable"])
def test_full_precision_readbacks_are_accepted(fake, value):
    fake(
        cudnn=FakeCudnn(allow_tf32=False, conv=value, rnn=value),
        readbacks={"backends": value, "cuda.matmul": value, "mkldnn": value},
    )
    verify_fp32_baseline()


@pytest.mark.parametrize(
    ("readbacks", "cudnn", "name"),
    [
        ({"cuda.matmul": "tf32"}, None, "cuda.matmul"),
        ({"mkldnn": "bf16"}, None, "mkldnn"),
        ({"backends": "tf32"}, None, "backends"),
        ({}, FakeCudnn(allow_tf32=False, conv="tf32"), "cudnn.conv"),
        ({}, FakeCudnn(allow_tf32=False, rnn="tf32"), "cudnn.rnn"),
    ],
)
def test_reduced_precision_new_api_readbacks_fail(fake, readbacks, cudnn, name):
    fake(readbacks=readbacks, cudnn=cudnn or FakeCudnn(allow_tf32=False))
    err = expect_config_invalid(verify_fp32_baseline)
    reduced = err.details["problems"][0]
    assert name in reduced
    assert "reduced-precision" in reduced


def test_mixed_api_state_that_breaks_the_legacy_getter_is_reported(fake):
    fake(cudnn=FakeCudnn(allow_tf32=False, mixed_error=True))
    err = expect_config_invalid(read_precision_settings)
    assert "mix" in err.message.lower()


def test_verify_accepts_explicit_settings_without_reading_torch(fake):
    fake(cudnn=FakeCudnn(allow_tf32=True))  # live state is bad ...
    good = PrecisionSettings(
        api="legacy",
        float32_matmul_precision="highest",
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        fp32_precision_readback={"cudnn.conv": "none"},
        torch_version="x",
    )
    assert verify_fp32_baseline(good) is good  # ... but explicit settings are checked
    bad = PrecisionSettings(**{**good.to_dict(), "cudnn_allow_tf32": True})
    expect_config_invalid(verify_fp32_baseline, bad)


# --- applying -------------------------------------------------------------------


def test_apply_sets_the_baseline_and_reads_it_back(fake):
    module = fake(matmul_precision="high")  # also repairs a non-highest start
    module.backends.cuda.matmul.allow_tf32 = True
    settings = apply_fp32_baseline()
    assert module.get_float32_matmul_precision() == "highest"
    assert module.backends.cuda.matmul.allow_tf32 is False
    assert module.backends.cudnn.allow_tf32 is False
    assert settings.float32_matmul_precision == "highest"
    assert settings.cuda_matmul_allow_tf32 is False
    assert settings.cudnn_allow_tf32 is False
    assert json.loads(json.dumps(settings.to_dict()))["api"] == "legacy"


def test_apply_does_not_write_the_new_api(fake):
    module = fake()
    apply_fp32_baseline()
    # The fp32_precision attributes are only ever read, never written.
    assert module.backends.cudnn.fp32_precision == "none"
    assert module.backends.cuda.matmul.fp32_precision == "none"
    assert module.backends.cudnn.conv.fp32_precision == "none"


def test_apply_fails_when_a_setting_does_not_stick(fake):
    fake(cudnn=StubbornCudnn(allow_tf32=True))
    err = expect_config_invalid(apply_fp32_baseline)
    assert "cuDNN TF32 is enabled" in err.message


def test_apply_reports_a_setter_that_raises(fake):
    fake(set_error=RuntimeError("cannot set"))
    err = expect_config_invalid(apply_fp32_baseline)
    assert "cannot set" in err.message


# --- the real PyTorch flags (restored afterwards) -------------------------------


@pytest.fixture
def restore_real_flags():
    saved = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    yield
    torch.set_float32_matmul_precision(saved[0])
    torch.backends.cuda.matmul.allow_tf32 = saved[1]
    torch.backends.cudnn.allow_tf32 = saved[2]


def test_real_settings_can_be_read_without_cuda_hardware():
    settings = read_precision_settings()
    assert settings.api == "legacy"
    assert isinstance(settings.cuda_matmul_allow_tf32, bool)
    assert isinstance(settings.cudnn_allow_tf32, bool)
    assert settings.torch_version == torch.__version__
    assert set(settings.fp32_precision_readback) == {
        "backends",
        "cuda.matmul",
        "cudnn",
        "cudnn.conv",
        "cudnn.rnn",
        "mkldnn",
    }
    json.dumps(settings.to_dict())


def test_real_apply_reaches_the_verified_baseline(restore_real_flags):
    settings = apply_fp32_baseline()
    assert settings.float32_matmul_precision == "highest"
    assert settings.cuda_matmul_allow_tf32 is False
    assert settings.cudnn_allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cudnn.allow_tf32 is False
    verify_fp32_baseline()
