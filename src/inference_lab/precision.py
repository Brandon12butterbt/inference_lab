"""FP32 baseline precision controls with read-back verification.

Implements ``docs/BENCHMARK_PROTOCOL.md`` section 2.3 for PyTorch 2.11: float32
matmul precision ``highest`` and TF32 disabled for CUDA matmul and cuDNN, with
no reduced-precision fallback.

One coherent approach is used: the *legacy* controls the protocol names
(``torch.set_float32_matmul_precision``, ``torch.backends.cuda.matmul.allow_tf32``,
``torch.backends.cudnn.allow_tf32``). On the installed version they are set and
read back consistently without warnings. The newer ``fp32_precision`` attributes
are never *written*; they are only *read* as a safety net, and any value that
means reduced precision (for example ``"tf32"``) fails verification. Writing only
one of cuDNN's conv/RNN new-API flags makes the legacy cuDNN getter raise, which
is why the two APIs must not be mixed.

Note: PyTorch enables cuDNN TF32 by default (``cudnn.allow_tf32 is True``), so it
must be switched off explicitly; matmul TF32 is already off by default.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch

from inference_lab.config import MATMUL_PRECISION
from inference_lab.errors import ErrorCode, InferenceLabError

# New-API read-back attributes (informational; never written). "none" defers to
# the legacy flags and "ieee" is full FP32; anything else is reduced precision.
_READBACK = {
    "backends": ("backends",),
    "cuda.matmul": ("backends", "cuda", "matmul"),
    "cudnn": ("backends", "cudnn"),
    "cudnn.conv": ("backends", "cudnn", "conv"),
    "cudnn.rnn": ("backends", "cudnn", "rnn"),
    "mkldnn": ("backends", "mkldnn"),
}
_FULL_PRECISION_READBACKS = {"none", "ieee", "unavailable"}


@dataclass(frozen=True)
class PrecisionSettings:
    """Effective FP32 precision settings, as read back from PyTorch."""

    api: str
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    fp32_precision_readback: dict[str, str]
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _error(message: str, details: dict[str, Any] | None = None) -> InferenceLabError:
    return InferenceLabError(ErrorCode.CONFIG_INVALID, message, details)


def _readback(path: tuple[str, ...]) -> str:
    obj: Any = torch
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return "unavailable"
    value = getattr(obj, "fp32_precision", None)
    return "unavailable" if value is None else str(value)


def read_precision_settings() -> PrecisionSettings:
    """Read the effective settings (legacy getters plus new-API read-back)."""
    try:
        matmul_precision = torch.get_float32_matmul_precision()
        matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    except (RuntimeError, AttributeError) as exc:
        raise _error(
            f"Cannot read the FP32 precision settings: {exc}. A mix of the legacy "
            "and new PyTorch TF32 APIs may be in use."
        ) from None
    return PrecisionSettings(
        api="legacy",
        float32_matmul_precision=matmul_precision,
        cuda_matmul_allow_tf32=matmul_tf32,
        cudnn_allow_tf32=cudnn_tf32,
        fp32_precision_readback={k: _readback(p) for k, p in _READBACK.items()},
        torch_version=torch.__version__,
    )


def verify_fp32_baseline(
    settings: PrecisionSettings | None = None,
) -> PrecisionSettings:
    """Raise ``CONFIG_INVALID`` unless the settings are the FP32 baseline."""
    settings = read_precision_settings() if settings is None else settings
    problems = []
    if settings.float32_matmul_precision != MATMUL_PRECISION:
        problems.append(
            f"float32_matmul_precision is {settings.float32_matmul_precision!r}, "
            f"expected {MATMUL_PRECISION!r}"
        )
    if settings.cuda_matmul_allow_tf32:
        problems.append("CUDA matmul TF32 is enabled")
    if settings.cudnn_allow_tf32:
        problems.append("cuDNN TF32 is enabled")
    reduced = {
        name: value
        for name, value in settings.fp32_precision_readback.items()
        if value not in _FULL_PRECISION_READBACKS
    }
    if reduced:
        problems.append(
            f"reduced-precision fp32_precision settings are active: {reduced}"
        )
    if problems:
        raise _error(
            "FP32 baseline precision is not in effect: " + "; ".join(problems) + ".",
            {"problems": problems, "settings": settings.to_dict()},
        )
    return settings


def apply_fp32_baseline() -> PrecisionSettings:
    """Set the FP32 baseline (legacy controls only), read back and verify it.

    Process-global: it changes PyTorch's global flags. Returns the verified
    effective settings so they can be recorded with the experiment.
    """
    try:
        torch.set_float32_matmul_precision(MATMUL_PRECISION)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except (RuntimeError, AttributeError, TypeError) as exc:
        raise _error(f"Cannot set the FP32 baseline precision: {exc}.") from None
    return verify_fp32_baseline()
