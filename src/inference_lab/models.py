"""Cache-only, secure loading of the approved DistilBERT checkpoint.

Implements the model-loading part of ``docs/BENCHMARK_PROTOCOL.md`` section 3:
the pinned model is loaded on the CPU in FP32 and ``.eval()`` mode, and device
placement is a separate, explicit step. Nothing here falls back silently: a
different revision, attention backend, dtype or device is an error.

Loading never initiates a download unless ``local_files_only=False`` is passed
explicitly; populating the Hugging Face cache is a separate, authorized step.
Weights are read from ``model.safetensors`` only (never a pickle) and remote
code is never trusted.
"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tokenizers
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, BertTokenizer

from inference_lab.config import PINNED_MODEL_REVISION, BenchmarkConfig
from inference_lab.errors import ErrorCode, InferenceLabError

# SHA-256 of model.safetensors at the pinned revision (owner-verified; also the
# Hugging Face LFS object id).
EXPECTED_SAFETENSORS_SHA256 = (
    "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa"
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
)
EXPECTED_TOKENIZER_CLASS = "BertTokenizer"


@dataclass(frozen=True)
class ModelMetadata:
    """What was actually loaded, for the reproducibility record."""

    model_id: str
    requested_revision: str
    resolved_revision: str
    dtype: str
    requested_attn_implementation: str
    resolved_attn_implementation: str
    parameter_count: int
    safetensors_sha256_verified: bool
    transformers_version: str
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerMetadata:
    """Tokenizer identity: revision, class, backend and library versions."""

    model_id: str
    requested_revision: str
    resolved_revision: str
    tokenizer_class: str
    is_fast: bool
    vocab_size: int
    vocab_sha256: str
    transformers_version: str
    tokenizers_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_error(message: str, details: dict[str, Any] | None = None):
    return InferenceLabError(ErrorCode.MODEL_LOAD_FAILED, message, details)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Lowercase hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_or_raise(path: Path) -> str:
    """``file_sha256`` with I/O failures reported as ``MODEL_LOAD_FAILED``."""
    try:
        return file_sha256(path)
    except OSError as exc:
        raise _load_error(
            f"Cannot read {path} to compute its checksum: {exc}",
            {"path": str(path)},
        ) from None


def require_pinned_revision(config: BenchmarkConfig) -> None:
    """Refuse any revision other than the approved pinned checkpoint."""
    if config.model_revision != PINNED_MODEL_REVISION:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Invalid model_revision: the MVP baseline only loads the pinned "
            f"revision {PINNED_MODEL_REVISION} (got {config.model_revision!r}).",
            {"field": "model_revision", "value": repr(config.model_revision)},
        )


def locate_snapshot(
    config: BenchmarkConfig,
    *,
    local_files_only: bool = True,
    verify_checksum: bool = True,
) -> Path:
    """Return the local snapshot directory of the pinned revision.

    With the default ``local_files_only=True`` this reads the Hugging Face cache
    only. The directory name must equal the requested commit hash, the four
    required files must exist and, if ``verify_checksum``, ``model.safetensors``
    must match ``EXPECTED_SAFETENSORS_SHA256``.
    """
    require_pinned_revision(config)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.model_revision,
                local_files_only=local_files_only,
                allow_patterns=list(REQUIRED_MODEL_FILES),
            )
        )
    except (OSError, ValueError) as exc:
        hint = (
            " Ordinary loading is cache-only; downloading is a separate, "
            "explicitly authorized step."
            if local_files_only
            else ""
        )
        raise _load_error(
            f"Model {config.model_id} at revision {config.model_revision} is not "
            f"available: {exc}.{hint}"
        ) from None
    if snapshot.name != config.model_revision:
        raise _load_error(
            f"Resolved snapshot {snapshot.name!r} does not match the requested "
            f"revision {config.model_revision!r}."
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise _load_error(
            f"Snapshot {snapshot} is missing required files: {missing}.",
            {"missing": missing},
        )
    if verify_checksum:
        actual = _hash_or_raise(snapshot / "model.safetensors")
        if actual != EXPECTED_SAFETENSORS_SHA256:
            raise _load_error(
                "model.safetensors checksum mismatch: expected "
                f"{EXPECTED_SAFETENSORS_SHA256}, got {actual}.",
                {"expected": EXPECTED_SAFETENSORS_SHA256, "actual": actual},
            )
    return snapshot


def load_model(
    config: BenchmarkConfig,
    *,
    local_files_only: bool = True,
    verify_checksum: bool = True,
) -> tuple[Any, ModelMetadata]:
    """Load the pinned sequence-classification model on the CPU (FP32, eval)."""
    snapshot = locate_snapshot(
        config, local_files_only=local_files_only, verify_checksum=verify_checksum
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot,
            dtype=torch.float32,
            attn_implementation=config.attn_implementation,
            use_safetensors=True,
            trust_remote_code=False,
            local_files_only=True,  # the snapshot is already a local directory
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        raise _load_error(f"Failed to load model from {snapshot}: {exc}") from None

    wrong_dtype = sorted(
        {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    )
    if wrong_dtype:
        raise _load_error(
            f"Model parameters must all be torch.float32; found {wrong_dtype}.",
            {"dtypes": wrong_dtype},
        )
    resolved_attn = getattr(model.config, "_attn_implementation", None)
    if resolved_attn != config.attn_implementation:
        raise InferenceLabError(
            ErrorCode.ATTENTION_BACKEND_MISMATCH,
            f"Requested attention backend {config.attn_implementation!r} but the "
            f"model resolved {resolved_attn!r}; no silent substitution.",
            {"requested": config.attn_implementation, "resolved": str(resolved_attn)},
        )
    model.eval()
    if model.training:
        raise _load_error("Model is not in evaluation mode after .eval().")

    metadata = ModelMetadata(
        model_id=config.model_id,
        requested_revision=config.model_revision,
        resolved_revision=snapshot.name,
        dtype="float32",
        requested_attn_implementation=config.attn_implementation,
        resolved_attn_implementation=resolved_attn,
        parameter_count=sum(p.numel() for p in model.parameters()),
        safetensors_sha256_verified=verify_checksum,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
    )
    return model, metadata


def load_tokenizer(
    config: BenchmarkConfig, *, local_files_only: bool = True
) -> tuple[Any, TokenizerMetadata]:
    """Load the pinned revision's tokenizer (explicit ``BertTokenizer``).

    Transformers 5 has a single tokenizers-backed implementation for this class;
    the class and its fast-backend flag are verified and recorded rather than
    left to ``AutoTokenizer`` selection.
    """
    snapshot = locate_snapshot(
        config, local_files_only=local_files_only, verify_checksum=False
    )
    try:
        tokenizer = BertTokenizer.from_pretrained(snapshot, local_files_only=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        raise _load_error(f"Failed to load tokenizer from {snapshot}: {exc}") from None

    tokenizer_class = type(tokenizer).__name__
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if tokenizer_class != EXPECTED_TOKENIZER_CLASS or not is_fast:
        raise _load_error(
            f"Expected a fast {EXPECTED_TOKENIZER_CLASS}; got {tokenizer_class} "
            f"(is_fast={is_fast}).",
            {"tokenizer_class": tokenizer_class, "is_fast": is_fast},
        )
    metadata = TokenizerMetadata(
        model_id=config.model_id,
        requested_revision=config.model_revision,
        resolved_revision=snapshot.name,
        tokenizer_class=tokenizer_class,
        is_fast=is_fast,
        vocab_size=len(tokenizer),
        vocab_sha256=_hash_or_raise(snapshot / "vocab.txt"),
        transformers_version=transformers.__version__,
        tokenizers_version=tokenizers.__version__,
    )
    return tokenizer, metadata


# --- Device availability and placement (no hard-coded hardware) ----------------


def cuda_available() -> bool:
    """True only if PyTorch reports CUDA and at least one device."""
    return bool(torch.cuda.is_available()) and torch.cuda.device_count() >= 1


def available_devices() -> list[str]:
    """Configurable devices on this machine: always ``cpu``, plus ``cuda:0`` if usable."""
    devices = ["cpu"]
    if cuda_available():
        devices.append("cuda:0")
    return devices


def require_cuda_available(device: str = "cuda:0") -> None:
    """Raise ``CUDA_UNAVAILABLE`` unless a usable CUDA device exists (no fallback)."""
    if not cuda_available():
        raise InferenceLabError(
            ErrorCode.CUDA_UNAVAILABLE,
            f"CUDA device {device!r} was requested but no usable CUDA device is "
            "available on this machine. The run was not switched to the CPU.",
        )


def place_on_device(model: Any, device: str) -> Any:
    """Move ``model`` to the requested device; never fall back silently.

    ``cpu`` never touches CUDA. ``cuda:0`` raises ``CUDA_UNAVAILABLE`` when no
    usable CUDA device exists, ``CUDA_OOM`` on out-of-memory and
    ``CUDA_RUNTIME_ERROR`` for other CUDA failures.
    """
    if device == "cpu":
        return model.to("cpu")
    if device != "cuda:0":
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Invalid device: must be 'cpu' or 'cuda:0' (got {device!r}).",
            {"field": "device", "value": repr(device)},
        )
    require_cuda_available(device)
    try:
        moved = model.to(device)
    except torch.cuda.OutOfMemoryError as exc:
        raise InferenceLabError(
            ErrorCode.CUDA_OOM,
            f"Out of GPU memory moving the model to {device}: {exc}",
        ) from None
    except RuntimeError as exc:
        raise InferenceLabError(
            ErrorCode.CUDA_RUNTIME_ERROR,
            f"CUDA error moving the model to {device}: {exc}",
        ) from None
    actual = {str(p.device) for p in moved.parameters()}
    if actual != {device}:
        raise InferenceLabError(
            ErrorCode.CUDA_RUNTIME_ERROR,
            f"Model parameters are on {sorted(actual)} after requesting {device}.",
        )
    return moved
