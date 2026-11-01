"""Offline unit tests for ``inference_lab.models``.

Everything is mocked or synthetic: fake ``from_pretrained`` functions, tiny
temporary "snapshot" directories and a faked ``torch.cuda``. No network, no
model download, no cached weights and no CUDA hardware are needed. The fake
file contents are placeholders, not real model data.
"""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from inference_lab import models
from inference_lab.config import PINNED_MODEL_REVISION, BenchmarkConfig
from inference_lab.errors import ErrorCode, InferenceLabError

FAKE_WEIGHTS = b"synthetic placeholder, not real model weights"
FAKE_VOCAB = b"[PAD]\n[UNK]\nthe\n"


def make_config(**overrides) -> BenchmarkConfig:
    values = {"device": "cpu", "batch_size": 8}
    values.update(overrides)
    return BenchmarkConfig(**values)


@pytest.fixture
def snapshot_dir(tmp_path, monkeypatch):
    """A fake cache snapshot named after the pinned revision."""
    path = tmp_path / "snapshots" / PINNED_MODEL_REVISION
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(FAKE_WEIGHTS)
    (path / "tokenizer_config.json").write_text("{}")
    (path / "vocab.txt").write_bytes(FAKE_VOCAB)
    monkeypatch.setattr(
        models, "EXPECTED_SAFETENSORS_SHA256", hashlib.sha256(FAKE_WEIGHTS).hexdigest()
    )
    return path


@pytest.fixture
def downloads(monkeypatch, snapshot_dir):
    """Replace snapshot_download; record its keyword arguments."""
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(models, "snapshot_download", fake_snapshot_download)
    return calls


class FakeModel(torch.nn.Module):
    def __init__(self, attn="sdpa", dtype=torch.float32):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(_attn_implementation=attn)


def install_fake_auto(monkeypatch, model=None, error=None):
    calls = []

    class FakeAuto:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            if error is not None:
                raise error
            return model

    monkeypatch.setattr(models, "AutoModelForSequenceClassification", FakeAuto)
    return calls


# --- pinned revision ------------------------------------------------------------


def test_non_pinned_revision_is_refused_before_any_lookup(downloads):
    cfg = make_config(model_revision="0123456789abcdef" * 2 + "01234567")
    for call in (models.locate_snapshot, models.load_model, models.load_tokenizer):
        with pytest.raises(InferenceLabError) as excinfo:
            call(cfg)
        assert excinfo.value.code is ErrorCode.CONFIG_INVALID
        assert "model_revision" in excinfo.value.message
    assert downloads == []  # nothing was looked up or downloaded


def test_pinned_revision_is_accepted():
    assert models.require_pinned_revision(make_config()) is None


# --- snapshot location ----------------------------------------------------------


def test_locate_snapshot_is_cache_only_by_default(downloads, snapshot_dir):
    assert models.locate_snapshot(make_config()) == snapshot_dir
    assert len(downloads) == 1
    call = downloads[0]
    assert call["local_files_only"] is True
    assert call["repo_id"] == make_config().model_id
    assert call["revision"] == PINNED_MODEL_REVISION
    assert sorted(call["allow_patterns"]) == sorted(models.REQUIRED_MODEL_FILES)


def test_network_is_only_possible_by_explicit_argument(downloads):
    models.locate_snapshot(make_config(), local_files_only=False)
    assert downloads[0]["local_files_only"] is False


@pytest.mark.parametrize("error", [FileNotFoundError("not cached"), ValueError("bad")])
def test_uncached_model_gives_clear_load_error(monkeypatch, error):
    def raising(**kwargs):
        raise error

    monkeypatch.setattr(models, "snapshot_download", raising)
    with pytest.raises(InferenceLabError) as excinfo:
        models.locate_snapshot(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert "cache-only" in excinfo.value.message


def test_snapshot_directory_must_match_revision(monkeypatch, tmp_path):
    wrong = tmp_path / ("a" * 40)
    wrong.mkdir()
    monkeypatch.setattr(models, "snapshot_download", lambda **kw: str(wrong))
    with pytest.raises(InferenceLabError) as excinfo:
        models.locate_snapshot(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert "does not match" in excinfo.value.message


def test_missing_required_file_is_reported(downloads, snapshot_dir):
    (snapshot_dir / "vocab.txt").unlink()
    with pytest.raises(InferenceLabError) as excinfo:
        models.locate_snapshot(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert excinfo.value.details["missing"] == ["vocab.txt"]


def test_safetensors_checksum_mismatch_is_rejected(downloads, snapshot_dir):
    (snapshot_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(InferenceLabError) as excinfo:
        models.locate_snapshot(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert "checksum mismatch" in excinfo.value.message
    # The check can be skipped explicitly (e.g. tokenizer-only loading).
    assert models.locate_snapshot(make_config(), verify_checksum=False) == snapshot_dir


def test_approved_checksum_constant():
    # Owner-verified digest of model.safetensors at the pinned revision.
    assert models.EXPECTED_SAFETENSORS_SHA256 == (
        "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa"
    )


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    data = b"x" * 3_000_000  # larger than one read chunk
    path.write_bytes(data)
    assert models.file_sha256(path) == hashlib.sha256(data).hexdigest()


# --- unreadable files (checksum I/O errors) -------------------------------------


def make_unreadable(monkeypatch, filename, error=None):
    """Make reading ``filename`` fail like a locked or broken file."""
    error = error or PermissionError(13, "Permission denied")
    real_open = type(models.Path("x")).open

    def open_(self, *args, **kwargs):
        if self.name == filename:
            raise error
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(type(models.Path("x")), "open", open_)


def test_unreadable_safetensors_is_a_load_error_with_path(
    monkeypatch, downloads, snapshot_dir
):
    make_unreadable(monkeypatch, "model.safetensors")
    with pytest.raises(InferenceLabError) as excinfo:
        models.locate_snapshot(make_config())
    err = excinfo.value
    assert err.code is ErrorCode.MODEL_LOAD_FAILED
    assert "model.safetensors" in err.message
    assert "Permission denied" in err.message
    assert err.details["path"] == str(snapshot_dir / "model.safetensors")


def test_unreadable_safetensors_stops_load_model_before_the_loader(
    monkeypatch, downloads
):
    make_unreadable(monkeypatch, "model.safetensors", OSError(5, "I/O error"))
    calls = install_fake_auto(monkeypatch, FakeModel())
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_model(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert calls == []


def test_checksum_is_not_attempted_when_verification_is_off(
    monkeypatch, downloads, snapshot_dir
):
    make_unreadable(monkeypatch, "model.safetensors")
    assert models.locate_snapshot(make_config(), verify_checksum=False) == snapshot_dir


def test_unreadable_vocab_is_a_load_error_with_path(
    monkeypatch, downloads, snapshot_dir
):
    install_fake_tokenizer(monkeypatch)
    make_unreadable(monkeypatch, "vocab.txt")
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_tokenizer(make_config())
    err = excinfo.value
    assert err.code is ErrorCode.MODEL_LOAD_FAILED
    assert "vocab.txt" in err.message
    assert err.details["path"] == str(snapshot_dir / "vocab.txt")


def test_hash_helper_wraps_any_oserror(monkeypatch, tmp_path):
    def boom(path):
        raise OSError(22, "device error")

    monkeypatch.setattr(models, "file_sha256", boom)
    with pytest.raises(InferenceLabError) as excinfo:
        models._hash_or_raise(tmp_path / "anything.bin")
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert "anything.bin" in excinfo.value.message


# --- model loading --------------------------------------------------------------


def test_load_model_uses_secure_pinned_arguments(monkeypatch, downloads, snapshot_dir):
    calls = install_fake_auto(monkeypatch, FakeModel())
    model, meta = models.load_model(make_config(attn_implementation="sdpa"))
    assert len(calls) == 1
    path, kwargs = calls[0]
    assert path == snapshot_dir  # pinned local snapshot, not a repo id / "main"
    assert kwargs == {
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
        "use_safetensors": True,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    assert model.training is False  # eval mode
    assert meta.resolved_revision == PINNED_MODEL_REVISION
    assert meta.requested_revision == PINNED_MODEL_REVISION
    assert meta.dtype == "float32"
    assert meta.requested_attn_implementation == meta.resolved_attn_implementation
    assert meta.safetensors_sha256_verified is True
    assert meta.parameter_count == 6  # Linear(2, 2): 4 weights + 2 biases
    assert meta.to_dict()["transformers_version"]


def test_load_model_stays_on_cpu(monkeypatch, downloads):
    install_fake_auto(monkeypatch, FakeModel())
    model, _ = models.load_model(make_config(device="cuda:0"))
    # Device placement is a separate, explicit step (protocol: CPU-first load).
    assert {p.device.type for p in model.parameters()} == {"cpu"}


def test_attention_backend_mismatch_is_an_error(monkeypatch, downloads):
    install_fake_auto(monkeypatch, FakeModel(attn="eager"))
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_model(make_config(attn_implementation="sdpa"))
    assert excinfo.value.code is ErrorCode.ATTENTION_BACKEND_MISMATCH
    assert excinfo.value.details == {"requested": "sdpa", "resolved": "eager"}


def test_non_fp32_weights_are_rejected(monkeypatch, downloads):
    install_fake_auto(monkeypatch, FakeModel(dtype=torch.float64))
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_model(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert "float32" in excinfo.value.message


@pytest.mark.parametrize(
    "error", [OSError("disk"), ValueError("bad attn"), RuntimeError("boom")]
)
def test_loader_failures_map_to_model_load_failed(monkeypatch, downloads, error):
    install_fake_auto(monkeypatch, error=error)
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_model(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED


def test_load_model_verifies_checksum_by_default(monkeypatch, downloads, snapshot_dir):
    (snapshot_dir / "model.safetensors").write_bytes(b"tampered")
    calls = install_fake_auto(monkeypatch, FakeModel())
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_model(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert calls == []  # tampered weights are never handed to the loader


# --- tokenizer loading ----------------------------------------------------------


def install_fake_tokenizer(monkeypatch, name="BertTokenizer", is_fast=True):
    calls = []

    class Fake:
        def __len__(self):
            return 3

    Fake.__name__ = name
    Fake.is_fast = is_fast

    class FakeFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return Fake()

    monkeypatch.setattr(models, "BertTokenizer", FakeFactory)
    return calls


def test_load_tokenizer_uses_same_pinned_snapshot(monkeypatch, downloads, snapshot_dir):
    calls = install_fake_tokenizer(monkeypatch)
    _tokenizer, meta = models.load_tokenizer(make_config())
    assert calls == [(snapshot_dir, {"local_files_only": True})]
    assert meta.resolved_revision == meta.requested_revision == PINNED_MODEL_REVISION
    assert meta.tokenizer_class == "BertTokenizer"
    assert meta.is_fast is True
    assert meta.vocab_size == 3
    assert meta.vocab_sha256 == hashlib.sha256(FAKE_VOCAB).hexdigest()
    assert meta.tokenizers_version and meta.transformers_version
    assert downloads[0]["local_files_only"] is True


@pytest.mark.parametrize(
    ("name", "is_fast"), [("SomeOtherTokenizer", True), ("BertTokenizer", False)]
)
def test_unexpected_tokenizer_backend_is_rejected(
    monkeypatch, downloads, name, is_fast
):
    install_fake_tokenizer(monkeypatch, name=name, is_fast=is_fast)
    with pytest.raises(InferenceLabError) as excinfo:
        models.load_tokenizer(make_config())
    assert excinfo.value.code is ErrorCode.MODEL_LOAD_FAILED


# --- device availability and placement ------------------------------------------


def fake_cuda(monkeypatch, available, count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_cpu_only_machine_lists_only_cpu(monkeypatch):
    fake_cuda(monkeypatch, available=False, count=0)
    assert models.available_devices() == ["cpu"]
    assert models.cuda_available() is False


def test_cuda_machine_lists_cuda_without_naming_hardware(monkeypatch):
    fake_cuda(monkeypatch, available=True, count=1)
    assert models.available_devices() == ["cpu", "cuda:0"]


def test_cuda_reported_but_no_devices_is_unavailable(monkeypatch):
    fake_cuda(monkeypatch, available=True, count=0)
    assert models.available_devices() == ["cpu"]


class FakeDeviceModel:
    def __init__(self, error=None):
        self.error = error
        self.device = "cpu"

    def to(self, device):
        if self.error is not None:
            raise self.error
        self.device = device
        return self

    def parameters(self):
        yield SimpleNamespace(device=torch.device(self.device))


def test_cpu_placement_never_touches_cuda(monkeypatch):
    def forbidden():
        raise AssertionError("CUDA must not be queried for a CPU placement")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    model = FakeDeviceModel()
    assert models.place_on_device(model, "cpu") is model
    assert model.device == "cpu"


def test_requested_cuda_unavailable_raises_and_does_not_fall_back(monkeypatch):
    fake_cuda(monkeypatch, available=False, count=0)
    model = FakeDeviceModel()
    with pytest.raises(InferenceLabError) as excinfo:
        models.place_on_device(model, "cuda:0")
    assert excinfo.value.code is ErrorCode.CUDA_UNAVAILABLE
    assert "not switched to the CPU" in excinfo.value.message
    assert model.device == "cpu"  # untouched


def test_cuda_placement_success(monkeypatch):
    fake_cuda(monkeypatch, available=True)
    model = FakeDeviceModel()
    assert models.place_on_device(model, "cuda:0") is model
    assert model.device == "cuda:0"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (torch.cuda.OutOfMemoryError("oom"), ErrorCode.CUDA_OOM),
        (RuntimeError("cuda failure"), ErrorCode.CUDA_RUNTIME_ERROR),
    ],
)
def test_cuda_placement_failures_use_specific_codes(monkeypatch, error, code):
    fake_cuda(monkeypatch, available=True)
    with pytest.raises(InferenceLabError) as excinfo:
        models.place_on_device(FakeDeviceModel(error=error), "cuda:0")
    assert excinfo.value.code is code


def test_model_left_on_wrong_device_is_detected(monkeypatch):
    fake_cuda(monkeypatch, available=True)

    class Stubborn(FakeDeviceModel):
        def to(self, device):
            return self  # claims success but stays on the CPU

    with pytest.raises(InferenceLabError) as excinfo:
        models.place_on_device(Stubborn(), "cuda:0")
    assert excinfo.value.code is ErrorCode.CUDA_RUNTIME_ERROR


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "gpu", "", None])
def test_unsupported_device_string_is_rejected(device):
    with pytest.raises(InferenceLabError) as excinfo:
        models.place_on_device(FakeDeviceModel(), device)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
