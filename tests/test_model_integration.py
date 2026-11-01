"""OPT-IN integration tests against the real, already-cached model and tokenizer.

Skipped in the ordinary ``pytest`` run, even if the model happens to be cached.
To run them (only after the approved ~268 MB download has populated the
Hugging Face cache)::

    $env:INFERENCE_LAB_RUN_MODEL_INTEGRATION = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_model_integration.py -v -s

They are cache-only: ``conftest.py`` keeps the Hugging Face libraries offline
and blocks sockets, so they can never start a download. The token-ID and
attention-mask hashes asserted here are the owner-approved values recorded in
the committed fixture manifest.
"""

import os

import pytest
import torch

from inference_lab import models
from inference_lab.config import PINNED_MODEL_REVISION, BenchmarkConfig
from inference_lab.dataset import generate_texts, text_sha256
from inference_lab.manifest import load_manifest, verify_fixture
from inference_lab.tokenization import tokenize_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("INFERENCE_LAB_RUN_MODEL_INTEGRATION") != "1",
    reason="opt-in: set INFERENCE_LAB_RUN_MODEL_INTEGRATION=1 (cache-only, needs the "
    "approved model download)",
)

APPROVED_TEXT_SHA256 = (
    "176cea2eb9a132a7647ab2632f8061448042477fa7a040cd02721e5d38030d2f"
)
APPROVED_TOKEN_IDS_SHA256 = (
    "dc9da83b57e7d32ad3bc53f2dff686b6745cacc20941d2b831d7d3dd65bf98e7"
)
APPROVED_ATTENTION_MASK_SHA256 = (
    "6e8fb30bd90854ce29410849f3112ed0686d021c0fc8f20c6dd9d1ada18e9e25"
)


@pytest.fixture(scope="module")
def config():
    return BenchmarkConfig(device="cpu", batch_size=4)


def test_cached_snapshot_is_the_pinned_revision_with_verified_checksum(config):
    snapshot = models.locate_snapshot(config)
    assert snapshot.name == PINNED_MODEL_REVISION
    assert models.file_sha256(snapshot / "model.safetensors") == (
        models.EXPECTED_SAFETENSORS_SHA256
    )


def test_model_loads_fp32_eval_sdpa_on_cpu(config):
    model, meta = models.load_model(config)
    assert model.training is False
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert {p.device.type for p in model.parameters()} == {"cpu"}
    assert meta.resolved_attn_implementation == "sdpa"
    assert meta.resolved_revision == PINNED_MODEL_REVISION
    print("model metadata:", meta.to_dict())


def test_tokenizer_identity(config):
    _tokenizer, meta = models.load_tokenizer(config)
    assert meta.tokenizer_class == "BertTokenizer"
    assert meta.is_fast is True
    print("tokenizer metadata:", meta.to_dict())


def test_cpu_forward_returns_finite_logits(config):
    model, _ = models.load_model(config)
    tokenizer, _ = models.load_tokenizer(config)
    texts = generate_texts()[:4]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
    assert tuple(logits.shape) == (4, 2)
    assert bool(torch.isfinite(logits).all())


def test_real_fixture_tokenization_facts(config):
    """Real pinned tokenizer vs the committed (owner-approved) manifest."""
    tokenizer, meta = models.load_tokenizer(config)
    texts = generate_texts()
    assert text_sha256(texts) == APPROVED_TEXT_SHA256  # unchanged
    manifest = load_manifest()

    fixture = tokenize_fixture(tokenizer, texts)
    assert tuple(fixture.input_ids.shape) == (512, 128)
    assert tuple(fixture.attention_mask.shape) == (512, 128)
    assert fixture.input_ids.dtype == torch.int64
    assert fixture.token_ids_sha256 == APPROVED_TOKEN_IDS_SHA256
    assert fixture.attention_mask_sha256 == APPROVED_ATTENTION_MASK_SHA256
    assert fixture.truncated_count == 0
    assert fixture.real_length_min == 7
    assert fixture.real_length_median == 33.0
    assert fixture.real_length_max == 98
    assert manifest["hashes"]["token_ids_sha256"] == fixture.token_ids_sha256
    assert manifest["hashes"]["attention_mask_sha256"] == (
        fixture.attention_mask_sha256
    )

    # The pre-run check the benchmark engine will call: identity, hashes, stats.
    verified = verify_fixture(tokenizer, meta)
    assert verified.manifest_facts() == fixture.manifest_facts()
    print("verified tokenizer-dependent facts:", verified.manifest_facts())


def test_cuda_placement_keeps_the_pinned_model(config):
    # Checked here, after the module's opt-in gate, so importing this module
    # during ordinary collection never queries CUDA.
    if not models.cuda_available():
        pytest.skip("no CUDA device available on this machine")
    model, _ = models.load_model(config)
    moved = models.place_on_device(model, "cuda:0")
    assert {str(p.device) for p in moved.parameters()} == {"cuda:0"}
