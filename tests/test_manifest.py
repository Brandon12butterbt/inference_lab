"""Offline unit tests for ``inference_lab.manifest``.

No network, model weights, tokenizer files or CUDA are used: tokenizer-dependent
checks run on synthetic/mocked ``TokenizerMetadata`` and ``TokenizedFixture``
objects built from the committed manifest values. The literal hashes asserted
below are the owner-approved values obtained from the real pinned tokenizer.
"""

import copy
import json
from importlib.resources import files

import pytest
import torch

from inference_lab import manifest as manifest_module
from inference_lab.dataset import generate_texts
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.manifest import (
    load_manifest,
    parse_manifest,
    validate_manifest,
    verify_fixture,
    verify_text_hash,
    verify_tokenized_fixture,
    verify_tokenizer_metadata,
)
from inference_lab.models import TokenizerMetadata
from inference_lab.tokenization import TokenizedFixture

APPROVED_TEXT_SHA256 = (
    "176cea2eb9a132a7647ab2632f8061448042477fa7a040cd02721e5d38030d2f"
)
APPROVED_TOKEN_IDS_SHA256 = (
    "dc9da83b57e7d32ad3bc53f2dff686b6745cacc20941d2b831d7d3dd65bf98e7"
)
APPROVED_ATTENTION_MASK_SHA256 = (
    "6e8fb30bd90854ce29410849f3112ed0686d021c0fc8f20c6dd9d1ada18e9e25"
)
APPROVED_VOCAB_SHA256 = (
    "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
)
PINNED = "714eb0fa89d2f80546fda750413ed43d93601a13"


@pytest.fixture
def manifest():
    return load_manifest()


def mutated(manifest, mutate):
    bad = copy.deepcopy(manifest)
    mutate(bad)
    return bad


def synthetic_metadata(manifest, **overrides) -> TokenizerMetadata:
    tok = manifest["tokenizer"]
    values = {
        "model_id": manifest["model"]["model_id"],
        "requested_revision": manifest["model"]["revision"],
        "resolved_revision": manifest["model"]["revision"],
        "tokenizer_class": tok["class"],
        "is_fast": tok["is_fast"],
        "vocab_size": tok["vocab_size"],
        "vocab_sha256": tok["vocab_sha256"],
        "transformers_version": tok["transformers_version"],
        "tokenizers_version": tok["tokenizers_version"],
    }
    values.update(overrides)
    return TokenizerMetadata(**values)


def synthetic_fixture(manifest, **overrides) -> TokenizedFixture:
    lengths = manifest["token_lengths"]
    values = {
        "input_ids": torch.zeros((512, 128), dtype=torch.int64),
        "attention_mask": torch.zeros((512, 128), dtype=torch.int64),
        "sequence_length": manifest["sequence_length"],
        "token_ids_sha256": manifest["hashes"]["token_ids_sha256"],
        "attention_mask_sha256": manifest["hashes"]["attention_mask_sha256"],
        "truncated_count": lengths["truncated_count"],
        "real_length_min": lengths["min"],
        "real_length_median": float(lengths["median"]),
        "real_length_max": lengths["max"],
    }
    values.update(overrides)
    return TokenizedFixture(**values)


def expect_integrity_error(call, *args, **kwargs):
    with pytest.raises(InferenceLabError) as excinfo:
        call(*args, **kwargs)
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED
    assert str(excinfo.value).startswith("[FIXTURE_INTEGRITY_FAILED] ")
    return excinfo.value


# --- committed manifest ---------------------------------------------------------


def test_committed_manifest_holds_the_approved_identity(manifest):
    assert manifest["manifest_version"] == 1
    assert manifest["fixture_id"] == "synthetic-sentiment-v1"
    assert manifest["seed"] == 42
    assert manifest["generator_version"] == 1
    assert manifest["text_count"] == 512
    assert manifest["sequence_length"] == 128
    assert manifest["hashes"] == {
        "text_sha256": APPROVED_TEXT_SHA256,
        "token_ids_sha256": APPROVED_TOKEN_IDS_SHA256,
        "attention_mask_sha256": APPROVED_ATTENTION_MASK_SHA256,
    }
    assert manifest["model"] == {
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "revision": PINNED,
    }
    assert manifest["tokenizer"] == {
        "class": "BertTokenizer",
        "is_fast": True,
        "vocab_size": 30522,
        "vocab_sha256": APPROVED_VOCAB_SHA256,
        "transformers_version": "5.22.1",
        "tokenizers_version": "0.23.3",
    }
    assert manifest["token_lengths"] == {
        "min": 7,
        "median": 33.0,
        "max": 98,
        "truncated_count": 0,
    }


def test_committed_manifest_text_hash_matches_generated_texts(manifest):
    verify_text_hash(generate_texts(), manifest)  # no model, no network


def test_validation_returns_the_manifest_unchanged(manifest):
    snapshot = copy.deepcopy(manifest)
    assert validate_manifest(manifest) is manifest
    assert manifest == snapshot


# --- schema and malformed values ------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.pop("hashes"),
        lambda m: m.pop("seed"),
        lambda m: m.update(extra_field=1),
        lambda m: m["hashes"].pop("token_ids_sha256"),
        lambda m: m["hashes"].update(extra="0" * 64),
        lambda m: m["tokenizer"].pop("vocab_sha256"),
        lambda m: m["model"].pop("revision"),
        lambda m: m["token_lengths"].pop("truncated_count"),
        lambda m: m.update(hashes="not an object"),
    ],
)
def test_missing_or_unexpected_fields_are_rejected(manifest, mutate):
    expect_integrity_error(validate_manifest, mutated(manifest, mutate))


@pytest.mark.parametrize(
    "bad_hash",
    ["", "abc", "A" * 64, "0" * 63, "0" * 65, "g" * 64, None, 123],
)
@pytest.mark.parametrize(
    "key", ["text_sha256", "token_ids_sha256", "attention_mask_sha256"]
)
def test_malformed_hashes_are_rejected(manifest, key, bad_hash):
    bad = mutated(manifest, lambda m: m["hashes"].update({key: bad_hash}))
    expect_integrity_error(validate_manifest, bad)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(fixture_id="synthetic-sentiment-v2"),
        lambda m: m.update(seed=43),
        lambda m: m.update(seed=True),
        lambda m: m.update(seed="42"),
        lambda m: m.update(generator_version=2),
        lambda m: m.update(text_count=511),
        lambda m: m.update(sequence_length=64),
        lambda m: m.update(manifest_version=2),
        lambda m: m["model"].update(revision="a" * 40),
        lambda m: m["model"].update(model_id="other/model"),
    ],
)
def test_altered_fixture_or_model_identity_is_rejected(manifest, mutate):
    expect_integrity_error(validate_manifest, mutated(manifest, mutate))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["tokenizer"].update(is_fast="yes"),
        lambda m: m["tokenizer"].update(vocab_size=0),
        lambda m: m["tokenizer"].update(vocab_size=30522.0),
        lambda m: m["tokenizer"].update(vocab_sha256="xyz"),
        lambda m: m["tokenizer"].update(transformers_version=""),
        lambda m: m["tokenizer"].update(tokenizers_version=None),
        lambda m: m["tokenizer"].update({"class": 5}),
        lambda m: m["token_lengths"].update(min=0),
        lambda m: m["token_lengths"].update(min=99),  # min > median
        lambda m: m["token_lengths"].update(max=10),  # max < median
        lambda m: m["token_lengths"].update(median="33"),
        lambda m: m["token_lengths"].update(truncated_count=-1),
        lambda m: m["token_lengths"].update(truncated_count=513),
        lambda m: m["token_lengths"].update(truncated_count=0.0),
    ],
)
def test_malformed_tokenizer_or_length_values_are_rejected(manifest, mutate):
    expect_integrity_error(validate_manifest, mutated(manifest, mutate))


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "null", "{}"])
def test_parse_manifest_rejects_bad_json(text):
    expect_integrity_error(parse_manifest, text)


def test_parse_manifest_accepts_the_committed_text(manifest):
    assert parse_manifest(json.dumps(manifest)) == manifest


def test_load_manifest_reports_unreadable_resource(monkeypatch):
    class Broken:
        def joinpath(self, *parts):
            return self

        def read_text(self, encoding=None):
            raise OSError("disk error")

    monkeypatch.setattr(manifest_module, "files", lambda package: Broken())
    err = expect_integrity_error(load_manifest)
    assert "disk error" in err.message


# --- text hash (no model) -------------------------------------------------------


def test_altered_text_content_is_detected(manifest):
    texts = generate_texts()
    texts[100] = texts[100] + " Extra."
    err = expect_integrity_error(verify_text_hash, texts, manifest)
    fields = [m["field"] for m in err.details["mismatches"]]
    assert fields == ["hashes.text_sha256"]
    assert err.details["mismatches"][0]["expected"] == APPROVED_TEXT_SHA256


def test_reordered_texts_are_detected(manifest):
    texts = generate_texts()
    texts[0], texts[1] = texts[1], texts[0]
    expect_integrity_error(verify_text_hash, texts, manifest)


def test_wrong_text_count_is_detected_with_both_mismatches(manifest):
    err = expect_integrity_error(verify_text_hash, generate_texts()[:511], manifest)
    assert {m["field"] for m in err.details["mismatches"]} == {
        "text_count",
        "hashes.text_sha256",
    }


def test_non_list_texts_are_rejected(manifest):
    expect_integrity_error(verify_text_hash, "not a list", manifest)


# --- tokenizer metadata ---------------------------------------------------------


def test_matching_tokenizer_metadata_passes(manifest):
    verify_tokenizer_metadata(synthetic_metadata(manifest), manifest)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"tokenizer_class": "OtherTokenizer"}, "tokenizer.class"),
        ({"is_fast": False}, "tokenizer.is_fast"),
        ({"vocab_size": 30000}, "tokenizer.vocab_size"),
        ({"vocab_sha256": "0" * 64}, "tokenizer.vocab_sha256"),
        ({"transformers_version": "5.23.0"}, "tokenizer.transformers_version"),
        ({"tokenizers_version": "0.24.0"}, "tokenizer.tokenizers_version"),
        ({"requested_revision": "a" * 40}, "requested_revision"),
        ({"resolved_revision": "b" * 40}, "resolved_revision"),
        ({"model_id": "other/model"}, "model_id"),
    ],
)
def test_altered_tokenizer_metadata_is_detected(manifest, override, field):
    meta = synthetic_metadata(manifest, **override)
    err = expect_integrity_error(verify_tokenizer_metadata, meta, manifest)
    assert [m["field"] for m in err.details["mismatches"]] == [field]
    assert field in err.message


def test_all_tokenizer_mismatches_are_reported_together(manifest):
    meta = synthetic_metadata(manifest, vocab_size=1, transformers_version="0.0.0")
    err = expect_integrity_error(verify_tokenizer_metadata, meta, manifest)
    assert len(err.details["mismatches"]) == 2


# --- tokenized fixture (synthetic results) --------------------------------------


def test_matching_tokenized_fixture_passes(manifest):
    verify_tokenized_fixture(synthetic_fixture(manifest), manifest)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"token_ids_sha256": "0" * 64}, "hashes.token_ids_sha256"),
        ({"attention_mask_sha256": "0" * 64}, "hashes.attention_mask_sha256"),
        ({"real_length_min": 6}, "token_lengths.min"),
        ({"real_length_median": 34.0}, "token_lengths.median"),
        ({"real_length_max": 99}, "token_lengths.max"),
        ({"truncated_count": 1}, "token_lengths.truncated_count"),
        ({"sequence_length": 64}, "sequence_length"),
    ],
)
def test_altered_tokenized_results_are_detected(manifest, override, field):
    fixture = synthetic_fixture(manifest, **override)
    err = expect_integrity_error(verify_tokenized_fixture, fixture, manifest)
    assert [m["field"] for m in err.details["mismatches"]] == [field]


# --- verify_fixture (the pre-run check) -----------------------------------------


@pytest.fixture
def stub_tokenization(monkeypatch, manifest):
    """Replace tokenize_fixture with a synthetic result; record calls."""
    calls = []
    result = {"fixture": synthetic_fixture(manifest)}

    def fake(tokenizer, texts, sequence_length):
        calls.append((tokenizer, len(texts), sequence_length))
        return result["fixture"]

    monkeypatch.setattr(manifest_module, "tokenize_fixture", fake)
    return calls, result


def test_verify_fixture_returns_the_verified_fixture(manifest, stub_tokenization):
    calls, result = stub_tokenization
    sentinel = object()
    out = verify_fixture(sentinel, synthetic_metadata(manifest))
    assert out is result["fixture"]
    assert calls == [(sentinel, 512, 128)]


def test_verify_fixture_fails_before_tokenizing_on_text_mismatch(
    manifest, stub_tokenization
):
    calls, _ = stub_tokenization
    texts = generate_texts()
    texts[5] = "tampered"
    expect_integrity_error(
        verify_fixture, object(), synthetic_metadata(manifest), texts
    )
    assert calls == []  # the cheap text check failed first


def test_verify_fixture_fails_on_tokenizer_metadata_mismatch(
    manifest, stub_tokenization
):
    calls, _ = stub_tokenization
    meta = synthetic_metadata(manifest, transformers_version="9.9.9")
    expect_integrity_error(verify_fixture, object(), meta)
    assert calls == []


def test_verify_fixture_fails_on_tokenized_hash_mismatch(manifest, stub_tokenization):
    _, result = stub_tokenization
    result["fixture"] = synthetic_fixture(manifest, token_ids_sha256="f" * 64)
    expect_integrity_error(verify_fixture, object(), synthetic_metadata(manifest))


def test_verify_fixture_rejects_a_corrupted_manifest(manifest, stub_tokenization):
    bad = mutated(manifest, lambda m: m["hashes"].update(text_sha256="oops"))
    calls, _ = stub_tokenization
    expect_integrity_error(
        verify_fixture, object(), synthetic_metadata(manifest), None, bad
    )
    assert calls == []


def test_failed_verification_never_rewrites_the_manifest(manifest, stub_tokenization):
    path = files("inference_lab").joinpath(
        "fixtures", "synthetic_sentiment_v1_manifest.json"
    )
    before = path.read_bytes()
    _, result = stub_tokenization
    result["fixture"] = synthetic_fixture(manifest, truncated_count=7)
    expect_integrity_error(verify_fixture, object(), synthetic_metadata(manifest))
    assert path.read_bytes() == before
