"""Load and verify the committed fixture manifest.

The manifest (``fixtures/synthetic_sentiment_v1_manifest.json``) records the
approved identity of the fixture: text/token/mask hashes, token-length facts and
the pinned model and tokenizer identity (``docs/BENCHMARK_PROTOCOL.md`` 2.1).
``verify_fixture`` is the check the benchmark engine calls before every run.

Verification only reads. A failed check raises ``FIXTURE_INTEGRITY_FAILED``; the
manifest is never regenerated or rewritten.
"""

import json
import re
from importlib.resources import files
from typing import Any

from inference_lab.config import (
    FIXTURE_ID,
    FIXTURE_SEED,
    MODEL_ID,
    PINNED_MODEL_REVISION,
    SEQUENCE_LENGTH,
)
from inference_lab.dataset import (
    FIXTURE_SIZE,
    GENERATOR_VERSION,
    generate_texts,
    text_sha256,
)
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.models import TokenizerMetadata
from inference_lab.tokenization import TokenizedFixture, tokenize_fixture

MANIFEST_RESOURCE = "synthetic_sentiment_v1_manifest.json"
MANIFEST_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_HASH_KEYS = {"text_sha256", "token_ids_sha256", "attention_mask_sha256"}
_TOKENIZER_KEYS = {
    "class",
    "is_fast",
    "vocab_size",
    "vocab_sha256",
    "transformers_version",
    "tokenizers_version",
}
_LENGTH_KEYS = {"min", "median", "max", "truncated_count"}
_TOP_KEYS = {
    "manifest_version",
    "fixture_id",
    "seed",
    "generator_version",
    "text_count",
    "sequence_length",
    "hashes",
    "model",
    "tokenizer",
    "token_lengths",
}


def _fail(message: str, details: dict[str, Any] | None = None) -> InferenceLabError:
    return InferenceLabError(ErrorCode.FIXTURE_INTEGRITY_FAILED, message, details)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _object(manifest: dict[str, Any], key: str, keys: set[str]) -> dict[str, Any]:
    section = manifest.get(key)
    if not isinstance(section, dict) or set(section) != keys:
        raise _fail(
            f"Manifest section '{key}' must be an object with exactly {sorted(keys)}.",
            {"section": key},
        )
    return section


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Check the manifest's shape, types and fixed identity; return it unchanged."""
    if not isinstance(manifest, dict) or set(manifest) != _TOP_KEYS:
        raise _fail(
            f"Manifest must be an object with exactly the keys {sorted(_TOP_KEYS)}.",
            {
                "missing": sorted(_TOP_KEYS - set(manifest))
                if isinstance(manifest, dict)
                else None,
                "unexpected": sorted(set(manifest) - _TOP_KEYS)
                if isinstance(manifest, dict)
                else None,
            },
        )
    expected_identity = {
        "manifest_version": MANIFEST_VERSION,
        "fixture_id": FIXTURE_ID,
        "seed": FIXTURE_SEED,
        "generator_version": GENERATOR_VERSION,
        "text_count": FIXTURE_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
    }
    for key, expected in expected_identity.items():
        value = manifest[key]
        if type(value) is not type(expected) or value != expected:
            raise _fail(
                f"Manifest {key} must be {expected!r} (got {value!r}).",
                {"field": key, "expected": expected, "actual": value},
            )

    hashes = _object(manifest, "hashes", _HASH_KEYS)
    for key, value in hashes.items():
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise _fail(
                f"Manifest hash '{key}' must be 64 lowercase hex characters.",
                {"field": f"hashes.{key}", "actual": value},
            )

    model = _object(manifest, "model", {"model_id", "revision"})
    if model["model_id"] != MODEL_ID or model["revision"] != PINNED_MODEL_REVISION:
        raise _fail(
            "Manifest model identity does not match the approved pinned checkpoint.",
            {"actual": model},
        )

    tokenizer = _object(manifest, "tokenizer", _TOKENIZER_KEYS)
    valid_tokenizer = (
        isinstance(tokenizer["class"], str)
        and type(tokenizer["is_fast"]) is bool
        and _is_int(tokenizer["vocab_size"])
        and tokenizer["vocab_size"] > 0
        and isinstance(tokenizer["vocab_sha256"], str)
        and _SHA256.fullmatch(tokenizer["vocab_sha256"]) is not None
        and isinstance(tokenizer["transformers_version"], str)
        and bool(tokenizer["transformers_version"])
        and isinstance(tokenizer["tokenizers_version"], str)
        and bool(tokenizer["tokenizers_version"])
    )
    if not valid_tokenizer:
        raise _fail("Manifest tokenizer section has malformed values.", dict(tokenizer))

    lengths = _object(manifest, "token_lengths", _LENGTH_KEYS)
    valid_lengths = (
        _is_int(lengths["min"])
        and _is_int(lengths["max"])
        and _is_number(lengths["median"])
        and _is_int(lengths["truncated_count"])
        and 1 <= lengths["min"] <= lengths["median"] <= lengths["max"]
        and 0 <= lengths["truncated_count"] <= FIXTURE_SIZE
    )
    if not valid_lengths:
        raise _fail("Manifest token_lengths has malformed values.", dict(lengths))
    return manifest


def parse_manifest(text: str) -> dict[str, Any]:
    """Parse and validate manifest JSON text."""
    try:
        manifest = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise _fail(f"Manifest is not valid JSON: {exc}") from None
    return validate_manifest(manifest)


def load_manifest() -> dict[str, Any]:
    """Load and validate the committed manifest from the package."""
    resource = files("inference_lab").joinpath("fixtures", MANIFEST_RESOURCE)
    try:
        text = resource.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail(f"Cannot read the fixture manifest: {exc}") from None
    return parse_manifest(text)


def _raise_on_mismatch(what: str, checks: list[tuple[str, Any, Any]]) -> None:
    """Raise one error listing every (field, expected, actual) that differs."""
    mismatches = [
        {"field": field, "expected": expected, "actual": actual}
        for field, expected, actual in checks
        if type(expected) is not type(actual) or expected != actual
    ]
    if mismatches:
        summary = "; ".join(
            f"{m['field']}: expected {m['expected']!r}, got {m['actual']!r}"
            for m in mismatches
        )
        raise _fail(
            f"{what} does not match the committed manifest - {summary}.",
            {"mismatches": mismatches},
        )


def verify_text_hash(texts: list[str], manifest: dict[str, Any]) -> None:
    """Compare the fixture's text count and canonical text hash (no model needed)."""
    count = len(texts) if isinstance(texts, list) else None
    actual_hash = text_sha256(texts) if count is not None else None
    _raise_on_mismatch(
        "Fixture texts",
        [
            ("text_count", manifest["text_count"], count),
            ("hashes.text_sha256", manifest["hashes"]["text_sha256"], actual_hash),
        ],
    )


def verify_tokenizer_metadata(
    metadata: TokenizerMetadata, manifest: dict[str, Any]
) -> None:
    """Compare tokenizer identity: revision, class, backend, vocab, versions."""
    recorded = manifest["tokenizer"]
    revision = manifest["model"]["revision"]
    _raise_on_mismatch(
        "Tokenizer metadata",
        [
            ("model_id", manifest["model"]["model_id"], metadata.model_id),
            ("requested_revision", revision, metadata.requested_revision),
            ("resolved_revision", revision, metadata.resolved_revision),
            ("tokenizer.class", recorded["class"], metadata.tokenizer_class),
            ("tokenizer.is_fast", recorded["is_fast"], metadata.is_fast),
            ("tokenizer.vocab_size", recorded["vocab_size"], metadata.vocab_size),
            ("tokenizer.vocab_sha256", recorded["vocab_sha256"], metadata.vocab_sha256),
            (
                "tokenizer.transformers_version",
                recorded["transformers_version"],
                metadata.transformers_version,
            ),
            (
                "tokenizer.tokenizers_version",
                recorded["tokenizers_version"],
                metadata.tokenizers_version,
            ),
        ],
    )


def verify_tokenized_fixture(
    fixture: TokenizedFixture, manifest: dict[str, Any]
) -> None:
    """Compare token/mask hashes, token-length statistics and truncation count."""
    lengths = manifest["token_lengths"]
    _raise_on_mismatch(
        "Tokenized fixture",
        [
            ("sequence_length", manifest["sequence_length"], fixture.sequence_length),
            (
                "hashes.token_ids_sha256",
                manifest["hashes"]["token_ids_sha256"],
                fixture.token_ids_sha256,
            ),
            (
                "hashes.attention_mask_sha256",
                manifest["hashes"]["attention_mask_sha256"],
                fixture.attention_mask_sha256,
            ),
            ("token_lengths.min", lengths["min"], fixture.real_length_min),
            (
                "token_lengths.median",
                float(lengths["median"]),
                fixture.real_length_median,
            ),
            ("token_lengths.max", lengths["max"], fixture.real_length_max),
            (
                "token_lengths.truncated_count",
                lengths["truncated_count"],
                fixture.truncated_count,
            ),
        ],
    )


def verify_fixture(
    tokenizer: Any,
    metadata: TokenizerMetadata,
    texts: list[str] | None = None,
    manifest: dict[str, Any] | None = None,
) -> TokenizedFixture:
    """Full pre-run fixture check; returns the verified tokenized fixture.

    Order: manifest schema, text hash (cheap, no model), tokenizer identity, then
    tokenization and the tokenizer-dependent hashes and statistics. Any mismatch
    raises ``FIXTURE_INTEGRITY_FAILED``. ``texts`` and ``manifest`` default to the
    committed fixture and manifest; they are parameters only so tests can inject
    altered values.
    """
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    texts = generate_texts() if texts is None else texts
    verify_text_hash(texts, manifest)
    verify_tokenizer_metadata(metadata, manifest)
    fixture = tokenize_fixture(tokenizer, texts, manifest["sequence_length"])
    verify_tokenized_fixture(fixture, manifest)
    return fixture
