"""Unit tests for ``inference_lab.dataset`` (M1.2).

No model, tokenizer, network or hardware is involved. The generated fixture's
SHA-256 is deliberately NOT pinned here: it stays provisional until the owner
approves the fixture content (the official manifest is a later work package).
"""

import copy
import hashlib
import json
import unicodedata

import pytest

from inference_lab import dataset
from inference_lab.config import SUPPORTED_BATCH_SIZES
from inference_lab.dataset import (
    FIXTURE_SIZE,
    MEASURED_POSITIONS,
    MEMORY_PROFILE_POSITIONS,
    PARITY_POSITION,
    WARMUP_POSITIONS,
    SplitMix64,
    batch_indices,
    canonical_text_bytes,
    distinct_inputs_per_trial,
    generate_texts,
    load_source_data,
    text_sha256,
    validate_source,
)
from inference_lab.errors import ErrorCode, InferenceLabError


@pytest.fixture(scope="module")
def texts():
    return generate_texts()


@pytest.fixture(scope="module")
def source():
    return load_source_data()


# --- SplitMix64 ---------------------------------------------------------------


def test_splitmix64_matches_published_reference_vector():
    # First outputs of the SplitMix64 reference algorithm for seed 0.
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]


def test_splitmix64_is_deterministic_and_below_stays_in_range():
    a, b = SplitMix64(42), SplitMix64(42)
    assert [a.next_u64() for _ in range(20)] == [b.next_u64() for _ in range(20)]
    rng = SplitMix64(7)
    assert all(0 <= rng.below(5) < 5 for _ in range(1000))


# --- generated texts ----------------------------------------------------------


def test_exactly_512_nonempty_distinct_texts(texts):
    assert len(texts) == FIXTURE_SIZE == 512
    assert all(isinstance(t, str) and t.strip() == t and t for t in texts)
    assert len(set(texts)) == 512


def test_texts_are_nfc_ascii_and_end_with_a_period(texts):
    for t in texts:
        assert unicodedata.normalize("NFC", t) == t
        assert t.endswith(".")
        assert t.isascii()


APPROVED_TEXT_SHA256 = (
    "176cea2eb9a132a7647ab2632f8061448042477fa7a040cd02721e5d38030d2f"
)


def test_default_fixture_matches_the_owner_approved_text_hash(texts):
    # Fails on ANY change to vocabulary, templates, generator logic, seed or
    # ordering. Update only with owner approval and a new fixture version.
    assert text_sha256(texts) == APPROVED_TEXT_SHA256


def test_generation_is_stable_across_repeated_runs(texts):
    again = generate_texts()
    assert again == texts  # same content and same order
    assert text_sha256(again) == text_sha256(texts)
    assert generate_texts(42) == texts  # seed 42 is the default


def test_order_is_part_of_the_fixture_identity(texts):
    assert text_sha256(list(reversed(texts))) != text_sha256(texts)


def test_seed_sensitivity(texts):
    other = generate_texts(43)
    assert other != texts
    assert other[:3] != texts[:3]
    assert len(set(other)) == 512
    assert text_sha256(other) != text_sha256(texts)


@pytest.mark.parametrize("seed", [0, 1, 2**64 - 1])
def test_boundary_seeds_are_accepted(seed):
    assert len(generate_texts(seed)) == 512


@pytest.mark.parametrize("seed", [-1, 2**64, 4.2, 42.0, "42", None, True, False])
def test_invalid_seed_rejected(seed):
    with pytest.raises(InferenceLabError) as excinfo:
        generate_texts(seed)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert "seed" in excinfo.value.message


def test_texts_vary_in_sentence_count_and_length(texts, source):
    # Vocabulary contains no '.', so each period ends exactly one sentence.
    assert not any("." in w for w in _all_source_phrases(source))
    sentence_counts = {t.count(".") for t in texts}
    assert sentence_counts == {1, 2, 3, 4}
    word_counts = [len(t.split()) for t in texts]
    assert min(word_counts) <= 12
    assert max(word_counts) >= 50
    assert len(set(word_counts)) > 30


def test_texts_cover_the_source_vocabulary(texts, source):
    joined = "\n".join(texts)
    for key in ("subjects", "aspects", "openers"):
        missing = [p for p in source[key] if p not in joined]
        assert not missing, f"{key} never used: {missing}"
    for polarity in ("positive", "negative"):
        used = [p for p in source[polarity]["predicates"] if p in joined]
        assert len(used) == len(source[polarity]["predicates"])


def test_sentiment_expressions_of_both_polarities_appear(texts, source):
    positive = source["positive"]["predicates"]
    negative = source["negative"]["predicates"]
    assert any(any(p in t for p in positive) for t in texts)
    assert any(any(n in t for n in negative) for t in texts)
    # Contrast sentences ("..., but ...") mix both polarities in one text.
    assert any(
        any(p in t for p in positive) and any(n in t for n in negative) for t in texts
    )


def test_custom_valid_source_is_used(source):
    custom = copy.deepcopy(source)
    custom["subjects"] = [f"gadget number {i}" for i in range(30)]
    custom_texts = generate_texts(source=custom)
    assert len(set(custom_texts)) == 512
    assert any("gadget number" in t for t in custom_texts)
    assert custom_texts != generate_texts()


# --- source validation --------------------------------------------------------


def _expect_integrity_error(bad_source):
    with pytest.raises(InferenceLabError) as excinfo:
        validate_source(bad_source)
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED
    return excinfo.value


def test_committed_source_is_valid(source):
    validate_source(source)
    assert source["fixture_id"] == "synthetic-sentiment-v1"
    assert source["generator_version"] == dataset.GENERATOR_VERSION == 1


def _mutate(source, mutate):
    bad = copy.deepcopy(source)
    mutate(bad)
    return bad


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(fixture_id="synthetic-sentiment-v2"),
        lambda s: s.update(generator_version=2),
        lambda s: s.update(subjects=[]),
        lambda s: s.update(aspects="not a list"),
        lambda s: s.update(openers=["Honestly", "Honestly"]),
        lambda s: s.update(subjects=["laptop", " padded "]),
        lambda s: s.update(subjects=["laptop", ""]),
        lambda s: s.update(subjects=["laptop", 5]),
        lambda s: s.update(subjects=["café"]),  # NFD, not NFC
        lambda s: s["positive"].pop("closers"),
        lambda s: s.pop("negative"),
        lambda s: s.update(sentence_templates=["The {nonsense} {pred}."]),
        # Malformed syntax / unsupported formatting must fail validation, not
        # surface later as a bare ValueError/IndexError during generation.
        lambda s: s.update(sentence_templates=["The {subject {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject} {pred."]),
        lambda s: s.update(sentence_templates=["The {subject} pred}."]),
        lambda s: s.update(sentence_templates=["The {subject:!bad} {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject:>10} {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject!r} {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject:{aspect}} {pred}."]),
        lambda s: s.update(sentence_templates=["The {} {pred}."]),
        lambda s: s.update(sentence_templates=["The {0} {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject.upper} {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject[0]} {pred}."]),
        lambda s: s.update(sentence_templates=["The {subject} {pred}.", 5]),
    ],
)
def test_malformed_source_rejected(source, mutate):
    err = _expect_integrity_error(_mutate(source, mutate))
    assert str(err).startswith("[FIXTURE_INTEGRITY_FAILED] ")


@pytest.mark.parametrize(
    "template",
    ["The {subject {pred}.", "The {subject:!bad} {pred}.", "The {subject!r} {pred}."],
)
def test_malformed_template_fails_generation_with_integrity_error(source, template):
    bad = copy.deepcopy(source)
    bad["sentence_templates"] = [template]
    with pytest.raises(InferenceLabError) as excinfo:
        generate_texts(source=bad)
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED


def test_escaped_braces_are_literal_text_not_slots(source):
    ok = copy.deepcopy(source)
    ok["sentence_templates"] = ["The {{literal}} {subject} {pred}."]
    validate_source(ok)


def test_non_object_source_rejected():
    _expect_integrity_error([1, 2, 3])


def test_source_unable_to_yield_512_distinct_texts_fails(source):
    tiny = copy.deepcopy(source)
    tiny["aspects"] = ["price"]
    tiny["positive"]["predicates"] = ["was excellent"]
    tiny["negative"]["predicates"] = ["was disappointing"]
    tiny["sentence_templates"] = ["The {aspect} {pred}."]
    with pytest.raises(InferenceLabError) as excinfo:
        generate_texts(source=tiny)
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED
    assert "512 distinct" in excinfo.value.message


# --- canonical serialization and hashing --------------------------------------


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], b"[]"),
        (["a"], b'["a"]'),
        (["a", "b"], b'["a","b"]'),  # compact separators, no spaces
        (["café"], b'["caf\xc3\xa9"]'),  # UTF-8, not é escapes
        (['say "hi"'], b'["say \\"hi\\""]'),
        (["line\nbreak"], b'["line\\nbreak"]'),
    ],
)
def test_canonical_bytes_from_independently_written_expectations(items, expected):
    assert canonical_text_bytes(items) == expected


def test_canonical_bytes_apply_nfc_normalization():
    nfd = "café"  # 'e' + combining acute accent
    nfc = "café"
    assert nfd != nfc
    assert canonical_text_bytes([nfd]) == canonical_text_bytes([nfc])
    assert canonical_text_bytes([nfd]) == b'["caf\xc3\xa9"]'


def test_canonical_bytes_have_no_trailing_newline_or_ascii_escaping(texts):
    data = canonical_text_bytes(texts)
    assert not data.endswith(b"\n")
    assert data.startswith(b'["') and data.endswith(b'."]')
    assert b"\\u" not in data


def test_canonical_bytes_match_an_independently_built_encoding(texts):
    manual = "[" + ",".join(json.dumps(t, ensure_ascii=False) for t in texts) + "]"
    assert canonical_text_bytes(texts) == manual.encode("utf-8")


def test_sha256_known_value_and_format():
    assert (
        text_sha256([])
        == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )
    digest = text_sha256(["a", "b"])
    assert digest == hashlib.sha256(b'["a","b"]').hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()


def test_fixture_hash_is_sha256_of_independently_built_bytes(texts):
    manual = "[" + ",".join(json.dumps(t, ensure_ascii=False) for t in texts) + "]"
    assert text_sha256(texts) == hashlib.sha256(manual.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("bad", [None, "abc", ["a", 1], [b"a"], ("a", "b")])
def test_canonical_bytes_reject_non_string_lists(bad):
    with pytest.raises(InferenceLabError) as excinfo:
        canonical_text_bytes(bad)
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED


# --- batch indexing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_size", "position", "expected"),
    [
        (1, 0, [0]),
        (1, 10, [10]),
        (1, 59, [59]),
        (1, 511, [511]),
        (1, 512, [0]),  # wraps
        (4, 0, [0, 1, 2, 3]),
        (4, 10, [40, 41, 42, 43]),
        (4, 59, [236, 237, 238, 239]),
        (8, 10, list(range(80, 88))),
        (8, 63, list(range(504, 512))),
        (8, 64, list(range(8))),  # wraps
        (16, 31, list(range(496, 512))),
        (16, 32, list(range(16))),  # wraps
        (16, 59, list(range(432, 448))),
        (32, 10, list(range(320, 352))),
        (32, 15, list(range(480, 512))),
        (32, 16, list(range(32))),  # wraps
        (32, 59, list(range(352, 384))),
    ],
)
def test_batch_indices_at_known_positions(batch_size, position, expected):
    assert batch_indices(batch_size, position) == expected


@pytest.mark.parametrize("batch_size", SUPPORTED_BATCH_SIZES)
def test_batch_indices_follow_the_protocol_formula(batch_size):
    for position in (0, 1, 9, 10, 59, 60, 100, 511, 512, 1000):
        expected = [(position * batch_size + j) % 512 for j in range(batch_size)]
        assert batch_indices(batch_size, position) == expected
        assert len(batch_indices(batch_size, position)) == batch_size
        assert all(0 <= i < 512 for i in batch_indices(batch_size, position))


@pytest.mark.parametrize("batch_size", SUPPORTED_BATCH_SIZES)
def test_one_full_cycle_covers_each_fixture_index_exactly_once(batch_size):
    cycle = [i for k in range(512 // batch_size) for i in batch_indices(batch_size, k)]
    assert cycle == list(range(512))


def test_protocol_position_sets():
    assert list(WARMUP_POSITIONS) == list(range(10))
    assert list(MEASURED_POSITIONS) == list(range(10, 60))
    assert list(MEMORY_PROFILE_POSITIONS) == list(range(10, 20))
    assert PARITY_POSITION == 10
    assert set(MEMORY_PROFILE_POSITIONS) <= set(MEASURED_POSITIONS)
    assert PARITY_POSITION in MEASURED_POSITIONS
    assert not set(WARMUP_POSITIONS) & set(MEASURED_POSITIONS)


@pytest.mark.parametrize("batch_size", SUPPORTED_BATCH_SIZES)
def test_replaying_positions_gives_identical_batches(batch_size):
    first = [batch_indices(batch_size, k) for k in MEASURED_POSITIONS]
    for _trial in range(3):  # every trial replays the same 50 batches
        assert [batch_indices(batch_size, k) for k in MEASURED_POSITIONS] == first
    assert len(first) == 50
    # Returned lists are fresh copies: mutating one cannot affect later calls.
    first[0].append(-1)
    assert batch_indices(batch_size, MEASURED_POSITIONS[0]) != first[0]


@pytest.mark.parametrize(
    ("batch_size", "expected"), [(1, 50), (4, 200), (8, 400), (16, 512), (32, 512)]
)
def test_distinct_inputs_per_trial(batch_size, expected):
    assert distinct_inputs_per_trial(batch_size) == expected
    assert expected == min(512, 50 * batch_size)
    # Cross-check against the indices the 50 measured batches really touch.
    touched = {i for k in MEASURED_POSITIONS for i in batch_indices(batch_size, k)}
    assert len(touched) == expected


@pytest.mark.parametrize("batch_size", [0, 2, 64, -1, 8.0, "8", True, None])
def test_invalid_batch_size_rejected(batch_size):
    for call in (batch_indices, distinct_inputs_per_trial):
        args = (batch_size, 0) if call is batch_indices else (batch_size,)
        with pytest.raises(InferenceLabError) as excinfo:
            call(*args)
        assert excinfo.value.code is ErrorCode.CONFIG_INVALID
        assert "batch_size" in excinfo.value.message


@pytest.mark.parametrize("position", [-1, 1.0, "1", None, True])
def test_invalid_position_rejected(position):
    with pytest.raises(InferenceLabError) as excinfo:
        batch_indices(8, position)
    assert excinfo.value.code is ErrorCode.CONFIG_INVALID
    assert "position" in excinfo.value.message


# --- helpers ------------------------------------------------------------------


def _all_source_phrases(source):
    for key in ("subjects", "aspects", "openers"):
        yield from source[key]
    for polarity in ("positive", "negative"):
        for key in ("predicates", "adjectives", "closers"):
            yield from source[polarity][key]
