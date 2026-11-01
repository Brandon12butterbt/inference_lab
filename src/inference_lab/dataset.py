"""Deterministic synthetic text fixture and batch indexing.

Implements ``docs/BENCHMARK_PROTOCOL.md`` sections 2.1 (fixture text generation
and canonical hashing) and 2.2 (batch sequencing). Tokenization and the
tokenizer-dependent manifest hashes are out of scope here. This module needs no
model, network access or hardware.

Reproducibility does not depend on Python's ``random`` module or any library
RNG: texts are generated from a committed source file with the explicit
SplitMix64 generator below. The source file, the seed and ``GENERATOR_VERSION``
together define the fixture; change any of them and ``FIXTURE_ID`` must change.
"""

import hashlib
import json
import string
import unicodedata
from importlib.resources import files
from typing import Any

from inference_lab.config import (
    FIXTURE_ID,
    FIXTURE_SEED,
    MEASURED_ITERATIONS_PER_TRIAL,
    MEMORY_PROFILE_FORWARDS,
    SUPPORTED_BATCH_SIZES,
    WARMUP_ITERATIONS,
)
from inference_lab.errors import ErrorCode, InferenceLabError

FIXTURE_SIZE = 512
GENERATOR_VERSION = 1
SOURCE_RESOURCE = "synthetic_sentiment_v1_source.json"

MAX_SENTENCES = 4
# Duplicate texts are skipped; this bounds the work if a source cannot yield 512.
_MAX_ATTEMPTS = FIXTURE_SIZE * 50

# Batch positions (BENCHMARK_PROTOCOL 2.2), derived from the protocol constants.
WARMUP_POSITIONS = range(WARMUP_ITERATIONS)
MEASURED_POSITIONS = range(
    WARMUP_ITERATIONS, WARMUP_ITERATIONS + MEASURED_ITERATIONS_PER_TRIAL
)
MEMORY_PROFILE_POSITIONS = range(
    WARMUP_ITERATIONS, WARMUP_ITERATIONS + MEMORY_PROFILE_FORWARDS
)
PARITY_POSITION = WARMUP_ITERATIONS

_MASK64 = (1 << 64) - 1
_LIST_KEYS = ("subjects", "aspects", "openers", "sentence_templates")
_POLARITY_KEYS = ("positive", "negative")  # index 0 = positive, 1 = negative
_POLARITY_LIST_KEYS = ("predicates", "adjectives", "closers")
_SLOTS = {
    "subject",
    "aspect",
    "opener",
    "pred",
    "pred2",
    "opposite_pred",
    "adj",
    "closer",
}


class SplitMix64:
    """SplitMix64 pseudo-random generator (explicit 64-bit integer arithmetic)."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Integer in ``[0, n)`` as ``next_u64() % n`` (bias below 2**-50 here)."""
        return self.next_u64() % n


def _integrity_error(message: str) -> InferenceLabError:
    return InferenceLabError(ErrorCode.FIXTURE_INTEGRITY_FAILED, message)


def _check_word_list(name: str, values: Any) -> None:
    if not isinstance(values, list) or not values:
        raise _integrity_error(f"Fixture source '{name}' must be a non-empty list.")
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise _integrity_error(
                f"Fixture source '{name}' has an invalid entry: {value!r}."
            )
        if unicodedata.normalize("NFC", value) != value:
            raise _integrity_error(
                f"Fixture source '{name}' entry is not NFC-normalized: {value!r}."
            )
    if len(set(values)) != len(values):
        raise _integrity_error(f"Fixture source '{name}' contains duplicate entries.")


def _template_slots(template: str) -> list[str]:
    """Slot names of a sentence template, in order of appearance.

    Only plain ``{name}`` fields with a known slot name are supported: malformed
    braces, conversions (``{x!r}``), format specs (``{x:>5}``) and empty or
    unknown names all raise ``FIXTURE_INTEGRITY_FAILED`` here, so generation
    never meets syntax it cannot handle.
    """
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise _integrity_error(
            f"Fixture template {template!r} is malformed: {exc}."
        ) from None
    slots = []
    for _, name, format_spec, conversion in parsed:
        if name is None:  # trailing literal text with no field
            continue
        if name not in _SLOTS:
            raise _integrity_error(
                f"Fixture template {template!r} uses unknown slot {name!r}."
            )
        if format_spec or conversion:
            raise _integrity_error(
                f"Fixture template {template!r} uses unsupported formatting "
                f"for slot {name!r}; only plain {{name}} fields are allowed."
            )
        slots.append(name)
    return slots


def validate_source(source: Any) -> None:
    """Raise ``FIXTURE_INTEGRITY_FAILED`` if the source data is malformed."""
    if not isinstance(source, dict):
        raise _integrity_error("Fixture source must be a JSON object.")
    if source.get("fixture_id") != FIXTURE_ID:
        raise _integrity_error(
            f"Fixture source fixture_id must be {FIXTURE_ID!r}, "
            f"got {source.get('fixture_id')!r}."
        )
    if source.get("generator_version") != GENERATOR_VERSION:
        raise _integrity_error(
            f"Fixture source generator_version must be {GENERATOR_VERSION}, "
            f"got {source.get('generator_version')!r}."
        )
    for key in _LIST_KEYS:
        _check_word_list(key, source.get(key))
    for polarity in _POLARITY_KEYS:
        group = source.get(polarity)
        if not isinstance(group, dict):
            raise _integrity_error(f"Fixture source '{polarity}' must be an object.")
        for key in _POLARITY_LIST_KEYS:
            _check_word_list(f"{polarity}.{key}", group.get(key))
    for template in source["sentence_templates"]:
        _template_slots(template)


def load_source_data() -> dict[str, Any]:
    """Load and validate the committed source data from the package."""
    resource = files("inference_lab").joinpath("fixtures", SOURCE_RESOURCE)
    try:
        source = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _integrity_error(f"Cannot read fixture source data: {exc}") from None
    validate_source(source)
    return source


def _check_seed(seed: Any) -> None:
    if type(seed) is not int or not 0 <= seed <= _MASK64:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Invalid seed: must be an integer in [0, 2**64) (got {seed!r}).",
            {"field": "seed", "value": repr(seed)},
        )


def _make_text(source: dict[str, Any], rng: SplitMix64) -> str:
    """Build one text: a polarity, 1-4 sentences, each from a random template.

    Draw order is fixed: polarity, sentence count, then per sentence the
    template index followed by one draw per slot in order of appearance.
    """
    polarity = rng.below(2)
    same = source[_POLARITY_KEYS[polarity]]
    opposite = source[_POLARITY_KEYS[1 - polarity]]
    pools = {
        "subject": source["subjects"],
        "aspect": source["aspects"],
        "opener": source["openers"],
        "pred": same["predicates"],
        "pred2": same["predicates"],
        "opposite_pred": opposite["predicates"],
        "adj": same["adjectives"],
        "closer": same["closers"],
    }
    templates = source["sentence_templates"]
    sentences = []
    for _ in range(1 + rng.below(MAX_SENTENCES)):
        template = templates[rng.below(len(templates))]
        values = {}
        for slot in _template_slots(template):
            pool = pools[slot]
            values[slot] = pool[rng.below(len(pool))]
        sentences.append(template.format(**values))
    return unicodedata.normalize("NFC", " ".join(sentences))


def generate_texts(
    seed: int = FIXTURE_SEED, source: dict[str, Any] | None = None
) -> list[str]:
    """Generate the ordered fixture: exactly 512 distinct, NFC-normalized texts.

    The default arguments produce the official ``synthetic-sentiment-v1``
    fixture. List position is the fixture index (0-511). Other seeds exist for
    testing seed sensitivity and are not official fixtures.
    """
    _check_seed(seed)
    if source is None:
        source = load_source_data()
    else:
        validate_source(source)
    rng = SplitMix64(seed)
    texts: list[str] = []
    seen: set[str] = set()
    for _ in range(_MAX_ATTEMPTS):
        text = _make_text(source, rng)
        if text not in seen:
            seen.add(text)
            texts.append(text)
            if len(texts) == FIXTURE_SIZE:
                return texts
    raise _integrity_error(
        f"Fixture source could not produce {FIXTURE_SIZE} distinct texts "
        f"within {_MAX_ATTEMPTS} attempts."
    )


def canonical_text_bytes(texts: list[str]) -> bytes:
    """Canonical serialization: NFC strings in a compact JSON array, UTF-8.

    ``ensure_ascii=False``, separators ``(",", ":")``, no trailing newline.
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise _integrity_error("Fixture texts must be a list of strings.")
    normalized = [unicodedata.normalize("NFC", t) for t in texts]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def text_sha256(texts: list[str]) -> str:
    """Lowercase hex SHA-256 of ``canonical_text_bytes(texts)``."""
    return hashlib.sha256(canonical_text_bytes(texts)).hexdigest()


def _check_batch_size(batch_size: Any) -> None:
    if type(batch_size) is not int or batch_size not in SUPPORTED_BATCH_SIZES:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Invalid batch_size: must be one of {list(SUPPORTED_BATCH_SIZES)} "
            f"(got {batch_size!r}).",
            {"field": "batch_size", "value": repr(batch_size)},
        )


def batch_indices(batch_size: int, position: int) -> list[int]:
    """Fixture indices of batch ``position``: ``(position * B + j) % 512``."""
    _check_batch_size(batch_size)
    if type(position) is not int or position < 0:
        raise InferenceLabError(
            ErrorCode.CONFIG_INVALID,
            f"Invalid batch position: must be a non-negative integer "
            f"(got {position!r}).",
            {"field": "position", "value": repr(position)},
        )
    return [(position * batch_size + j) % FIXTURE_SIZE for j in range(batch_size)]


def distinct_inputs_per_trial(batch_size: int) -> int:
    """``min(512, 50 * B)``: distinct fixture inputs seen in one trial."""
    _check_batch_size(batch_size)
    return min(FIXTURE_SIZE, MEASURED_ITERATIONS_PER_TRIAL * batch_size)
