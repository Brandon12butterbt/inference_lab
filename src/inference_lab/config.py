"""Typed, validated benchmark configuration.

Fields and allowed values follow ``docs/ARCHITECTURE.md`` section 4; the fixed
protocol constants follow ``docs/BENCHMARK_PROTOCOL.md``. A ``BenchmarkConfig``
is validated on construction, so an instance is always a legal configuration.
This module has no device, model or filesystem access.
"""

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import psutil

from inference_lab.errors import ErrorCode, InferenceLabError

# --- Approved protocol constants (change only via an owner-approved revision) ---

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SUPPORTED_DEVICES = ("cpu", "cuda:0")
SUPPORTED_BATCH_SIZES = (1, 4, 8, 16, 32)
SEQUENCE_LENGTH = 128
PRECISION = "fp32"
MATMUL_PRECISION = "highest"
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"  # a request only; verified at model load
WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS_PER_TRIAL = 50
TRIALS = 3
MEMORY_PROFILE_FORWARDS = 10
FIXTURE_ID = "synthetic-sentiment-v1"
FIXTURE_SEED = 42

# Fixture identity: each supported fixture id is tied to its generation seed.
SUPPORTED_FIXTURES = {FIXTURE_ID: FIXTURE_SEED}

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def default_cpu_threads() -> int:
    """Hardware-aware default for ``cpu_threads`` on the current machine.

    Prefers the physical core count reported by psutil, then the logical count,
    then 1 if neither can be detected. Only used when a configuration is created
    without an explicit ``cpu_threads``; the result is stored in the config as a
    concrete integer and is never re-derived when a saved config is reloaded.
    """
    for logical in (False, True):
        try:
            count = psutil.cpu_count(logical=logical)
        except (OSError, RuntimeError, ValueError):  # detection must not block config
            count = None
        if type(count) is int and count >= 1:
            return count
    return 1


def _invalid(field: str, message: str, value: Any) -> InferenceLabError:
    return InferenceLabError(
        ErrorCode.CONFIG_INVALID,
        f"Invalid {field}: {message} (got {value!r}).",
        {"field": field, "value": repr(value)},
    )


def _check_str(field: str, value: Any) -> None:
    if not isinstance(value, str):
        raise _invalid(field, "must be a string", value)


def _check_int(field: str, value: Any) -> None:
    # bool is a subclass of int, and 50.0 is not an iteration count.
    if type(value) is not int:
        raise _invalid(field, "must be an integer", value)


def _check_fixed_int(field: str, value: Any, expected: int) -> None:
    _check_int(field, value)
    if value != expected:
        raise _invalid(field, f"must be {expected} under the protocol", value)


def _check_fixed_str(field: str, value: Any, expected: str) -> None:
    _check_str(field, value)
    if value != expected:
        raise _invalid(field, f"must be {expected!r} under the protocol", value)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """One benchmark configuration (device + batch size + fixed parameters).

    ``model_revision`` has no default: it must be an explicit 40-character
    lowercase hexadecimal commit hash. Only its format is checked here; the
    exact checkpoint is pinned later in M1.
    """

    model_revision: str
    device: str
    batch_size: int
    model_id: str = MODEL_ID
    sequence_length: int = SEQUENCE_LENGTH
    precision: str = PRECISION
    matmul_precision: str = MATMUL_PRECISION
    # Configuration only: actual backend compatibility is checked at model load.
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    warmup_iterations: int = WARMUP_ITERATIONS
    measured_iterations_per_trial: int = MEASURED_ITERATIONS_PER_TRIAL
    trials: int = TRIALS
    memory_profile_forwards: int = MEMORY_PROFILE_FORWARDS
    fixture_id: str = FIXTURE_ID
    fixture_seed: int = FIXTURE_SEED
    # Resolved to a concrete integer at creation when not given. The lambda looks
    # the function up at call time (so tests can substitute detection).
    cpu_threads: int = field(default_factory=lambda: default_cpu_threads())

    def __post_init__(self) -> None:
        _check_fixed_str("model_id", self.model_id, MODEL_ID)

        _check_str("model_revision", self.model_revision)
        if not _REVISION_PATTERN.fullmatch(self.model_revision):
            raise _invalid(
                "model_revision",
                "must be a 40-character lowercase hexadecimal commit hash",
                self.model_revision,
            )

        _check_str("device", self.device)
        if self.device not in SUPPORTED_DEVICES:
            raise _invalid(
                "device", f"must be one of {list(SUPPORTED_DEVICES)}", self.device
            )

        _check_int("batch_size", self.batch_size)
        if self.batch_size not in SUPPORTED_BATCH_SIZES:
            raise _invalid(
                "batch_size",
                f"must be one of {list(SUPPORTED_BATCH_SIZES)}",
                self.batch_size,
            )

        _check_fixed_int("sequence_length", self.sequence_length, SEQUENCE_LENGTH)
        _check_fixed_str("precision", self.precision, PRECISION)
        _check_fixed_str("matmul_precision", self.matmul_precision, MATMUL_PRECISION)

        _check_str("attn_implementation", self.attn_implementation)
        if not self.attn_implementation or self.attn_implementation != (
            self.attn_implementation.strip()
        ):
            raise _invalid(
                "attn_implementation",
                "must be a non-empty name without surrounding whitespace",
                self.attn_implementation,
            )

        _check_fixed_int("warmup_iterations", self.warmup_iterations, WARMUP_ITERATIONS)
        _check_fixed_int(
            "measured_iterations_per_trial",
            self.measured_iterations_per_trial,
            MEASURED_ITERATIONS_PER_TRIAL,
        )
        _check_fixed_int("trials", self.trials, TRIALS)
        _check_fixed_int(
            "memory_profile_forwards",
            self.memory_profile_forwards,
            MEMORY_PROFILE_FORWARDS,
        )

        _check_str("fixture_id", self.fixture_id)
        if self.fixture_id not in SUPPORTED_FIXTURES:
            raise _invalid(
                "fixture_id",
                f"must be one of {list(SUPPORTED_FIXTURES)}",
                self.fixture_id,
            )
        _check_fixed_int(
            "fixture_seed", self.fixture_seed, SUPPORTED_FIXTURES[self.fixture_id]
        )

        _check_int("cpu_threads", self.cpu_threads)
        # No hardware-based upper bound: an explicit value is never capped or
        # altered, so a saved config reloads identically on any machine.
        if self.cpu_threads < 1:
            raise _invalid(
                "cpu_threads", "must be a positive integer", self.cpu_threads
            )

    def to_dict(self) -> dict[str, Any]:
        """All fields as a plain JSON-compatible dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Deterministic JSON text (sorted keys, compact separators)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Rebuild a config from ``to_dict`` output.

        Strict: every field must be present and no unknown field is accepted,
        so a stored configuration is never silently completed or reinterpreted.
        """
        if not isinstance(data, dict):
            raise _invalid("config", "must be a JSON object", data)
        expected = {f.name for f in fields(cls)}
        missing = sorted(expected - data.keys())
        unknown = sorted(data.keys() - expected)
        if missing or unknown:
            raise InferenceLabError(
                ErrorCode.CONFIG_INVALID,
                f"Invalid config fields: missing={missing}, unknown={unknown}.",
                {"missing": missing, "unknown": unknown},
            )
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkConfig":
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise InferenceLabError(
                ErrorCode.CONFIG_INVALID, f"Config is not valid JSON: {exc}"
            ) from None
        return cls.from_dict(data)
