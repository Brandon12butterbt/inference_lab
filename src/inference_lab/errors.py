"""Stable error-code taxonomy shared by the engine, worker and service.

The codes are defined in ``docs/ARCHITECTURE.md`` section 6. They are persisted
and cross the worker process boundary as JSON, so the string values are part of
the contract and must not be renamed casually.
"""

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Machine-readable failure categories. Each value equals its name."""

    CONFIG_INVALID = "CONFIG_INVALID"
    CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    ATTENTION_BACKEND_MISMATCH = "ATTENTION_BACKEND_MISMATCH"
    FIXTURE_INTEGRITY_FAILED = "FIXTURE_INTEGRITY_FAILED"
    PARITY_FAILED = "PARITY_FAILED"
    INVALID_MEASUREMENT = "INVALID_MEASUREMENT"
    CUDA_OOM = "CUDA_OOM"
    CUDA_RUNTIME_ERROR = "CUDA_RUNTIME_ERROR"
    WORKER_CRASHED = "WORKER_CRASHED"
    STORAGE_ERROR = "STORAGE_ERROR"
    INTERRUPTED = "INTERRUPTED"


class InferenceLabError(Exception):
    """Application error carrying a stable code and a human-readable message.

    ``details`` is optional structured context and must stay JSON-compatible.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.code = ErrorCode(code)
        except ValueError:
            raise ValueError(f"Unknown error code: {code!r}") from None
        self.message = message
        self.details = dict(details) if details else {}
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form using the ``error_code``/``error_message`` names
        of ``ExperimentResult``."""
        return {
            "error_code": self.code.value,
            "error_message": self.message,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceLabError":
        return cls(
            data["error_code"],
            data["error_message"],
            data.get("details") or None,
        )
