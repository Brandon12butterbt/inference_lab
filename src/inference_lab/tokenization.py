"""Tokenize the fixture to fixed-length tensors and compute their hashes.

Implements ``docs/BENCHMARK_PROTOCOL.md`` section 2.1 (tokenization with
``return_tensors="pt"``, ``token_ids_sha256`` / ``attention_mask_sha256`` over
little-endian int64 C-order bytes) and the truncation count. Kept separate from
``dataset.py`` so the text generator stays independent of PyTorch.
"""

import hashlib
import statistics
from dataclasses import dataclass
from typing import Any

import torch

from inference_lab.config import SEQUENCE_LENGTH
from inference_lab.errors import ErrorCode, InferenceLabError


@dataclass(frozen=True, eq=False)
class TokenizedFixture:
    """Fixed-length token tensors plus the facts recorded in the fixture manifest."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    sequence_length: int
    token_ids_sha256: str
    attention_mask_sha256: str
    # Original lengths include special tokens ([CLS]/[SEP]) before truncation.
    truncated_count: int
    real_length_min: int
    real_length_median: float
    real_length_max: int

    def manifest_facts(self) -> dict[str, Any]:
        """JSON-compatible tokenizer-dependent facts (no tensors)."""
        return {
            "sequence_length": self.sequence_length,
            "count": int(self.input_ids.shape[0]),
            "token_ids_sha256": self.token_ids_sha256,
            "attention_mask_sha256": self.attention_mask_sha256,
            "truncated_count": self.truncated_count,
            "real_length_min": self.real_length_min,
            "real_length_median": self.real_length_median,
            "real_length_max": self.real_length_max,
        }


def _integrity_error(message: str) -> InferenceLabError:
    return InferenceLabError(ErrorCode.FIXTURE_INTEGRITY_FAILED, message)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of ``tensor`` as little-endian int64 bytes in C (row-major) order."""
    array = tensor.detach().to("cpu").to(torch.int64).contiguous().numpy()
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes()).hexdigest()


def tokenize_fixture(
    tokenizer: Any, texts: list[str], sequence_length: int = SEQUENCE_LENGTH
) -> TokenizedFixture:
    """Tokenize ``texts`` to ``(len(texts), sequence_length)`` int64 tensors.

    Uses explicit ``padding="max_length"`` and ``truncation=True``. Truncated
    inputs are counted from a second, untruncated pass that includes special
    tokens, so an input of exactly ``sequence_length`` tokens is not truncated.
    """
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(t, str) for t in texts)
    ):
        raise _integrity_error("Fixture texts must be a non-empty list of strings.")

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    expected_shape = (len(texts), sequence_length)
    for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask)):
        if tuple(tensor.shape) != expected_shape:
            raise _integrity_error(
                f"{name} has shape {tuple(tensor.shape)}; expected {expected_shape}."
            )
        if tensor.dtype != torch.int64:
            raise _integrity_error(f"{name} has dtype {tensor.dtype}; expected int64.")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise _integrity_error("attention_mask must contain only 0 and 1.")

    untruncated = tokenizer(
        texts, padding=False, truncation=False, add_special_tokens=True
    )["input_ids"]
    lengths = [len(ids) for ids in untruncated]
    if len(lengths) != len(texts):
        raise _integrity_error("Untruncated tokenization returned a different count.")
    expected_mask_sums = torch.tensor(
        [min(n, sequence_length) for n in lengths], dtype=torch.int64
    )
    if not torch.equal(attention_mask.sum(dim=1), expected_mask_sums):
        raise _integrity_error(
            "attention_mask row sums do not match the original token lengths."
        )

    return TokenizedFixture(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        token_ids_sha256=tensor_sha256(input_ids),
        attention_mask_sha256=tensor_sha256(attention_mask),
        truncated_count=sum(1 for n in lengths if n > sequence_length),
        real_length_min=min(lengths),
        real_length_median=float(statistics.median(lengths)),
        real_length_max=max(lengths),
    )
