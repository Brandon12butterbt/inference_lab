"""Offline unit tests for ``inference_lab.tokenization``.

A tiny synthetic word-level tokenizer built with the ``tokenizers`` library
stands in for the real DistilBERT tokenizer: no network, no model files. Its
token ids are NOT the real vocabulary, so none of these hashes relate to the
real fixture manifest.
"""

import hashlib
import json
import struct

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.tokenization import TokenizedFixture, tensor_sha256, tokenize_fixture

CLS, SEP, PAD = 2, 3, 0


@pytest.fixture(scope="module")
def tokenizer():
    vocab = {
        w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "good"])
    }
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", CLS), ("[SEP]", SEP)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def words(n: int) -> str:
    return " ".join(["the"] * n)


def pack(values) -> bytes:
    return struct.pack(f"<{len(values)}q", *values)


# --- tensor_sha256 --------------------------------------------------------------


def test_hash_is_little_endian_int64_c_order():
    tensor = torch.tensor([[1, -2], [3, 256]], dtype=torch.int64)
    expected = hashlib.sha256(pack([1, -2, 3, 256])).hexdigest()
    assert tensor_sha256(tensor) == expected
    assert len(expected) == 64


def test_hash_uses_row_major_order_even_for_non_contiguous_views():
    view = torch.arange(6, dtype=torch.int64).reshape(2, 3).T  # non-contiguous
    assert not view.is_contiguous()
    expected = hashlib.sha256(pack([0, 3, 1, 4, 2, 5])).hexdigest()
    assert tensor_sha256(view) == expected


def test_hash_is_independent_of_input_integer_width():
    wide = torch.tensor([[7, 8, 9]], dtype=torch.int64)
    narrow = wide.to(torch.int32)
    assert tensor_sha256(narrow) == tensor_sha256(wide)


def test_hash_differs_when_content_or_shape_order_differs():
    a = torch.tensor([[1, 2], [3, 4]])
    assert tensor_sha256(a) != tensor_sha256(a.flip(0))
    assert tensor_sha256(a) != tensor_sha256(a + 1)


# --- tokenize_fixture -----------------------------------------------------------


def test_shapes_dtypes_padding_and_special_tokens(tokenizer):
    result = tokenize_fixture(tokenizer, ["the good", words(3), ""])
    assert isinstance(result, TokenizedFixture)
    assert result.input_ids.shape == (3, 128)
    assert result.attention_mask.shape == (3, 128)
    assert result.input_ids.dtype == torch.int64
    assert result.attention_mask.dtype == torch.int64
    assert result.sequence_length == 128
    first = result.input_ids[0].tolist()
    assert first[:4] == [CLS, 4, 5, SEP]  # [CLS] the good [SEP]
    assert first[4:] == [PAD] * 124
    assert result.attention_mask[0].tolist() == [1] * 4 + [0] * 124
    assert result.attention_mask[2].sum().item() == 2  # empty text: [CLS] [SEP]


def test_truncation_counts_use_lengths_including_special_tokens(tokenizer):
    # n words -> n + 2 tokens ([CLS], [SEP]). 126 words = exactly 128: kept whole.
    texts = [words(5), words(126), words(127), words(300)]
    result = tokenize_fixture(tokenizer, texts)
    assert result.truncated_count == 2  # only the 129- and 302-token inputs
    assert result.real_length_min == 7
    assert result.real_length_max == 302
    assert result.real_length_median == pytest.approx((128 + 129) / 2)
    assert isinstance(result.real_length_median, float)
    # Exactly-128 input is complete; truncated inputs still end with [SEP].
    assert result.attention_mask[1].sum().item() == 128
    assert result.input_ids[1, 127].item() == SEP
    assert result.attention_mask[2].sum().item() == 128
    assert result.attention_mask[3].sum().item() == 128
    assert result.input_ids[3, 127].item() == SEP
    assert result.input_ids[3, 126].item() == tokenizer.convert_tokens_to_ids("the")


def test_no_truncation_reported_for_short_inputs(tokenizer):
    result = tokenize_fixture(tokenizer, [words(1), words(50)])
    assert result.truncated_count == 0


def test_hashes_match_independently_built_bytes(tokenizer):
    result = tokenize_fixture(tokenizer, ["the good", words(130), words(3)])
    ids = [v for row in result.input_ids.tolist() for v in row]
    mask = [v for row in result.attention_mask.tolist() for v in row]
    assert result.token_ids_sha256 == hashlib.sha256(pack(ids)).hexdigest()
    assert result.attention_mask_sha256 == hashlib.sha256(pack(mask)).hexdigest()
    assert result.token_ids_sha256 != result.attention_mask_sha256


def test_tokenization_is_deterministic(tokenizer):
    texts = ["the good", words(40), words(200)]
    first = tokenize_fixture(tokenizer, texts)
    second = tokenize_fixture(tokenizer, texts)
    assert torch.equal(first.input_ids, second.input_ids)
    assert first.manifest_facts() == second.manifest_facts()


def test_manifest_facts_are_json_compatible_without_tensors(tokenizer):
    facts = tokenize_fixture(tokenizer, [words(3), words(200)]).manifest_facts()
    assert json.loads(json.dumps(facts)) == facts
    assert facts["count"] == 2
    assert facts["sequence_length"] == 128
    assert facts["truncated_count"] == 1


@pytest.mark.parametrize("bad", [None, "text", [], ["ok", 1], ("a", "b")])
def test_invalid_texts_rejected(tokenizer, bad):
    with pytest.raises(InferenceLabError) as excinfo:
        tokenize_fixture(tokenizer, bad)
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED


class MisbehavingTokenizer:
    """Returns configurable (wrong) tensors for the padded call."""

    def __init__(self, ids, mask, lengths):
        self.ids, self.mask, self.lengths = ids, mask, lengths

    def __call__(self, texts, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": self.ids, "attention_mask": self.mask}
        return {"input_ids": [[1] * n for n in self.lengths]}


def test_wrong_shape_is_rejected():
    bad = MisbehavingTokenizer(
        torch.zeros(1, 64, dtype=torch.int64), torch.ones(1, 64, dtype=torch.int64), [3]
    )
    with pytest.raises(InferenceLabError) as excinfo:
        tokenize_fixture(bad, ["x"])
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED
    assert "shape" in excinfo.value.message


def test_wrong_dtype_is_rejected():
    bad = MisbehavingTokenizer(
        torch.zeros(1, 128, dtype=torch.int32),
        torch.ones(1, 128, dtype=torch.int64),
        [3],
    )
    with pytest.raises(InferenceLabError) as excinfo:
        tokenize_fixture(bad, ["x"])
    assert "dtype" in excinfo.value.message


def test_non_binary_mask_is_rejected():
    mask = torch.ones(1, 128, dtype=torch.int64) * 2
    bad = MisbehavingTokenizer(torch.zeros(1, 128, dtype=torch.int64), mask, [3])
    with pytest.raises(InferenceLabError) as excinfo:
        tokenize_fixture(bad, ["x"])
    assert "0 and 1" in excinfo.value.message


def test_mask_inconsistent_with_original_lengths_is_rejected():
    ones = torch.ones(1, 128, dtype=torch.int64)
    bad = MisbehavingTokenizer(torch.zeros(1, 128, dtype=torch.int64), ones, [3])
    with pytest.raises(InferenceLabError) as excinfo:
        tokenize_fixture(bad, ["x"])
    assert excinfo.value.code is ErrorCode.FIXTURE_INTEGRITY_FAILED
    assert "row sums" in excinfo.value.message
