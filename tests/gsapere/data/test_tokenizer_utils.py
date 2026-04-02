"""Tests for gsapere.data.tokenizer_utils."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch

from gsapere.data.tokenizer_utils import (
    _get_marker_ids,
    _get_word_embeddings,
    adjust_tokenizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50300
EMBED_DIM = 8
logger = logging.getLogger(__name__)


def _make_embeddings() -> torch.Tensor:
    """Return a (VOCAB_SIZE, EMBED_DIM) tensor where row i = float(i)."""
    t = torch.zeros(VOCAB_SIZE, EMBED_DIM)
    for i in range(VOCAB_SIZE):
        t[i] = float(i)
    return t


def _make_bert_model(embeddings: torch.Tensor) -> Any:
    emb = SimpleNamespace(
        word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=embeddings))
    )
    bert = MagicMock()
    bert.embeddings = emb
    model = SimpleNamespace(bert=bert)
    return model


def _make_modernbert_model(embeddings: torch.Tensor) -> Any:
    emb = SimpleNamespace(
        tok_embeddings=SimpleNamespace(weight=SimpleNamespace(data=embeddings))
    )
    bert = MagicMock()
    bert.embeddings = emb
    model = SimpleNamespace(bert=bert)
    return model


def _make_albert_model(embeddings: torch.Tensor) -> Any:
    emb = SimpleNamespace(
        word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=embeddings))
    )
    albert = MagicMock()
    albert.embeddings = emb
    model = SimpleNamespace(albert=albert)
    return model


def _make_roberta_model(embeddings: torch.Tensor) -> Any:
    emb = SimpleNamespace(
        word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=embeddings))
    )
    roberta = MagicMock()
    roberta.embeddings = emb
    model = SimpleNamespace(roberta=roberta)
    return model


def _bert_tokenizer(
    mask_id: int = 103,
    extra_encode: dict[str, int] | None = None,
) -> MagicMock:
    """Tokenizer stub that returns single-token encodes and a fixed vocab size."""
    tok = MagicMock()
    encode_map: dict[str, list[int]] = {"[MASK]": [mask_id]}
    if extra_encode:
        for k, v in extra_encode.items():
            encode_map[k] = [v]

    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        return encode_map.get(text, [999])

    tok.encode.side_effect = encode
    tok.__len__ = lambda self: 30522  # standard BERT vocab size
    return tok


def _modernbert_tokenizer(
    mask_id: int = 103,
    extra_encode: dict[str, int] | None = None,
    multi_token_words: list[str] | None = None,
) -> MagicMock:
    tok = MagicMock()
    encode_map: dict[str, list[int]] = {"[MASK]": [mask_id]}
    if extra_encode:
        for k, v in extra_encode.items():
            encode_map[k] = [v]
    for word in multi_token_words or []:
        encode_map[word] = [1001, 1002]  # multi-token

    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        return encode_map.get(text, [999])

    def convert_tokens_to_ids(token: str) -> int:
        # [unusedX] → 30000 + X after being added
        if token.startswith("[unused") and token.endswith("]"):
            x = int(token[7:-1])
            return 30000 + x
        return 999

    tok.encode.side_effect = encode
    tok.convert_tokens_to_ids.side_effect = convert_tokens_to_ids
    tok.__len__ = lambda self: 30524
    return tok


# ---------------------------------------------------------------------------
# _get_word_embeddings
# ---------------------------------------------------------------------------


class TestGetWordEmbeddings:
    def test_bert(self) -> None:
        emb = object()
        model = SimpleNamespace(
            bert=SimpleNamespace(
                embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=emb))
                )
            )
        )
        assert _get_word_embeddings(model, "bert") is emb

    def test_scibert(self) -> None:
        emb = object()
        model = SimpleNamespace(
            bert=SimpleNamespace(
                embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=emb))
                )
            )
        )
        assert _get_word_embeddings(model, "scibert") is emb

    def test_albert(self) -> None:
        emb = object()
        model = SimpleNamespace(
            albert=SimpleNamespace(
                embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=emb))
                )
            )
        )
        assert _get_word_embeddings(model, "albert") is emb

    def test_roberta(self) -> None:
        emb = object()
        model = SimpleNamespace(
            roberta=SimpleNamespace(
                embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=SimpleNamespace(data=emb))
                )
            )
        )
        assert _get_word_embeddings(model, "roberta") is emb

    def test_modernbert(self) -> None:
        emb = object()
        model = SimpleNamespace(
            bert=SimpleNamespace(
                embeddings=SimpleNamespace(
                    tok_embeddings=SimpleNamespace(weight=SimpleNamespace(data=emb))
                )
            )
        )
        assert _get_word_embeddings(model, "modernbert") is emb


# ---------------------------------------------------------------------------
# _get_marker_ids
# ---------------------------------------------------------------------------


class TestGetMarkerIds:
    def test_bert(self) -> None:
        tok = MagicMock()
        assert _get_marker_ids(tok, "bert", 4) == [1, 2, 3, 4]

    def test_bert_n2(self) -> None:
        tok = MagicMock()
        assert _get_marker_ids(tok, "bert", 2) == [1, 2]

    def test_albert(self) -> None:
        tok = MagicMock()
        assert _get_marker_ids(tok, "albert", 3) == [30000, 30001, 30002]

    def test_roberta(self) -> None:
        tok = MagicMock()
        assert _get_marker_ids(tok, "roberta", 2) == [50261, 50262]

    def test_modernbert_uses_tokenizer(self) -> None:
        tok = MagicMock()
        tok.convert_tokens_to_ids.side_effect = lambda t: {
            "[unused0]": 111,
            "[unused1]": 222,
        }[t]
        result = _get_marker_ids(tok, "modernbert", 2)
        assert result == [111, 222]
        tok.convert_tokens_to_ids.assert_any_call("[unused0]")
        tok.convert_tokens_to_ids.assert_any_call("[unused1]")

    def test_n_zero(self) -> None:
        tok = MagicMock()
        assert _get_marker_ids(tok, "bert", 0) == []


# ---------------------------------------------------------------------------
# adjust_tokenizer — special-token addition
# ---------------------------------------------------------------------------


class TestAdjustTokenizerSpecialTokens:
    def test_albert_adds_tokens_and_resizes(self) -> None:
        emb = _make_embeddings()
        model = _make_albert_model(emb)
        tok = _bert_tokenizer(extra_encode={"entity": [42]})

        adjust_tokenizer(
            tok,
            model,
            "albert",
            n_special_tokens=4,
            lminit=False,
            init_tokens=[],
            logger=logger,
        )

        tok.add_special_tokens.assert_called_once_with(
            {
                "additional_special_tokens": [
                    "[unused0]",
                    "[unused1]",
                    "[unused2]",
                    "[unused3]",
                ]
            }
        )
        model.albert.resize_token_embeddings.assert_called_once_with(len(tok))

    def test_modernbert_adds_tokens_and_resizes(self) -> None:
        emb = _make_embeddings()
        model = _make_modernbert_model(emb)
        tok = _modernbert_tokenizer()

        adjust_tokenizer(
            tok,
            model,
            "modernbert",
            n_special_tokens=2,
            lminit=False,
            init_tokens=[],
            logger=logger,
        )

        tok.add_special_tokens.assert_called_once_with(
            {"additional_special_tokens": ["[unused0]", "[unused1]"]}
        )
        model.bert.resize_token_embeddings.assert_called_once_with(len(tok))

    def test_bert_does_not_add_tokens(self) -> None:
        emb = _make_embeddings()
        model = _make_bert_model(emb)
        tok = _bert_tokenizer()

        adjust_tokenizer(
            tok,
            model,
            "bert",
            n_special_tokens=4,
            lminit=False,
            init_tokens=[],
            logger=logger,
        )

        tok.add_special_tokens.assert_not_called()

    def test_roberta_does_not_add_tokens(self) -> None:
        emb = _make_embeddings()
        model = _make_roberta_model(emb)
        tok = MagicMock()

        adjust_tokenizer(
            tok,
            model,
            "roberta",
            n_special_tokens=4,
            lminit=False,
            init_tokens=[],
            logger=logger,
        )

        tok.add_special_tokens.assert_not_called()

    def test_albert_typemarker_token_count(self) -> None:
        """n_special_tokens is forwarded verbatim — caller controls the count."""
        emb = _make_embeddings()
        model = _make_albert_model(emb)
        tok = _bert_tokenizer()

        adjust_tokenizer(
            tok,
            model,
            "albert",
            n_special_tokens=10,
            lminit=False,
            init_tokens=[],
            logger=logger,
        )

        added = tok.add_special_tokens.call_args[0][0]["additional_special_tokens"]
        assert len(added) == 10
        assert added[0] == "[unused0]"
        assert added[9] == "[unused9]"


# ---------------------------------------------------------------------------
# adjust_tokenizer — lminit=False skips embedding init
# ---------------------------------------------------------------------------


class TestAdjustTokenizerNoLminit:
    def test_bert_no_lminit(self) -> None:
        emb = _make_embeddings()
        original = emb.clone()
        model = _make_bert_model(emb)
        tok = _bert_tokenizer()

        adjust_tokenizer(
            tok,
            model,
            "bert",
            n_special_tokens=4,
            lminit=False,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb, original), (
            "Embeddings should be unchanged when lminit=False"
        )

    def test_albert_no_lminit(self) -> None:
        emb = _make_embeddings()
        original = emb.clone()
        model = _make_albert_model(emb)
        tok = _bert_tokenizer()

        adjust_tokenizer(
            tok,
            model,
            "albert",
            n_special_tokens=4,
            lminit=False,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb, original)


# ---------------------------------------------------------------------------
# adjust_tokenizer — lminit=True, embedding initialization
# ---------------------------------------------------------------------------


class TestAdjustTokenizerLminit:
    def test_bert_one_pair(self) -> None:
        """BERT: [unused0]=1, [unused1]=2; init with 'entity'."""
        mask_id, entity_id = 103, 42
        emb = _make_embeddings()
        expected_mask_emb = emb[mask_id].clone()
        expected_entity_emb = emb[entity_id].clone()
        model = _make_bert_model(emb)
        tok = _bert_tokenizer(mask_id=mask_id, extra_encode={"entity": entity_id})

        adjust_tokenizer(
            tok,
            model,
            "bert",
            n_special_tokens=4,
            lminit=True,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb[1], expected_mask_emb), "[unused0] should equal [MASK]"
        assert torch.equal(emb[2], expected_entity_emb), (
            "[unused1] should equal 'entity'"
        )

    def test_bert_two_pairs(self) -> None:
        """BERT: init_tokens=['subject', 'object'] → 4 marker slots."""
        mask_id, subj_id, obj_id = 103, 10, 20
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        expected_subj = emb[subj_id].clone()
        expected_obj = emb[obj_id].clone()
        model = _make_bert_model(emb)
        tok = _bert_tokenizer(
            mask_id=mask_id, extra_encode={"subject": subj_id, "object": obj_id}
        )

        adjust_tokenizer(
            tok,
            model,
            "bert",
            n_special_tokens=4,
            lminit=True,
            init_tokens=["subject", "object"],
            logger=logger,
        )

        assert torch.equal(emb[1], expected_mask), "[unused0] ← [MASK]"
        assert torch.equal(emb[2], expected_subj), "[unused1] ← subject"
        assert torch.equal(emb[3], expected_mask), "[unused2] ← [MASK]"
        assert torch.equal(emb[4], expected_obj), "[unused3] ← object"

    def test_albert_one_pair(self) -> None:
        """Albert: marker IDs start at 30000."""
        mask_id, entity_id = 103, 42
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        expected_entity = emb[entity_id].clone()
        model = _make_albert_model(emb)
        tok = _bert_tokenizer(mask_id=mask_id, extra_encode={"entity": entity_id})

        adjust_tokenizer(
            tok,
            model,
            "albert",
            n_special_tokens=4,
            lminit=True,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb[30000], expected_mask), "[unused0] ← [MASK]"
        assert torch.equal(emb[30001], expected_entity), "[unused1] ← entity"

    def test_albert_two_pairs(self) -> None:
        mask_id, subj_id, obj_id = 103, 10, 20
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        expected_subj = emb[subj_id].clone()
        expected_obj = emb[obj_id].clone()
        model = _make_albert_model(emb)
        tok = _bert_tokenizer(
            mask_id=mask_id, extra_encode={"subject": subj_id, "object": obj_id}
        )

        adjust_tokenizer(
            tok,
            model,
            "albert",
            n_special_tokens=4,
            lminit=True,
            init_tokens=["subject", "object"],
            logger=logger,
        )

        assert torch.equal(emb[30000], expected_mask)
        assert torch.equal(emb[30001], expected_subj)
        assert torch.equal(emb[30002], expected_mask)
        assert torch.equal(emb[30003], expected_obj)

    def test_roberta_uses_hardcoded_ids(self) -> None:
        """RoBERTa: mask=50264, marker start=50261, end=50262."""
        mask_id, entity_id = 50264, 10014
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        expected_entity = emb[entity_id].clone()
        model = _make_roberta_model(emb)
        tok = MagicMock()

        adjust_tokenizer(
            tok,
            model,
            "roberta",
            n_special_tokens=2,
            lminit=True,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb[50261], expected_mask), "[unused0] ← [MASK]"
        assert torch.equal(emb[50262], expected_entity), "[unused1] ← hardcoded entity"

    def test_modernbert_single_token(self) -> None:
        """ModernBERT: single-token init word uses that token's embedding."""
        mask_id, entity_id = 103, 99
        unused0_id, unused1_id = 30000, 30001
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        expected_entity = emb[entity_id].clone()
        model = _make_modernbert_model(emb)
        tok = _modernbert_tokenizer(mask_id=mask_id, extra_encode={"entity": entity_id})

        adjust_tokenizer(
            tok,
            model,
            "modernbert",
            n_special_tokens=2,
            lminit=True,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb[unused0_id], expected_mask), "[unused0] ← [MASK]"
        assert torch.equal(emb[unused1_id], expected_entity), "[unused1] ← entity"

    def test_modernbert_multi_token_fallback(self) -> None:
        """ModernBERT: multi-token init word falls back to [MASK] embedding."""
        mask_id = 103
        unused0_id, unused1_id = 30000, 30001
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        model = _make_modernbert_model(emb)
        tok = _modernbert_tokenizer(mask_id=mask_id, multi_token_words=["entity"])

        adjust_tokenizer(
            tok,
            model,
            "modernbert",
            n_special_tokens=2,
            lminit=True,
            init_tokens=["entity"],
            logger=logger,
        )

        assert torch.equal(emb[unused0_id], expected_mask), "[unused0] ← [MASK]"
        assert torch.equal(emb[unused1_id], expected_mask), (
            "[unused1] ← [MASK] (fallback)"
        )

    def test_modernbert_two_pairs(self) -> None:
        mask_id, subj_id, obj_id = 103, 55, 66
        emb = _make_embeddings()
        expected_mask = emb[mask_id].clone()
        expected_subj = emb[subj_id].clone()
        expected_obj = emb[obj_id].clone()
        model = _make_modernbert_model(emb)
        tok = _modernbert_tokenizer(
            mask_id=mask_id, extra_encode={"subject": subj_id, "object": obj_id}
        )

        adjust_tokenizer(
            tok,
            model,
            "modernbert",
            n_special_tokens=4,
            lminit=True,
            init_tokens=["subject", "object"],
            logger=logger,
        )

        # unused0=30000, unused1=30001, unused2=30002, unused3=30003
        assert torch.equal(emb[30000], expected_mask)
        assert torch.equal(emb[30001], expected_subj)
        assert torch.equal(emb[30002], expected_mask)
        assert torch.equal(emb[30003], expected_obj)

    def test_other_embeddings_not_modified(self) -> None:
        """Only the target marker slots are written; everything else stays put."""
        mask_id, entity_id = 103, 42
        emb = _make_embeddings()
        original = emb.clone()
        model = _make_bert_model(emb)
        tok = _bert_tokenizer(mask_id=mask_id, extra_encode={"entity": entity_id})

        adjust_tokenizer(
            tok,
            model,
            "bert",
            n_special_tokens=4,
            lminit=True,
            init_tokens=["entity"],
            logger=logger,
        )

        # Rows 1 and 2 are written; all others must be identical to original
        modified_rows = {1, 2}
        for i in range(VOCAB_SIZE):
            if i not in modified_rows:
                assert torch.equal(emb[i], original[i]), f"Row {i} should be unmodified"
