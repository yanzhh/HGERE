"""Shared fixtures for hgere/data tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any  # noqa: F401 — used in mock_tokenizer _call signature
from unittest.mock import MagicMock

import pytest


def _make_doc(
    doc_key: str,
    sentences: list[list[str]],
    ner: list[list[list[Any]]] | None = None,
    relations: list[list[list[Any]]] | None = None,
) -> dict[str, Any]:
    if ner is None:
        ner = [[] for _ in sentences]
    if relations is None:
        relations = [[] for _ in sentences]
    return {
        "doc_key": doc_key,
        "sentences": sentences,
        "ner": ner,
        "relations": relations,
    }


@pytest.fixture
def tiny_jsonl_path(tmp_path: Path) -> Path:
    """Three-document JSONL file for testing file indexing."""
    docs = [
        _make_doc(
            "doc_0",
            [["Hello", "world", "."]],
            ner=[[[0, 1, "Task"]]],
        ),
        _make_doc(
            "doc_1",
            [
                ["The", "model", "trains", "fast", "."],
                ["Results", "show", "accuracy", "."],
            ],
            ner=[[[1, 1, "Method"]], []],
            relations=[[[1, 1, 1, 1, "Used-for"]], []],
        ),
        _make_doc(
            "doc_2",
            [["Simple", "sentence", "."]],
        ),
    ]
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    return path


@pytest.fixture
def single_doc_jsonl_path(tmp_path: Path) -> Path:
    """Single-document JSONL with 5 tokens and 1 gold entity for pruner tests."""
    doc = _make_doc(
        "doc_0",
        [["The", "neural", "model", "trains", "fast"]],
        ner=[[[1, 2, "Method"]]],
    )
    path = tmp_path / "single.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Stub tokenizer: tokenize splits on 'long' words (>4 chars → 2 subwords),
    short words map 1:1.  Has cls_token, sep_token, pad_token_id.

    Supports both the legacy per-word API (tok.tokenize) and the modern batch
    API (tok(words, is_split_into_words=True)) used by _tokenize_document.
    """
    tok = MagicMock()
    tok.cls_token = "[CLS]"
    tok.sep_token = "[SEP]"
    tok.pad_token_id = 0
    tok.is_fast = True

    def _split_word(text: str) -> list[str]:
        # Words longer than 4 chars split into two subwords for testing
        if len(text) > 4:
            return [text[:2], "##" + text[2:]]
        return [text]

    def _tokenize(text: str, add_prefix_space: bool = False) -> list[str]:
        return _split_word(text)

    def _convert_tokens_to_ids(tokens: list[str]) -> list[int]:
        vocab: dict[str, int] = {"[CLS]": 101, "[SEP]": 102}
        return [vocab.get(t, abs(hash(t)) % 900 + 100) for t in tokens]

    class _MockEncoding:
        """Minimal BatchEncoding stand-in for is_split_into_words=True calls."""

        def __init__(self, ids: list[int], wids: list[int], toks: list[str]) -> None:
            self.input_ids = ids
            self._word_ids = wids
            self._tokens = toks

        def word_ids(self) -> list[int]:
            return list(self._word_ids)

        def tokens(self) -> list[str]:
            return list(self._tokens)

    def _call(
        texts: list[str],
        is_split_into_words: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> _MockEncoding:
        if not is_split_into_words:
            raise NotImplementedError("mock only supports is_split_into_words=True")
        all_subtokens: list[str] = []
        wids: list[int] = []
        for word_idx, word in enumerate(texts):
            subtokens = _split_word(word)
            all_subtokens.extend(subtokens)
            wids.extend([word_idx] * len(subtokens))
        ids = _convert_tokens_to_ids(all_subtokens)
        return _MockEncoding(ids, wids, all_subtokens)

    tok.side_effect = _call
    tok.tokenize.side_effect = _tokenize
    tok.convert_tokens_to_ids.side_effect = _convert_tokens_to_ids
    # RobertaTokenizer check — return False so prefix-space branch is not taken
    tok.__class__ = MagicMock  # not RobertaTokenizer
    return tok


@pytest.fixture
def relation_jsonl_path(tmp_path: Path) -> Path:
    """JSONL for RelationDataset tests: 1 doc, 2 sentences, with predicted_ner."""
    doc = {
        "doc_key": "doc_rel_0",
        "sentences": [
            ["The", "model", "trains", "fast"],
            ["Results", "show", "accuracy"],
        ],
        "ner": [[[1, 1, "Method"]], [[2, 2, "Task"]]],
        "predicted_ner": [[[1, 1, "Method"]], [[2, 2, "Task"]]],
        "relations": [[[1, 1, 1, 1, "Used-for"]], []],
    }
    path = tmp_path / "relation.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def mock_labels() -> MagicMock:
    """Mock labels object mimicking hgere Labels interface."""
    labels = MagicMock()
    labels.ner = ["NIL", "Method", "Task", "Dataset"]
    rel_obj = MagicMock()
    rel_obj.all = ["NIL", "Used-for", "Result"]
    rel_obj.symmetric.return_value = ["NIL"]
    labels.rel = rel_obj
    return labels


@pytest.fixture
def mock_args(mock_tokenizer: MagicMock) -> MagicMock:
    """Mock args object for RelationDataset."""
    args = MagicMock()
    args.max_seq_length = 64
    args.use_typemarker = False
    args.local_rank = -1
    args.model_type = "bert"
    args.no_sym = True
    args.nocross = False
    return args
