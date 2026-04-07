"""Shared fixtures for hgere/data tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any  # noqa: F401 — used in mock_tokenizer _call signature
from unittest.mock import MagicMock

import pytest

from gsapere.data.config import RelationDatasetParams


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
        "ner": [[[1, 1, "Method"], [2, 2, "Dataset"]], [[2, 2, "Task"]]],
        "predicted_ner": [[[1, 1, "Method"], [2, 2, "Dataset"]], [[2, 2, "Task"]]],
        "relations": [[[1, 1, 2, 2, "Used-for"]], []],
    }
    path = tmp_path / "relation.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def relation_jsonl_double_ann(tmp_path: Path) -> Path:
    """JSONL with a doubly-annotated span: span (1,1) has both 'Method' and 'Task'."""
    doc = {
        "doc_key": "doc_double_ann",
        "sentences": [["The", "model", "trains", "fast"]],
        "ner": [[[1, 1, "Method"], [1, 1, "Task"], [2, 2, "Dataset"]]],
        "predicted_ner": [[[1, 1, "Method"], [2, 2, "Dataset"]]],
        "relations": [[[1, 1, 2, 2, "Used-for"]]],
    }
    path = tmp_path / "relation_double_ann.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def relation_jsonl_double_ann_connectivity(tmp_path: Path) -> Path:
    """JSONL where connectivity breaks the tiebreak for a doubly-annotated span.

    Span (1,1): labels {Method, Task} — doubly annotated.
    Span (2,2): singly annotated as "Task".
    Span (3,3): singly annotated as "Dataset".

    Relations:
      (2,2) → (3,3) "Used-for"   — contributes +1 to "Task" connectivity
      (1,1) → (3,3) "Result"     — (1,1) is multi-annotated so excluded from count

    Connectivity from single-label spans:
      "Task": 1  (via span (2,2))
      "Method": 0

    Expected chosen label for (1,1): "Task"  (higher connectivity than "Method")
    """
    doc = {
        "doc_key": "doc_connectivity",
        "sentences": [["The", "model", "trains", "fast", "well"]],
        "ner": [
            [
                [1, 1, "Method"],
                [1, 1, "Task"],
                [2, 2, "Task"],
                [3, 3, "Dataset"],
            ]
        ],
        "predicted_ner": [[[1, 1, "Method"], [2, 2, "Task"], [3, 3, "Dataset"]]],
        "relations": [[[2, 2, 3, 3, "Used-for"], [1, 1, 3, 3, "Result"]]],
    }
    path = tmp_path / "relation_double_ann_connectivity.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def relation_jsonl_self_rel(tmp_path: Path) -> Path:
    """JSONL with an explicit self-relation for testing the warning."""
    doc = {
        "doc_key": "doc_self_rel",
        "sentences": [["The", "model", "trains", "fast"]],
        "ner": [[[1, 1, "Method"], [2, 2, "Dataset"]]],
        "predicted_ner": [[[1, 1, "Method"], [2, 2, "Dataset"]]],
        "relations": [[[1, 1, 1, 1, "Used-for"]]],
    }
    path = tmp_path / "relation_self_rel.jsonl"
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
def mock_labels_with_sym() -> MagicMock:
    """Mock labels with a symmetric relation label 'Similar'."""
    labels = MagicMock()
    labels.ner = ["NIL", "Method", "Task", "Dataset"]
    rel_obj = MagicMock()
    rel_obj.all = ["NIL", "Similar", "Used-for"]
    # symmetric(only_nil=True) → ["NIL"], symmetric(only_nil=False) → ["NIL", "Similar"]
    rel_obj.symmetric.side_effect = lambda only_nil=False: (
        ["NIL"] if only_nil else ["NIL", "Similar"]
    )
    labels.rel = rel_obj
    return labels


@pytest.fixture
def relation_jsonl_sym_reverse(tmp_path: Path) -> Path:
    """JSONL with a symmetric relation annotated in reverse direction.

    Gold: span (2,2) → span (1,1) with 'Similar' (reverse of canonical).
    Canonical is (1,1) < (2,2), so after normalisation both map to (1,1)→(2,2).
    """
    doc = {
        "doc_key": "doc_sym_rev",
        "sentences": [["The", "model", "trains", "fast"]],
        "ner": [[[1, 1, "Method"], [2, 2, "Method"]]],
        "predicted_ner": [[[1, 1, "Method"], [2, 2, "Method"]]],
        "relations": [[[2, 2, 1, 1, "Similar"]]],  # reverse direction
    }
    path = tmp_path / "relation_sym_rev.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def relation_jsonl_path_with_fp(tmp_path: Path) -> Path:
    """JSONL for candidate-stats tests: predicted_ner has 1 FP and 1 FN vs gold ner.

    Gold NER: sentence 0 → [(1,1,"Method")], sentence 1 → [(2,2,"Task")]
    Predicted: sentence 0 → [(1,1,"Method"), (0,0,"Task")]  ← extra FP at (0,0)
               sentence 1 → []                              ← gold (2,2) missing → FN
    Expected stats: n_gold=2, n_candidates=2, n_tp=1, n_fp=1, n_fn=1
    """
    doc = {
        "doc_key": "doc_fp_fn",
        "sentences": [
            ["The", "model", "trains", "fast"],
            ["Results", "show", "accuracy"],
        ],
        "ner": [[[1, 1, "Method"]], [[2, 2, "Task"]]],
        "predicted_ner": [[[1, 1, "Method"], [0, 0, "Task"]], []],
        "relations": [[], []],
    }
    path = tmp_path / "relation_fp_fn.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def relation_params() -> RelationDatasetParams:
    """RelationDatasetParams for RelationDataset tests."""
    return RelationDatasetParams(
        max_seq_length=64,
        use_typemarker=False,
        local_rank=-1,
        model_type="bert",
        no_sym=True,
        nocross=False,
    )
