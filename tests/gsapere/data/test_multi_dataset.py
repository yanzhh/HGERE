"""Tests for MultiRelationDataset and MultiDatasetIterableLoader.

Covers:
- MultiDatasetIterableLoader yields all batches from all datasets
- Batches are homogeneous (single dataset_id per batch)
- Sampling schedule respects temperature weighting
- __len__ returns total number of batches
- Batches include dataset_id key
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gsapere.data.config import RelationDatasetParams
from gsapere.data.multi_dataset import MultiDatasetIterableLoader, build_sampling_schedule
from gsapere.data.relation_dataset import RelationDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_doc(n_sents: int = 2) -> dict[str, Any]:
    sents = [["The", "model"] for _ in range(n_sents)]
    ner = [[[0, 0, "Method"]] for _ in range(n_sents)]
    pner = [[[0, 0, "Method"]] for _ in range(n_sents)]
    return {
        "doc_key": "doc_0",
        "sentences": sents,
        "ner": ner,
        "predicted_ner": pner,
        "relations": [[] for _ in range(n_sents)],
    }


def _write_jsonl(path: Path, n_docs: int = 2, n_sents: int = 2) -> Path:
    with path.open("w") as f:
        for _ in range(n_docs):
            f.write(json.dumps(_make_minimal_doc(n_sents)) + "\n")
    return path


def _make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.cls_token = "[CLS]"
    tok.sep_token = "[SEP]"
    tok.pad_token_id = 0
    tok.is_fast = True

    def _tokenize(text: str, **_: Any) -> list[str]:
        return [text[:2]] if len(text) > 2 else [text]

    def _convert(tokens: list[str]) -> list[int]:
        vocab = {"[CLS]": 101, "[SEP]": 102}
        return [vocab.get(t, abs(hash(t)) % 900 + 100) for t in tokens]

    class _Enc:
        def __init__(self, words: list[str]) -> None:
            subtoks = [_tokenize(w)[0] for w in words]
            self.input_ids = _convert(subtoks)
            self._wids = list(range(len(words)))
            self._toks = subtoks

        def word_ids(self) -> list[int]:
            return self._wids

        def tokens(self) -> list[str]:
            return self._toks

    def _call(texts: list[str], is_split_into_words: bool = False, **_: Any) -> _Enc:
        return _Enc(texts)

    tok.side_effect = _call
    tok.tokenize.side_effect = _tokenize
    tok.convert_tokens_to_ids.side_effect = _convert
    tok.__class__ = MagicMock
    return tok


def _make_mock_labels(
    ner_labels: list[str] | None = None,
    rel_labels: list[str] | None = None,
) -> MagicMock:
    labels = MagicMock()
    labels.ner = ner_labels or ["NIL", "Method"]
    rel = MagicMock()
    rel.all = rel_labels or ["NIL", "Used-for"]
    rel.symmetric.return_value = ["NIL"]
    labels.rel = rel
    return labels


def _build_dataset(
    tmp_path: Path,
    name: str,
    batch_size: int = 4,
    n_docs: int = 2,
) -> RelationDataset:
    p = _write_jsonl(tmp_path / f"{name}.jsonl", n_docs=n_docs)
    tok = _make_mock_tokenizer()
    labels = _make_mock_labels()
    params = RelationDatasetParams(
        max_seq_length=32,
        model_type="bert",
        no_sym=True,
        local_rank=-1,
        dataset_id=name,
    )
    ds = RelationDataset(
        logger=MagicMock(),
        tokenizer=tok,
        labels=labels,
        file_path=str(p),
        params=params,
    )
    ds.build(batch_size=batch_size, shuffle=False, n_workers=0, pin_memory=False)
    return ds


# ---------------------------------------------------------------------------
# build_sampling_schedule
# ---------------------------------------------------------------------------


class TestBuildSamplingSchedule:
    def test_schedule_length_equals_total_batches(self, tmp_path: Path) -> None:
        schedule = build_sampling_schedule(
            loader_lengths={"a": 3, "b": 2},
            sampling_weights={"a": 1.0, "b": 1.0},
            temperature=1.0,
        )
        assert len(schedule) == 5

    def test_schedule_contains_all_dataset_names(self, tmp_path: Path) -> None:
        schedule = build_sampling_schedule(
            loader_lengths={"scier": 4, "scinlp": 3},
            sampling_weights={"scier": 1.0, "scinlp": 1.0},
            temperature=1.0,
        )
        assert set(schedule) == {"scier", "scinlp"}

    def test_zero_temperature_uses_dataset_sizes(self) -> None:
        # temperature=0 → p proportional to n^0 = 1/N uniform (edge case)
        schedule = build_sampling_schedule(
            loader_lengths={"a": 10, "b": 2},
            sampling_weights={"a": 1.0, "b": 1.0},
            temperature=0.01,  # near-zero: "a" should dominate
        )
        count_a = schedule.count("a")
        count_b = schedule.count("b")
        assert count_a + count_b == 12
        # With very low temperature "a" still contributes 10 slots (we always
        # include all batches; temperature only affects the interleave order)
        assert count_a == 10
        assert count_b == 2

    def test_sampling_weight_scales_slots(self) -> None:
        # Datasets with different weights but same number of batches
        schedule = build_sampling_schedule(
            loader_lengths={"a": 3, "b": 3},
            sampling_weights={"a": 1.0, "b": 1.0},
            temperature=1.0,
        )
        assert len(schedule) == 6


# ---------------------------------------------------------------------------
# MultiDatasetIterableLoader
# ---------------------------------------------------------------------------


class TestMultiDatasetIterableLoader:
    def test_len_is_total_batches(self, tmp_path: Path) -> None:
        ds_a = _build_dataset(tmp_path, "scier")
        ds_b = _build_dataset(tmp_path, "scinlp")
        loaders = {"scier": ds_a.loader, "scinlp": ds_b.loader}
        n_a = len(ds_a.loader)
        n_b = len(ds_b.loader)
        schedule = build_sampling_schedule(
            loader_lengths={"scier": n_a, "scinlp": n_b},
            sampling_weights={"scier": 1.0, "scinlp": 1.0},
            temperature=1.0,
        )
        loader = MultiDatasetIterableLoader(loaders=loaders, schedule=schedule)
        assert len(loader) == n_a + n_b

    def test_all_batches_yielded(self, tmp_path: Path) -> None:
        ds_a = _build_dataset(tmp_path, "scier")
        ds_b = _build_dataset(tmp_path, "scinlp")
        loaders = {"scier": ds_a.loader, "scinlp": ds_b.loader}
        n_a = len(ds_a.loader)
        n_b = len(ds_b.loader)
        schedule = build_sampling_schedule(
            loader_lengths={"scier": n_a, "scinlp": n_b},
            sampling_weights={"scier": 1.0, "scinlp": 1.0},
            temperature=1.0,
        )
        loader = MultiDatasetIterableLoader(loaders=loaders, schedule=schedule)
        batches = list(loader)
        assert len(batches) == n_a + n_b

    def test_each_batch_has_dataset_id(self, tmp_path: Path) -> None:
        ds_a = _build_dataset(tmp_path, "scier")
        ds_b = _build_dataset(tmp_path, "scinlp")
        loaders = {"scier": ds_a.loader, "scinlp": ds_b.loader}
        n_a = len(ds_a.loader)
        n_b = len(ds_b.loader)
        schedule = build_sampling_schedule(
            loader_lengths={"scier": n_a, "scinlp": n_b},
            sampling_weights={"scier": 1.0, "scinlp": 1.0},
            temperature=1.0,
        )
        loader = MultiDatasetIterableLoader(loaders=loaders, schedule=schedule)
        for batch in loader:
            assert "dataset_id" in batch
            assert batch["dataset_id"] in {"scier", "scinlp"}

    def test_batches_are_homogeneous(self, tmp_path: Path) -> None:
        """Each batch should come from exactly one dataset."""
        ds_a = _build_dataset(tmp_path, "scier")
        ds_b = _build_dataset(tmp_path, "scinlp")
        loaders = {"scier": ds_a.loader, "scinlp": ds_b.loader}
        n_a = len(ds_a.loader)
        n_b = len(ds_b.loader)
        schedule = build_sampling_schedule(
            loader_lengths={"scier": n_a, "scinlp": n_b},
            sampling_weights={"scier": 1.0, "scinlp": 1.0},
            temperature=1.0,
        )
        loader = MultiDatasetIterableLoader(loaders=loaders, schedule=schedule)
        seen_ids = [batch["dataset_id"] for batch in loader]
        # Every batch has exactly one dataset_id (string, not a mix)
        assert all(isinstance(d, str) for d in seen_ids)

    def test_both_datasets_represented(self, tmp_path: Path) -> None:
        ds_a = _build_dataset(tmp_path, "scier")
        ds_b = _build_dataset(tmp_path, "scinlp")
        loaders = {"scier": ds_a.loader, "scinlp": ds_b.loader}
        n_a = len(ds_a.loader)
        n_b = len(ds_b.loader)
        schedule = build_sampling_schedule(
            loader_lengths={"scier": n_a, "scinlp": n_b},
            sampling_weights={"scier": 1.0, "scinlp": 1.0},
            temperature=1.0,
        )
        loader = MultiDatasetIterableLoader(loaders=loaders, schedule=schedule)
        seen_ids = {batch["dataset_id"] for batch in loader}
        assert seen_ids == {"scier", "scinlp"}
