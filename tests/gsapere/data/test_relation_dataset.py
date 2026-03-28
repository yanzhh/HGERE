"""Tests for RelationDataset (reworked) — Steps 11-13 of TDD plan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from gsapere.data.collators import RelationCollator
from gsapere.data.config import RelationDatasetParams
from gsapere.data.data_types import CandidateStats
from gsapere.data.relation_dataset import RelationDataset


def _make_dataset(
    file_path: Path,
    tokenizer: MagicMock,
    labels: MagicMock,
    params: RelationDatasetParams,
) -> RelationDataset:
    return RelationDataset(
        logger=MagicMock(),
        tokenizer=tokenizer,
        labels=labels,
        file_path=str(file_path),
        params=params,
    )


# ---------------------------------------------------------------------------
# Step 11 — _build_index (initialize): length, golden labels, predicted NERs
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_len_equals_n_sentences(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """The fixture has 1 doc with 2 sentences → dataset length == 2."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds) == 2

    def test_golden_labels_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds.golden_labels) > 0

    def test_ner_golden_labels_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds.ner_golden_labels) > 0

    def test_global_predicted_ners_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        # (0, 0) = doc_idx=0, sent_idx=0
        assert (0, 0) in ds.global_predicted_ners
        assert len(ds.global_predicted_ners[(0, 0)]) >= 1

    def test_sizes_list_length(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """self.sizes should have one entry per sentence."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds.sizes) == len(ds)


# ---------------------------------------------------------------------------
# Step 12 — _item_to_tensors (prepare_item)
# ---------------------------------------------------------------------------


class TestItemToTensors:
    def test_item_is_dict(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        assert isinstance(item, dict)

    def test_item_has_required_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        expected_keys = {
            "indices",
            "input_ids",
            "attention_mask",
            "position_ids",
            "sub_positions",
            "rel_labels",
            "ner_labels",
            "obj_token_pos",
            "n_ent",
            "subtoken_len",
            "sub",
        }
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        assert expected_keys == set(item.keys())

    def test_input_ids_shape_with_entities(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """For sentence with n_ent > 0, input_ids shape = (n_ent, seq_len+2*n_ent)."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        n_ent = item["n_ent"]
        if n_ent > 0:
            seq_len = item["subtoken_len"]
            assert item["input_ids"].shape[0] == n_ent
            assert item["input_ids"].shape[1] == seq_len + 2 * n_ent

    def test_rel_labels_shape(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        n_ent = item["n_ent"]
        if n_ent > 0:
            assert item["rel_labels"].shape == (n_ent, n_ent)

    def test_n_ent_zero_sentence_returns_empty_tensors(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Sentence 1 (index 1) has a predicted entity → n_ent might be 0 or 1.
        But if we can find a sentence with no entities, tensors should be empty."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        for idx in range(len(ds)):
            item = ds[idx]
            if item["n_ent"] == 0:
                assert item["input_ids"].numel() == 0
                break


# ---------------------------------------------------------------------------
# Step 13 — RelationCollator
# ---------------------------------------------------------------------------


class TestRelationCollator:
    def test_single_sentence_output_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        # Find a sentence with entities
        item = None
        for idx in range(len(ds)):
            candidate = ds[idx]
            if candidate["n_ent"] > 0:
                item = candidate
                break
        if item is None:
            return  # skip if no entities found

        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=relation_params.max_seq_length,
        )
        batch = collator([item])
        expected_keys = {
            "indices",
            "input_ids",
            "attention_mask",
            "position_ids",
            "sub_positions",
            "rel_labels",
            "ner_labels",
            "sub",
            "obj_token_pos",
            "ent_numbers",
        }
        assert expected_keys == set(batch.keys())

    def test_ent_numbers_tensor(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        items = [ds[i] for i in range(len(ds))]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=relation_params.max_seq_length,
        )
        batch = collator(items)
        assert "ent_numbers" in batch
        assert isinstance(batch["ent_numbers"], torch.Tensor)
        assert len(batch["ent_numbers"]) == len(items)

    def test_multi_sentence_rel_labels_shape(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """For a multi-sentence batch, rel_labels should be (n_sent, max_ent, max_ent)."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        items = [ds[i] for i in range(len(ds))]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=relation_params.max_seq_length,
        )
        batch = collator(items)
        n_sent = len(items)
        max_ent = max(item["n_ent"] for item in items)
        if max_ent > 0:
            assert batch["rel_labels"].shape[0] == n_sent
            assert batch["rel_labels"].shape[1] == max_ent
            assert batch["rel_labels"].shape[2] == max_ent


# ---------------------------------------------------------------------------
# Step 14 — candidate_stats
# ---------------------------------------------------------------------------


class TestCandidateStatsOnDataset:
    def test_candidate_stats_is_candidate_stats(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert isinstance(ds.candidate_stats, CandidateStats)

    def test_perfect_candidates_all_tp(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Fixture has predicted_ner == ner → TP=n_gold, FP=0, FN=0."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        stats = ds.candidate_stats
        assert stats.n_tp == stats.n_gold
        assert stats.n_fp == 0
        assert stats.n_fn == 0
        assert stats.recall == pytest.approx(1.0)
        assert stats.precision == pytest.approx(1.0)

    def test_fp_and_fn_counted_correctly(
        self,
        relation_jsonl_path_with_fp: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Fixture has 1 FP and 1 FN: n_gold=2, n_candidates=2, n_tp=1, n_fp=1, n_fn=1."""
        ds = _make_dataset(
            relation_jsonl_path_with_fp, mock_tokenizer, mock_labels, relation_params
        )
        stats = ds.candidate_stats
        assert stats.n_gold == 2
        assert stats.n_tp == 1
        assert stats.n_fp == 1
        assert stats.n_fn == 1

    def test_candidate_stats_logged(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Logger should be called with candidate stats during _build_index."""
        mock_logger = MagicMock()
        ds = RelationDataset(
            logger=mock_logger,
            tokenizer=mock_tokenizer,
            labels=mock_labels,
            file_path=str(relation_jsonl_path),
            params=relation_params,
        )
        assert ds.candidate_stats is not None
        assert mock_logger.info.called
