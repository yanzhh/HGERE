"""Tests for RelationDataset (reworked) — Steps 11-13 of TDD plan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

from hgere.data.collators import RelationCollator
from hgere.data.relation_dataset import RelationDataset


def _make_dataset(
    file_path: Path,
    tokenizer: MagicMock,
    labels: MagicMock,
    args: MagicMock,
    max_pair_length: int = 10,
    doc_limit: int | None = None,
) -> RelationDataset:
    return RelationDataset(
        logger=MagicMock(),
        tokenizer=tokenizer,
        labels=labels,
        file_path=str(file_path),
        args=args,
        max_pair_length=max_pair_length,
        doc_limit=doc_limit,
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
        mock_args: MagicMock,
    ) -> None:
        """The fixture has 1 doc with 2 sentences → dataset length == 2."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        assert len(ds) == 2

    def test_golden_labels_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        assert len(ds.golden_labels) > 0

    def test_ner_golden_labels_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        assert len(ds.ner_golden_labels) > 0

    def test_global_predicted_ners_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        # (0, 0) = doc_idx=0, sent_idx=0
        assert (0, 0) in ds.global_predicted_ners
        assert len(ds.global_predicted_ners[(0, 0)]) >= 1

    def test_sizes_list_length(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """self.sizes should have one entry per sentence."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
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
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        item = ds[0]
        assert isinstance(item, dict)

    def test_item_has_required_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        expected_keys = {
            "indexs",
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
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        item = ds[0]
        assert expected_keys == set(item.keys())

    def test_input_ids_shape_with_entities(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """For sentence with n_ent > 0, input_ids shape = (n_ent, seq_len+2*n_ent)."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
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
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        item = ds[0]
        n_ent = item["n_ent"]
        if n_ent > 0:
            assert item["rel_labels"].shape == (n_ent, n_ent)

    def test_n_ent_zero_sentence_returns_empty_tensors(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """Sentence 1 (index 1) has a predicted entity → n_ent might be 0 or 1.
        But if we can find a sentence with no entities, tensors should be empty."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
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
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
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
            max_seq_length=mock_args.max_seq_length,
        )
        batch = collator([item])
        expected_keys = {
            "indexs",
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
        mock_args: MagicMock,
    ) -> None:
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        items = [ds[i] for i in range(len(ds))]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
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
        mock_args: MagicMock,
    ) -> None:
        """For a multi-sentence batch, rel_labels should be (n_sent, max_ent, max_ent)."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        items = [ds[i] for i in range(len(ds))]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
        )
        batch = collator(items)
        n_sent = len(items)
        max_ent = max(item["n_ent"] for item in items)
        if max_ent > 0:
            assert batch["rel_labels"].shape[0] == n_sent
            assert batch["rel_labels"].shape[1] == max_ent
            assert batch["rel_labels"].shape[2] == max_ent
