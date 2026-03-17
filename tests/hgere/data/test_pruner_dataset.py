"""Tests for pruner_dataset.py — Steps 6-8 of TDD plan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

from hgere.data.collators import PrunerCollator
from hgere.data.pruner_dataset import PrunerDataset


def _make_dataset(
    file_path: Path,
    tokenizer: MagicMock,
    max_seq_length: int = 64,
    max_pair_length: int = 8,
    max_mention_ori_length: int = 5,
    model_type: str = "bert",
    label_set: str = "scier",
    evaluate: bool = False,
    nocross: bool = False,
) -> PrunerDataset:
    return PrunerDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_pair_length=max_pair_length,
        max_mention_ori_length=max_mention_ori_length,
        model_type=model_type,
        label_set=label_set,
        evaluate=evaluate,
        nocross=nocross,
    )


# ---------------------------------------------------------------------------
# Step 6 — _build_index: span enumeration, recall, gold-in-candidates
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_tot_recall_equals_gold_entities(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """The fixture has exactly 1 gold entity → tot_recall == 1."""
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        assert ds.tot_recall == 1

    def test_gold_in_candidates_at_least_one(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """The gold entity must appear in candidates."""
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        assert ds.n_gold_in_candidates == 1

    def test_data_list_nonempty(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        assert len(ds.data) > 0

    def test_data_chunked_by_max_pair_length(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """All chunks except the last must have exactly max_pair_length examples."""
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_pair_length=4)
        for item in ds.data[:-1]:
            assert len(item["examples"]) == 4

    def test_ner_golden_labels_populated(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        assert len(ds.ner_golden_labels) == 1

    def test_gold_label_in_items(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """At least one item must contain an example with label > 0."""
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        has_positive = any(
            any(ex[1] > 0 for ex in item["examples"]) for item in ds.data
        )
        assert has_positive

    def test_len(self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        assert len(ds) == len(ds.data)

    def test_max_mention_ori_length_limits_spans(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """Reducing max_mention_ori_length should reduce the number of spans."""
        ds_full = _make_dataset(
            single_doc_jsonl_path, mock_tokenizer, max_mention_ori_length=10
        )
        ds_small = _make_dataset(
            single_doc_jsonl_path, mock_tokenizer, max_mention_ori_length=1
        )
        total_full = sum(len(item["examples"]) for item in ds_full.data)
        total_small = sum(len(item["examples"]) for item in ds_small.data)
        assert total_small < total_full


# ---------------------------------------------------------------------------
# Step 7 — __getitem__: tensor shapes and padding
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_item_is_list(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_seq_length=64)
        item = ds[0]
        assert isinstance(item, list)

    def test_input_ids_shape(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        max_seq_length = 64
        max_pair_length = 8
        ds = _make_dataset(
            single_doc_jsonl_path,
            mock_tokenizer,
            max_seq_length=max_seq_length,
            max_pair_length=max_pair_length,
        )
        item = ds[0]
        # input_ids: max_seq_length + 2 * max_pair_length
        expected_len = max_seq_length + 2 * max_pair_length
        assert item[0].shape == (expected_len,)

    def test_labels_length(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        max_pair_length = 8
        ds = _make_dataset(
            single_doc_jsonl_path, mock_tokenizer, max_pair_length=max_pair_length
        )
        item = ds[0]
        assert item[3].shape == (max_pair_length,)

    def test_labels_padding_is_minus_one(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """Positions beyond active spans must be padded with -1."""
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_pair_length=20)
        # Use a large max_pair_length to guarantee padding exists
        item = ds[0]
        labels = item[3]
        n_active = len(ds.data[0]["examples"])
        assert int(labels[n_active]) == -1

    def test_mention_pos_padding_is_zero_zero(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_pair_length=20)
        item = ds[0]
        mention_pos = item[4]
        n_active = len(ds.data[0]["examples"])
        # padding positions should be (0, 0)
        assert tuple(mention_pos[n_active].tolist()) == (0, 0)

    def test_sent_subword_length_field(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        item = ds[0]
        # field 5: sent_subword_length as tensor scalar
        assert item[5].item() > 0

    def test_evaluate_mode_includes_sent_length(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds_eval = _make_dataset(single_doc_jsonl_path, mock_tokenizer, evaluate=True)
        item = ds_eval[0]
        # evaluate=True adds field 6 (sent_length) before metadata
        assert len(item) >= 7

    def test_metadata_fields_are_not_tensors(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer)
        item = ds[0]
        # last 2 fields: example_index (tuple) and mentions (list) — NOT tensors
        assert not isinstance(item[-1], torch.Tensor)
        assert not isinstance(item[-2], torch.Tensor)


# ---------------------------------------------------------------------------
# Step 8 — PrunerCollator
# ---------------------------------------------------------------------------


class TestPrunerCollator:
    def test_collator_stacks_tensors(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_pair_length=4)
        batch = [ds[i] for i in range(min(2, len(ds)))]
        collator = PrunerCollator()
        result = collator(batch)
        # All fields except last 2 should be stacked tensors
        for field in result[:-2]:
            assert isinstance(field, torch.Tensor)

    def test_collator_preserves_metadata_as_lists(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_pair_length=4)
        batch = [ds[i] for i in range(min(2, len(ds)))]
        collator = PrunerCollator()
        result = collator(batch)
        # last 2 fields (example_index, mentions) should remain as lists/tuples
        assert isinstance(result[-1], (list, tuple))
        assert isinstance(result[-2], (list, tuple))

    def test_collator_batch_size_dim(
        self, single_doc_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """Stacked tensors should have batch size as first dim."""
        ds = _make_dataset(single_doc_jsonl_path, mock_tokenizer, max_pair_length=16)
        n = min(2, len(ds))
        batch = [ds[i] for i in range(n)]
        collator = PrunerCollator()
        result = collator(batch)
        assert result[0].shape[0] == n
