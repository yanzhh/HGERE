"""Integration tests — Step 15 of TDD plan.

Verifies that a full RelationDataset → RelationCollator → DataLoader pipeline
produces batches with all keys expected by run_hgnn.py (TRAIN_KEYS + metadata).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader

from hgere.data.collators import RelationCollator
from hgere.data.relation_dataset import RelationDataset


# Keys that run_hgnn.py accesses on every training batch (see TRAIN_KEYS in run_hgnn.py)
TRAIN_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ner_labels",
    "rel_labels",
    "ent_numbers",
}

# Additional keys used during evaluation / inference
EVAL_EXTRA_KEYS = {"indexs", "obj_token_pos", "sub"}


def _make_dataset(
    file_path: Path,
    tokenizer: MagicMock,
    labels: MagicMock,
    args: MagicMock,
) -> RelationDataset:
    return RelationDataset(
        logger=MagicMock(),
        tokenizer=tokenizer,
        labels=labels,
        file_path=str(file_path),
        args=args,
        max_pair_length=10,
    )


class TestRelationDataloaderIntegration:
    def test_batch_contains_train_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """A collated batch must contain all keys required by run_hgnn.py training."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
        )
        loader = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        batch = next(iter(loader))
        assert TRAIN_KEYS <= set(batch.keys()), (
            f"Missing keys: {TRAIN_KEYS - set(batch.keys())}"
        )

    def test_batch_contains_eval_extra_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """A collated batch must also contain the metadata keys used during eval."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
        )
        loader = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        batch = next(iter(loader))
        assert EVAL_EXTRA_KEYS <= set(batch.keys()), (
            f"Missing keys: {EVAL_EXTRA_KEYS - set(batch.keys())}"
        )

    def test_input_ids_is_tensor(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """input_ids in a batch must be a torch.Tensor."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
        )
        loader = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        batch = next(iter(loader))
        assert isinstance(batch["input_ids"], torch.Tensor)

    def test_ent_numbers_length_equals_batch_size(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """ent_numbers must have one entry per sentence in the batch."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
        )
        loader = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        batch = next(iter(loader))
        assert len(batch["ent_numbers"]) == len(ds)

    def test_rel_labels_shape_batch_x_max_ent_x_max_ent(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """rel_labels must be shaped (n_sent, max_ent, max_ent)."""
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, mock_args)
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=mock_args.max_seq_length,
        )
        loader = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        batch = next(iter(loader))
        n_sent = len(ds)
        max_ent = int(batch["ent_numbers"].max())
        if max_ent > 0:
            assert batch["rel_labels"].shape == (n_sent, max_ent, max_ent)
