"""
collators.py — Collator classes for PrunerDataset and RelationDataset.

Replaces the static/instance collate_fn methods embedded in the dataset classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PrunerCollator:
    """Collator for PrunerDataset batches.

    Stacks all tensor fields; preserves the last `num_metadata_fields` fields
    as plain Python lists (example_index and mentions).

    Replaces ACEDatasetNER.collate_fn (static method).
    """

    num_metadata_fields: int = 2

    def __call__(self, batch: list[list[Any]]) -> list[Any]:
        fields = [x for x in zip(*batch)]
        stacked = [torch.stack(f) for f in fields[: -self.num_metadata_fields]]
        stacked.extend(fields[-self.num_metadata_fields :])
        return stacked


@dataclass
class RelationCollator:
    """Collator for RelationDataset batches.

    Pads batches to the maximum number of entities in the batch and stacks
    all tensor fields into a dict keyed by the batch field names used by
    run_hgnn.py.

    Replaces RelationDataset.collate_fn (instance method).
    """

    tokenizer_pad_id: int
    max_seq_length: int

    def __call__(self, batch: list[Any]) -> dict[str, Any]:
        # Deferred import to avoid circular dependency with relation_dataset
        from hgere.data.relation_dataset import _collate_relation_batch

        return _collate_relation_batch(
            batch, self.tokenizer_pad_id, self.max_seq_length
        )
