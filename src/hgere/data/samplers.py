"""
samplers.py — Bucket-based samplers and batch-creation helpers.

Extracted from relation_dataset.py.
"""

from __future__ import annotations

import math
import random
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Batch-creation helpers
# ---------------------------------------------------------------------------


def create_batches(n_ents_by_sent: list[int], batch_size: int) -> list[list[int]]:
    """Pack sentences into batches such that the total entity count per batch
    does not exceed batch_size (greedy first-fit)."""
    sort_index = np.arange(len(n_ents_by_sent))
    batches: list[list[int]] = []
    current_sizes: list[int] = []
    current_batch: list[int] = []

    for sent_idx, n_sent_ents in enumerate(n_ents_by_sent):
        if sum(current_sizes) + n_sent_ents <= batch_size:
            current_sizes.append(n_sent_ents)
            current_batch.append(int(sort_index[sent_idx]))
        else:
            if current_batch:
                batches.append(current_batch)
            current_sizes = [n_sent_ents]
            current_batch = [int(sort_index[sent_idx])]

    if current_batch:
        batches.append(current_batch)

    return batches


def create_size_sorted_batches(
    n_ents_by_sent: list[int], batch_size: int
) -> list[list[int]]:
    """Sort sentences by entity count, pack consecutive (similar-size) sentences
    into batches, then shuffle batch order.  This minimises max_ent_num padding."""
    ordered = sorted(enumerate(n_ents_by_sent), key=lambda x: x[1])
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_total = 0

    for idx, size in ordered:
        size = max(size, 1)
        if current_total + size > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_total = size
        else:
            current_batch.append(idx)
            current_total += size

    if current_batch:
        batches.append(current_batch)

    random.shuffle(batches)
    return batches


def create_shuffled_batches(
    n_ents_by_sent: list[int], batch_size: int
) -> list[list[int]]:
    """Shuffle sentences, sort by size, and pack greedily."""
    ordered = [(idx, max(size, 1)) for idx, size in enumerate(n_ents_by_sent)]
    random.shuffle(ordered)
    ordered.sort(key=lambda x: x[1])

    batches: list[list[int]] = []
    sizes: list[int] = []
    while ordered:
        batch, size = _get_batch_filled_up_smallest(ordered, batch_size)
        if batch is not None:
            batches.append(batch)
            sizes.append(size)

    random.shuffle(batches)
    return batches


def _get_batch_filled_up_smallest(
    ordered: list[tuple[int, int]], batch_size: int
) -> tuple[list[int] | None, int]:
    current_batch: list[int] | None = None
    current_size = 0

    if ordered:
        sent_id, size = ordered.pop()
        current_batch = [sent_id]
        current_size = size

    while ordered:
        next_sent_id, next_size = ordered[0]
        if current_size + next_size > batch_size:
            break
        ordered.pop(0)
        assert current_batch is not None
        current_batch.append(next_sent_id)
        current_size += next_size

    return current_batch, current_size


# ---------------------------------------------------------------------------
# Sampler classes
# ---------------------------------------------------------------------------


class BucketSampler(Sampler):
    """Yields pre-built buckets (list-of-index-lists) one bucket per step.

    Replaces ``Sampler`` in relation_dataset.py.
    """

    def __init__(
        self,
        buckets: list[list[int]],
        shuffle: bool = False,
    ) -> None:
        self.buckets = buckets
        self.shuffle = shuffle
        self.n_samples = len(buckets)

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(self.n_samples))
        if self.shuffle:
            random.shuffle(indices)
        return iter([self.buckets[i] for i in indices])

    def __len__(self) -> int:
        return self.n_samples


class DistributedBucketSampler(DistributedSampler):
    """Distributed variant of BucketSampler.

    Replaces ``DistrSampler`` in relation_dataset.py.
    """

    def __init__(
        self,
        buckets: list[list[int]],
        shuffle: bool = False,
    ) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(
                f"Invalid rank {self.rank}, must be in [0, {self.num_replicas - 1}]"
            )
        self.drop_last = False
        self.n_samples = len(buckets)
        self.num_samples = math.ceil(self.n_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.buckets = buckets

    def __iter__(self) -> Iterator[list[int]]:
        indices: list[list[int]] = [
            self.buckets[i] for i in torch.arange(self.n_samples).tolist()
        ]
        # Pad to total_size so all ranks get equal slices
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.n_samples
