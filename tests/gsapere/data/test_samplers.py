"""Tests for samplers.py — Steps 9-10 of TDD plan."""

from __future__ import annotations

from gsapere.data.samplers import (
    BucketSampler,
    create_batches,
    create_shuffled_batches,
    create_size_sorted_batches,
)


# ---------------------------------------------------------------------------
# Step 9 — BucketSampler and batch creation helpers
# ---------------------------------------------------------------------------


class TestCreateBatches:
    def test_covers_all_indices(self) -> None:
        """All sentence indices must appear exactly once across all buckets."""
        sizes = [1, 1, 1, 1, 1, 1]  # 6 sentences, each size 1
        buckets = create_batches(sizes, batch_size=3)
        all_indices = sorted(idx for bucket in buckets for idx in bucket)
        assert all_indices == list(range(len(sizes)))

    def test_no_index_duplicated(self) -> None:
        sizes = [2, 3, 1, 4, 2]
        buckets = create_batches(sizes, batch_size=5)
        all_indices = [idx for bucket in buckets for idx in bucket]
        assert len(all_indices) == len(set(all_indices))

    def test_empty_input(self) -> None:
        buckets = create_batches([], batch_size=4)
        # Empty input → no buckets (or one empty bucket at most)
        assert sum(len(b) for b in buckets) == 0

    def test_single_element(self) -> None:
        buckets = create_batches([5], batch_size=4)
        assert sum(len(b) for b in buckets) == 1
        assert all_indices_covered(buckets, 1)

    def test_large_batch_size_groups_all(self) -> None:
        """When batch_size >= sum(sizes), all items fit in one bucket."""
        sizes = [1, 1, 1]
        buckets = create_batches(sizes, batch_size=10)
        all_idx = [idx for b in buckets for idx in b]
        assert sorted(all_idx) == list(range(3))


class TestCreateShuffledBatches:
    def test_covers_all_indices(self) -> None:
        sizes = [1, 1, 1, 1, 1, 1, 1, 1]
        buckets = create_shuffled_batches(sizes, batch_size=3)
        all_indices = sorted(idx for bucket in buckets for idx in bucket)
        assert all_indices == list(range(len(sizes)))

    def test_no_index_duplicated(self) -> None:
        sizes = [2, 1, 3, 2, 1]
        buckets = create_shuffled_batches(sizes, batch_size=4)
        all_indices = [idx for bucket in buckets for idx in bucket]
        assert len(all_indices) == len(set(all_indices))


class TestCreateSizeSortedBatches:
    def test_covers_all_indices(self) -> None:
        sizes = [5, 1, 3, 2, 4]
        buckets = create_size_sorted_batches(sizes, batch_size=6)
        all_indices = sorted(idx for bucket in buckets for idx in bucket)
        assert all_indices == list(range(len(sizes)))

    def test_no_index_duplicated(self) -> None:
        sizes = [3, 1, 4, 1, 5, 9, 2, 6]
        buckets = create_size_sorted_batches(sizes, batch_size=10)
        all_indices = [idx for bucket in buckets for idx in bucket]
        assert len(all_indices) == len(set(all_indices))


def all_indices_covered(buckets: list[list[int]], n: int) -> bool:
    all_idx = sorted(idx for b in buckets for idx in b)
    return all_idx == list(range(n))


class TestBucketSampler:
    def test_yields_all_buckets(self) -> None:
        buckets = [[0, 1], [2, 3], [4]]
        sampler = BucketSampler(buckets)
        yielded = list(sampler)
        assert len(yielded) == 3

    def test_len_equals_num_buckets(self) -> None:
        buckets = [[0, 1], [2, 3], [4]]
        sampler = BucketSampler(buckets)
        assert len(sampler) == 3

    def test_each_yielded_item_is_bucket(self) -> None:
        buckets = [[0, 1, 2], [3, 4]]
        sampler = BucketSampler(buckets)
        for item in sampler:
            assert isinstance(item, list)

    def test_all_indices_covered(self) -> None:
        sizes = [1] * 10
        batch_size = 3
        buckets = create_batches(sizes, batch_size)
        sampler = BucketSampler(buckets)
        all_idx = sorted(idx for bucket in sampler for idx in bucket)
        assert all_idx == list(range(10))

    def test_shuffle_still_covers_all(self) -> None:
        buckets = [[0, 1], [2, 3], [4, 5]]
        sampler = BucketSampler(buckets, shuffle=True)
        all_idx = sorted(idx for bucket in sampler for idx in bucket)
        assert all_idx == list(range(6))

    def test_empty_buckets(self) -> None:
        sampler = BucketSampler([])
        assert list(sampler) == []
        assert len(sampler) == 0
