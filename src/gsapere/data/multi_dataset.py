"""multi_dataset.py — Utilities for interleaved multi-dataset training.

Provides:
- build_sampling_schedule: compute a deterministic interleave order for batches
  from multiple datasets, respecting temperature-weighted sampling.
- MultiDatasetIterableLoader: wraps per-dataset DataLoaders and yields batches
  in the pre-computed schedule order.
- MultiRelationDataset: thin wrapper around multiple RelationDatasets that
  exposes a unified .loader and len() for the training loop.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from gsapere.data.data_types import CandidateStats
    from gsapere.data.relation_dataset import RelationDataset


def build_sampling_schedule(
    loader_lengths: dict[str, int],
    sampling_weights: dict[str, float],
    temperature: float,
) -> list[str]:
    """Build a batch-interleave schedule for multi-dataset training.

    Each dataset contributes exactly ``loader_lengths[name]`` slots to the
    schedule (one per batch). The *order* in which those slots are interleaved
    is determined by temperature-weighted probabilities:

        p(d) ∝ n_d^temperature * sampling_weight[d]

    where n_d is the number of batches for dataset d.

    Args:
        loader_lengths: Number of batches per dataset (``{name: n_batches}``).
        sampling_weights: Per-dataset scalar weight (``{name: weight}``).
        temperature: Controls how much dataset size influences ordering. At
            temperature=1.0 larger datasets appear proportionally more often
            in each sampling step; at temperature≈0 all datasets are sampled
            uniformly until their quota is exhausted.

    Returns:
        A list of dataset names of length ``sum(loader_lengths.values())``,
        where each name appears exactly ``loader_lengths[name]`` times.
    """
    # Remaining quota for each dataset
    remaining: dict[str, int] = dict(loader_lengths)
    schedule: list[str] = []

    names = list(remaining.keys())

    while any(r > 0 for r in remaining.values()):
        # Compute unnormalised weights for datasets that still have quota
        weights: list[float] = []
        active: list[str] = []
        for name in names:
            if remaining[name] > 0:
                active.append(name)
                w = (remaining[name] ** temperature) * sampling_weights.get(name, 1.0)
                weights.append(w)

        if not active:
            break

        # Sample one dataset proportional to weights
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        chosen = active[-1]
        for name, w in zip(active, weights):
            cumulative += w
            if r <= cumulative:
                chosen = name
                break

        schedule.append(chosen)
        remaining[chosen] -= 1

    return schedule


class MultiRelationDataset:
    """Wrapper around multiple :class:`~gsapere.data.relation_dataset.RelationDataset`
    instances that exposes a unified ``loader`` and ``__len__`` for the training loop.

    Each dataset must already have been built (i.e. ``.build()`` called) before
    being passed here. A new :class:`MultiDatasetIterableLoader` is created from a
    fresh sampling schedule each time :attr:`loader` is accessed.

    Args:
        datasets: Mapping from dataset name to the corresponding dataset object.
        sampling_temperature: Temperature for interleave ordering (see
            :func:`build_sampling_schedule`).
        sampling_weights: Optional per-dataset scalar weights. Defaults to 1.0
            for each dataset.
    """

    def __init__(
        self,
        datasets: dict[str, RelationDataset],
        sampling_temperature: float = 1.0,
        sampling_weights: dict[str, float] | None = None,
    ) -> None:
        self.datasets = datasets
        self.sampling_temperature = sampling_temperature
        self.sampling_weights: dict[str, float] = sampling_weights or {
            name: 1.0 for name in datasets
        }
        self._build_loader()

    def _build_loader(self) -> None:
        loaders: dict[str, DataLoader[Any]] = {
            name: ds.loader for name, ds in self.datasets.items()
        }
        loader_lengths: dict[str, int] = {
            name: len(loader) for name, loader in loaders.items()
        }
        schedule = build_sampling_schedule(
            loader_lengths=loader_lengths,
            sampling_weights=self.sampling_weights,
            temperature=self.sampling_temperature,
        )
        self._loader = MultiDatasetIterableLoader(loaders=loaders, schedule=schedule)

    @property
    def loader(self) -> "MultiDatasetIterableLoader":
        return self._loader

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets.values())

    @property
    def candidate_stats(self) -> dict[str, CandidateStats]:
        return {name: ds.candidate_stats for name, ds in self.datasets.items()}


class MultiDatasetIterableLoader:
    """Yields batches from multiple DataLoaders in a pre-computed schedule order.

    Each step in the schedule names one dataset; the corresponding batch is
    fetched from that dataset's DataLoader. When a DataLoader is exhausted
    before the schedule ends (can happen with non-uniform temperature), it is
    automatically reset.

    Args:
        loaders: Mapping from dataset name to its DataLoader.
        schedule: List of dataset names produced by :func:`build_sampling_schedule`.
    """

    def __init__(
        self,
        loaders: dict[str, DataLoader[Any]],
        schedule: list[str],
    ) -> None:
        self.loaders = loaders
        self.schedule = schedule

    def __len__(self) -> int:
        return len(self.schedule)

    def __iter__(self):  # type: ignore[override]
        iters: dict[str, Any] = {
            name: iter(loader) for name, loader in self.loaders.items()
        }
        for ds_name in self.schedule:
            try:
                batch = next(iters[ds_name])
            except StopIteration:
                iters[ds_name] = iter(self.loaders[ds_name])
                batch = next(iters[ds_name])
            yield batch
