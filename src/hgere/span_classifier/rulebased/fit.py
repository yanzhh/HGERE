"""Fit a RuleBasedPruner from collected n-gram statistics.

Selection criteria for a pruning pattern:
  1. purity >= purity_threshold  — fraction of occurrences that are NIL spans
  2. frequency >= min_count      — enough evidence to trust the pattern
  3. pattern_types               — restrict to prefix / suffix / infix / full

The default purity_threshold=1.0 gives a hard guarantee: no pattern that ever
appeared in a gold entity span will be used for pruning IN TRAINING.  However,
prefix/suffix/infix patterns can still cause false negatives on unseen dev/test
spans due to train–dev distribution mismatch.  Restricting to pattern_types={"full"}
(exact span match) is the safest option: if a complete span text was never a gold
entity in training it is far less likely to be one at inference time.

min_count can be set as an absolute number or derived as a fraction of total
gold entity spans in the training data (see min_count_from_entity_ratio).
"""

from typing import Set

import pandas as pd

from .filter import RuleBasedPruner
from .statistics import AFTER, BEFORE, END, START

VALID_PATTERN_TYPES = {"prefix", "suffix", "full", "infix", "before", "after"}


def _build_type_mask(stats: pd.DataFrame, pattern_types: Set[str]) -> pd.Series:
    """Vectorized type mask using has_start/has_end/has_before/has_after columns."""
    hs = stats["has_start"]
    he = stats["has_end"]
    hb = stats["has_before"]
    ha = stats["has_after"]
    mask = hs & False  # all-False series aligned to stats index
    if "full" in pattern_types:
        mask |= hs & he
    if "prefix" in pattern_types:
        mask |= hs & ~he
    if "suffix" in pattern_types:
        mask |= ~hs & he
    if "infix" in pattern_types:
        mask |= ~hs & ~he & ~hb & ~ha
    if "before" in pattern_types:
        mask |= hb
    if "after" in pattern_types:
        mask |= ha
    return mask


def fit(
    stats: pd.DataFrame,
    min_count: int,
    purity_threshold: float = 1.0,
    max_ngram_tokens: int = 12,
    pattern_types: Set[str] = None,
) -> RuleBasedPruner:
    """Select pruning patterns from n-gram statistics.

    Parameters
    ----------
    stats : DataFrame returned by collect_stats()
    min_count : minimum frequency for a pattern to be trusted
    purity_threshold : minimum fraction of NIL occurrences (default 1.0)
    max_ngram_tokens : only consider patterns with at most this many word tokens
        (boundary markers <S>/<E> are not counted)
    pattern_types : set of allowed pattern types — any subset of
        {"prefix", "suffix", "full", "infix"}.  Default: all four types.
        Use {"full"} for the safest setting (exact span match only), which
        minimises false negatives on unseen data at the cost of fewer patterns.

    Returns
    -------
    RuleBasedPruner ready to call .should_prune(words)
    """
    if pattern_types is None:
        pattern_types = VALID_PATTERN_TYPES
    unknown = pattern_types - VALID_PATTERN_TYPES
    if unknown:
        raise ValueError(f"Unknown pattern types: {unknown}. Choose from {VALID_PATTERN_TYPES}")

    type_mask = _build_type_mask(stats, pattern_types)
    mask = (
        (stats["purity"] >= purity_threshold)
        & (stats["frequency"] >= min_count)
        & (stats["n_tokens"] <= max_ngram_tokens)
        & type_mask
    )
    patterns = set(stats.loc[mask, "ngram"])
    return RuleBasedPruner(patterns)


def min_count_from_entity_ratio(
    stats: pd.DataFrame,
    ratio: float,
) -> int:
    """Derive min_count as ratio * total_entity_ngram_occurrences.

    Ensures the frequency bar scales with dataset size.
    E.g. ratio=0.01 means a pattern must appear at least 1% as often
    as the total number of entity n-gram occurrences.
    """
    total_entity = stats["count_entity"].sum()
    return max(1, int(ratio * total_entity))
