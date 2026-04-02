"""Rule-based span pruner: applies learned n-gram pruning patterns.

A span is pruned if ANY of its n-grams (extracted from the boundary-marked word list)
matches a pattern in the prune set.
"""

import json
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

from .statistics import (
    AFTER,
    BEFORE,
    END,
    START,
    _iter_after_ngrams,
    _iter_before_ngrams,
    _iter_ngrams,
)


class RuleBasedPruner:
    """Prunes candidate spans based on learned n-gram patterns.

    A span is pruned when at least one of its n-grams — extracted from
    [<S>] + words + [<E>] — is in the prune pattern set, OR when a before/after
    context n-gram matches (requires context_before / context_after to be passed).

    Parameters
    ----------
    prune_patterns : set of n-gram tuples that always indicate NIL spans.
    max_span_len   : if set, spans longer than this many word tokens are pruned
                     unconditionally (purely a speed optimisation).
    """

    def __init__(
        self,
        prune_patterns: Set[Tuple[str, ...]],
        max_span_len: Optional[int] = None,
    ):
        self.prune_patterns: FrozenSet[Tuple[str, ...]] = frozenset(prune_patterns)
        self.max_span_len: Optional[int] = max_span_len
        # Computed lazily on first call to should_prune (skipped during sweep)
        self._max_before: int = None
        self._max_after: int = None

    def should_prune(
        self,
        words: List[str],
        context_before: List[str] = None,
        context_after: List[str] = None,
    ) -> bool:
        """Return True if this span should be pruned.

        Parameters
        ----------
        words : span tokens
        context_before : sentence tokens immediately before the span (sentence-bounded)
        context_after  : sentence tokens immediately after the span (sentence-bounded)
        """
        if self.max_span_len is not None and len(words) > self.max_span_len:
            return True
        marked = [START] + words + [END]
        for ngram in _iter_ngrams(marked, len(marked)):
            if ngram in self.prune_patterns:
                return True
        if context_before:
            if self._max_before is None:
                self._max_before = max(
                    (
                        sum(1 for t in p if t != BEFORE)
                        for p in self.prune_patterns
                        if p[-1] == BEFORE
                    ),
                    default=0,
                )
            if self._max_before:
                for ngram in _iter_before_ngrams(context_before, self._max_before):
                    if ngram in self.prune_patterns:
                        return True
        if context_after:
            if self._max_after is None:
                self._max_after = max(
                    (
                        sum(1 for t in p if t != AFTER)
                        for p in self.prune_patterns
                        if p[0] == AFTER
                    ),
                    default=0,
                )
            if self._max_after:
                for ngram in _iter_after_ngrams(context_after, self._max_after):
                    if ngram in self.prune_patterns:
                        return True
        return False

    def save(self, path: str) -> None:
        data = {
            "patterns": [list(p) for p in self.prune_patterns],
            "max_span_len": self.max_span_len,
        }
        with open(Path(path), "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "RuleBasedPruner":
        with open(Path(path), "r") as f:
            raw = json.load(f)
        # Backward-compatible: old format is a plain list of patterns.
        if isinstance(raw, list):
            return cls({tuple(p) for p in raw})
        return cls(
            {tuple(p) for p in raw["patterns"]},
            max_span_len=raw.get("max_span_len"),
        )

    def __len__(self) -> int:
        return len(self.prune_patterns)

    def __repr__(self) -> str:
        return f"RuleBasedPruner({len(self.prune_patterns)} patterns)"
