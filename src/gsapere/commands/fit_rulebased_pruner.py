"""CLI: fit and evaluate the rule-based span pruner.

Workflow
--------
1. Collect n-gram statistics from --train_file
2. Fit a RuleBasedPruner with the given thresholds
3. Enumerate all candidate spans in --dev_file and apply the pruner
4. Evaluate against gold entities and print stats + metrics

Example (YAML config)
---------------------
    uv run gsapere-fit-rulebased-pruner configs/train/scier/fit_rulebased_pruner.yaml

Example (CLI flags)
-------------------
    uv run gsapere-fit-rulebased-pruner \\
        --train_file data/scier/train.jsonl \\
        --dev_file   data/scier/dev.jsonl   \\
        --max_span_length 20                \\
        --min_count_ratio 0.005             \\
        --save rulebased_pruner.json
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..config import load_yaml_strict
from ._cli_utils import load_config_from_argv

import numpy as np
import pandas as pd
from tabulate import tabulate

from ..evaluation.pruner import compute_metrics
from ..pre_filter import (
    RuleBasedPruner,
    _build_type_mask,
    collect_stats,
    load_docs,
    min_count_from_entity_ratio,
)
from ..pre_filter.statistics import (
    AFTER,
    BEFORE,
    END,
    SENT_END,
    SENT_START,
    START,
    _iter_after_ngrams,
    _iter_before_ngrams,
    _iter_boundary_ngrams,
    _iter_infix_ngrams,
    _iter_ngrams,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Dev evaluation
# ---------------------------------------------------------------------------


def _load_dev_spans(
    source,
    max_span_length: int,
) -> Tuple[List, List]:
    """Enumerate all candidate spans from a file path or list of doc dicts.

    Returns
    -------
    gold_spans  : list of (doc_idx, sent_idx, start, end) for gold entities
    all_spans   : list of (doc_idx, sent_idx, start, end, words, ctx_before, ctx_after)
                  where ctx_before/ctx_after are sentence-bounded context word lists
    """
    gold_spans = []
    all_spans = []

    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
    else:
        docs = source

    for doc_idx, data in enumerate(docs):
        sentences: List[List[str]] = data["sentences"]
        ner_gold = data["ner"]

        words: List[str] = []
        sentence_boundaries = [0]
        for sentence in sentences:
            words.extend(sentence)
            sentence_boundaries.append(len(words))

        for sent_idx, sent_ner in enumerate(ner_gold):
            for start, end, label in sent_ner:
                if label != "NIL":
                    gold_spans.append((doc_idx, sent_idx, start, end))

        for sent_idx in range(len(sentences)):
            sent_start = sentence_boundaries[sent_idx]
            sent_end = sentence_boundaries[sent_idx + 1]
            for span_start in range(sent_start, sent_end):
                for span_end in range(
                    span_start, min(sent_end, span_start + max_span_length)
                ):
                    span_words = words[span_start : span_end + 1]
                    ctx_before = [SENT_START] + words[sent_start:span_start]
                    ctx_after = words[span_end + 1 : sent_end] + [SENT_END]
                    all_spans.append(
                        (
                            doc_idx,
                            sent_idx,
                            span_start,
                            span_end,
                            span_words,
                            ctx_before,
                            ctx_after,
                        )
                    )

    return gold_spans, all_spans


def _apply_pruner(pruner: RuleBasedPruner, all_spans: List) -> List:
    """Return the subset of all_spans not pruned by pruner."""
    return [
        (doc_idx, sent_idx, start, end)
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_spans
        if not pruner.should_prune(words, ctx_before, ctx_after)
    ]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _pattern_type(ngram: tuple) -> str:
    if ngram[-1] == BEFORE:
        return "before"
    if ngram[0] == AFTER:
        return "after"
    has_start = ngram[0] == START
    has_end = ngram[-1] == END
    if has_start and has_end:
        return "full"
    if has_start:
        return "prefix"
    if has_end:
        return "suffix"
    return "infix"


def _print_stats_summary(
    stats, pruner: RuleBasedPruner, min_count: int, max_entity_rate: float
):
    total_ngrams = len(stats)
    n_patterns = len(pruner)

    # Breakdown by type
    type_counts = {
        "prefix": 0,
        "suffix": 0,
        "full": 0,
        "infix": 0,
        "before": 0,
        "after": 0,
    }
    for p in pruner.prune_patterns:
        type_counts[_pattern_type(p)] += 1

    print("\n── Training statistics ──────────────────────────────────────")
    print(f"  Unique n-grams in training data : {total_ngrams:>10,}")
    print(f"  min_count threshold             : {min_count:>10,}")
    print(f"  max entity rate                 : {max_entity_rate:>10.3f}")
    print(f"  Pruning patterns learned        : {n_patterns:>10,}")
    print()
    print(
        tabulate(
            [[t, c] for t, c in type_counts.items()],
            headers=["Pattern type", "Count"],
            tablefmt="simple",
        )
    )

    # Top-20 patterns by frequency
    prune_set = pruner.prune_patterns
    top = (
        stats[stats["ngram"].isin(prune_set)]
        .nlargest(20, "frequency")[["ngram", "frequency", "n_tokens", "entity_rate"]]
        .copy()
    )
    top["type"] = top["ngram"].apply(_pattern_type)
    top["ngram"] = top["ngram"].apply(lambda t: " ".join(t))
    top["entity_rate"] = top["entity_rate"].map("{:.3f}".format)

    print("\n── Top-20 pruning patterns (by frequency) ───────────────────")
    print(
        tabulate(
            top[
                ["ngram", "type", "frequency", "n_tokens", "entity_rate"]
            ].values.tolist(),
            headers=["n-gram", "type", "freq", "n_tok", "entity_rate"],
            tablefmt="simple",
        )
    )


def _eval_metrics(gold_spans, kept_spans, total_candidates) -> dict:
    metrics = compute_metrics(gold_spans, kept_spans)
    n_pruned = total_candidates - len(kept_spans)
    metrics["pruning_rate"] = n_pruned / total_candidates if total_candidates else 0.0
    metrics["n_pruned"] = n_pruned
    metrics["total_candidates"] = total_candidates
    return metrics


def _print_eval_results(
    gold_spans: List,
    kept_spans: List,
    total_candidates: int,
    metrics: dict,
    span_words: dict = None,
):
    """Print evaluation results.

    Parameters
    ----------
    span_words : optional dict mapping (doc_idx, sent_idx, start, end) → List[str].
        When provided, pruned gold entities are listed with their text.
    """
    n_pruned = metrics["n_pruned"]
    print("\n── Dev evaluation ───────────────────────────────────────────")
    print(
        tabulate(
            [
                ["Total candidate spans", f"{total_candidates:,}"],
                ["Spans pruned", f"{n_pruned:,}  ({metrics['pruning_rate']:.1%})"],
                ["Spans kept", f"{len(kept_spans):,}"],
                ["Gold entities", f"{len(gold_spans):,}"],
                ["Gold entities pruned (FN)", f"{metrics['fn']:,}"],
            ],
            tablefmt="simple",
        )
    )
    print()
    print(
        tabulate(
            [
                ["recall", f"{metrics['recall']:.3f}"],
                ["precision", f"{metrics['precision']:.3f}"],
                ["f1", f"{metrics['f1']:.3f}"],
                ["false_positive_share", f"{metrics['false_positive_share']:.3f}"],
            ],
            headers=["metric", "value"],
            tablefmt="simple",
        )
    )

    if metrics["fn"] > 0 and span_words is not None:
        kept_set = set(kept_spans)
        pruned_gold = [s for s in gold_spans if s not in kept_set]
        print(
            f"\n── Pruned gold entities ({len(pruned_gold)}) ─────────────────────────"
        )
        rows = []
        for coord in pruned_gold:
            words = span_words.get(coord, [])
            rows.append([" ".join(words), coord[0], coord[2], coord[3]])
        print(
            tabulate(rows, headers=["text", "doc", "start", "end"], tablefmt="simple")
        )


def _print_pruned_gold_details(
    gold_spans: List,
    kept_spans: List,
    span_words: Dict,
    span_context: Dict,
    pruner: RuleBasedPruner,
    stats: pd.DataFrame,
) -> None:
    """For each pruned gold entity, list every matching pattern with its training stats."""
    kept_set = set(kept_spans)
    pruned_gold = [s for s in gold_spans if s not in kept_set]
    if not pruned_gold:
        return

    stats_lookup: Dict = stats.set_index("ngram")[
        ["frequency", "entity_rate", "n_tokens"]
    ].to_dict("index")

    # Precompute context limits (mirrors RuleBasedPruner lazy init)
    max_before = max(
        (
            sum(1 for t in p if t != BEFORE)
            for p in pruner.prune_patterns
            if p[-1] == BEFORE
        ),
        default=0,
    )
    max_after = max(
        (
            sum(1 for t in p if t != AFTER)
            for p in pruner.prune_patterns
            if p[0] == AFTER
        ),
        default=0,
    )

    print(f"\n── Pruned gold entities — matched patterns ({len(pruned_gold)}) ────────")
    rows = []
    for coord in pruned_gold:
        words = span_words.get(coord, [])
        ctx_before, ctx_after = span_context.get(coord, ([], []))

        matching: List = []
        marked = [START] + list(words) + [END]
        for ng in _iter_ngrams(marked, len(marked)):
            if ng in pruner.prune_patterns:
                matching.append(ng)
        if max_before and ctx_before:
            for ng in _iter_before_ngrams(ctx_before, max_before):
                if ng in pruner.prune_patterns:
                    matching.append(ng)
        if max_after and ctx_after:
            for ng in _iter_after_ngrams(ctx_after, max_after):
                if ng in pruner.prune_patterns:
                    matching.append(ng)

        # Sort: highest entity_rate first (most suspicious), then highest frequency
        matching.sort(
            key=lambda ng: (
                -stats_lookup.get(ng, {}).get("entity_rate", 0.0),
                -stats_lookup.get(ng, {}).get("frequency", 0),
            )
        )

        span_text = " ".join(words)
        for i, ng in enumerate(matching):
            s = stats_lookup.get(ng, {})
            rows.append(
                [
                    span_text if i == 0 else "",
                    coord[0] if i == 0 else "",
                    " ".join(ng),
                    _pattern_type(ng),
                    s.get("frequency", "?"),
                    f"{s.get('entity_rate', 0.0):.4f}",
                ]
            )

    print(
        tabulate(
            rows,
            headers=["entity", "doc", "pattern", "type", "freq", "entity_rate"],
            tablefmt="simple",
        )
    )


def _precompute_span_keys(
    all_spans,
    full_only: bool,
    max_prefix_tokens=None,
    max_suffix_tokens=None,
    max_infix_tokens: int = None,
    max_before_tokens: int = None,
    max_after_tokens: int = None,
):
    """Precompute per-span pattern keys for fast sweep lookups.

    full_only (and no prefix/suffix) → one tuple (START, *words, END) per span
    boundary mode (prefix/suffix set) → frozenset of boundary + infix n-grams
    general                           → frozenset of all n-grams

    Before/after context n-grams are always added to a frozenset (not compatible
    with the full_only tuple fast path, which falls back to frozenset when needed).
    """
    need_context = max_before_tokens or max_after_tokens

    def _context_keys(ctx_before, ctx_after):
        keys = set()
        if max_before_tokens:
            keys.update(_iter_before_ngrams(ctx_before, max_before_tokens))
        if max_after_tokens:
            keys.update(_iter_after_ngrams(ctx_after, max_after_tokens))
        return keys

    if (
        full_only
        and max_prefix_tokens is None
        and max_suffix_tokens is None
        and not need_context
    ):
        return [
            ((doc_idx, sent_idx, start, end), (START,) + tuple(words) + (END,))
            for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_spans
        ]
    elif max_prefix_tokens is not None or max_suffix_tokens is not None:
        mp = max_prefix_tokens or 0
        ms = max_suffix_tokens or 0
        result = []
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_spans:
            marked = [START] + list(words) + [END]
            keys = frozenset(_iter_boundary_ngrams(marked, mp, ms))
            if max_infix_tokens is not None:
                keys |= frozenset(_iter_infix_ngrams(list(words), max_infix_tokens))
            if need_context:
                keys |= frozenset(_context_keys(ctx_before, ctx_after))
            result.append(((doc_idx, sent_idx, start, end), keys))
        return result
    else:
        result = []
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_spans:
            keys = frozenset(
                _iter_ngrams([START] + list(words) + [END], len(words) + 2)
            )
            if need_context:
                keys |= frozenset(_context_keys(ctx_before, ctx_after))
            result.append(((doc_idx, sent_idx, start, end), keys))
        return result


def _apply_patterns_fast(precomputed, patterns, full_only: bool) -> List:
    """Apply a pattern set to precomputed span keys without regenerating n-grams."""
    if full_only:
        return [coord for coord, key in precomputed if key not in patterns]
    else:
        return [coord for coord, ngrams in precomputed if ngrams.isdisjoint(patterns)]


def _greedy_select(
    stats: pd.DataFrame,
    gold_spans: List,
    all_candidate_spans: List,
    pattern_types: Set,
    max_entity_rate: float,
    min_count: int,
    target_fn: int,
    max_prefix_tokens: int = None,
    max_suffix_tokens: int = None,
    max_infix_tokens: int = None,
    max_before_tokens: int = None,
    max_after_tokens: int = None,
) -> RuleBasedPruner:
    """Greedy pattern selection ordered by (per-pattern FN asc, TN desc).

    Patterns are pre-filtered by max_entity_rate, min_count, and pattern_types.
    Any pattern that alone exceeds target_fn gold spans is removed.  The
    remaining patterns are added greedily — lowest individual FN first, highest
    TN as tiebreaker — until the cumulative FN budget would be exceeded.

    Returns a RuleBasedPruner containing the selected patterns.
    """
    full_only = (
        pattern_types == {"full"}
        and max_prefix_tokens is None
        and max_suffix_tokens is None
        and not max_before_tokens
        and not max_after_tokens
    )

    logger.info(
        "Precomputing span keys for greedy selection (%d spans) ...",
        len(all_candidate_spans),
    )
    precomputed = _precompute_span_keys(
        all_candidate_spans,
        full_only,
        max_prefix_tokens,
        max_suffix_tokens,
        max_infix_tokens=max_infix_tokens,
        max_before_tokens=max_before_tokens,
        max_after_tokens=max_after_tokens,
    )

    n_spans = len(precomputed)

    # Map coords → span indices; mark gold spans
    coord_to_span_idx: Dict = {coord: i for i, (coord, _) in enumerate(precomputed)}
    span_is_gold = np.zeros(n_spans, dtype=bool)
    always_fn = 0
    for c in gold_spans:
        idx = coord_to_span_idx.get(c)
        if idx is None:
            always_fn += 1
        else:
            span_is_gold[idx] = True

    # Pre-filter: entity_rate + min_count + type
    type_mask = _build_type_mask(stats, pattern_types)
    pre_filtered = stats[
        (stats["entity_rate"] <= max_entity_rate)
        & (stats["frequency"] >= min_count)
        & type_mask
    ]
    valid_pattern_set: Set = set(pre_filtered["ngram"].to_numpy())
    n_eligible = len(valid_pattern_set)

    # Build inverted index over eligible patterns
    ngram_to_span_idx: Dict = defaultdict(list)
    if full_only:
        for span_idx, (coord, key) in enumerate(precomputed):
            if key in valid_pattern_set:
                ngram_to_span_idx[key].append(span_idx)
    else:
        for span_idx, (coord, ngrams_fs) in enumerate(precomputed):
            for ng in ngrams_fs:
                if ng in valid_pattern_set:
                    ngram_to_span_idx[ng].append(span_idx)

    # Per-pattern FN and TN counts
    pattern_fn: Dict = {}
    pattern_tn: Dict = {}
    for ng, indices in ngram_to_span_idx.items():
        fn_count = int(np.sum(span_is_gold[indices]))
        pattern_fn[ng] = fn_count
        pattern_tn[ng] = len(indices) - fn_count

    # Remove patterns that alone would exceed the FN budget
    dangerous = {ng for ng, fn in pattern_fn.items() if fn > target_fn}
    if dangerous:
        for ng in dangerous:
            del ngram_to_span_idx[ng]
            del pattern_fn[ng]
            del pattern_tn[ng]
        logger.info(
            "  Removed %d patterns exceeding per-pattern FN > %d.",
            len(dangerous),
            target_fn,
        )

    # Sort by (per_pattern_fn asc, per_pattern_tn desc)
    sorted_patterns = sorted(
        pattern_fn.keys(),
        key=lambda ng: (pattern_fn[ng], -pattern_tn[ng]),
    )

    # Greedy selection — track which gold spans are already pruned to avoid
    # double-counting FN when multiple patterns match the same gold span.
    selected_patterns: Set = set()
    gold_pruned: Set = set()  # gold span indices already marked as FN
    span_pruned = np.zeros(n_spans, dtype=bool)
    skipped = 0

    for ng in sorted_patterns:
        indices = ngram_to_span_idx[ng]
        new_fn = sum(
            1 for idx in indices if span_is_gold[idx] and idx not in gold_pruned
        )
        if len(gold_pruned) + new_fn <= target_fn:
            selected_patterns.add(ng)
            for idx in indices:
                span_pruned[idx] = True
                if span_is_gold[idx]:
                    gold_pruned.add(idx)
        else:
            skipped += 1

    # Patterns that pass all criteria but match no span in the sweep set have 0
    # observed FN risk — include them unconditionally.
    # Exclude `dangerous`: they were removed from ngram_to_span_idx but are still
    # in valid_pattern_set, so without this subtraction they would be re-added here.
    unseen_safe = valid_pattern_set - set(ngram_to_span_idx.keys()) - dangerous
    selected_patterns |= unseen_safe

    print(f"\n── Greedy selection (target FN ≤ {target_fn}) ────────────────────────")
    print(
        tabulate(
            [
                ["Eligible patterns (pre-filter)", n_eligible],
                ["Removed (per-pattern FN > target)", len(dangerous)],
                ["Considered (appear in sweep)", len(sorted_patterns)],
                ["Selected (greedy)", len(selected_patterns) - len(unseen_safe)],
                ["Skipped (would exceed FN budget)", skipped],
                ["Added (unseen in sweep, 0 FN risk)", len(unseen_safe)],
                ["Total selected", len(selected_patterns)],
            ],
            tablefmt="simple",
        )
    )

    return RuleBasedPruner(selected_patterns)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FitRulebasedPrunerConfig(BaseModel):
    """Configuration for fitting and evaluating the rule-based span pruner."""

    model_config = ConfigDict(extra="forbid")

    train_file: str = Field(description="Training JSONL file.")
    dev_file: str = Field(description="Dev JSONL file.")
    max_span_length: int = Field(
        default=20,
        description="Maximum span length in tokens.",
    )
    max_ngram: int = Field(
        default=14,
        description=(
            "Maximum n-gram length including <S>/<E> markers "
            "(default: 14 = 12 words + 2 markers)."
        ),
    )
    max_entity_rate: float = Field(
        default=0.0,
        description=(
            "Maximum fraction of entity occurrences for a pattern to be used. "
            "0.0 = pattern must never appear inside entities."
        ),
    )
    min_count: Optional[int] = Field(
        default=None,
        description=(
            "Minimum frequency for a pattern. If not set, derived from min_count_ratio."
        ),
    )
    min_count_ratio: float = Field(
        default=0.01,
        description=(
            "Derive min_count as this fraction of total entity n-gram occurrences "
            "(used when min_count is not set)."
        ),
    )
    pattern_types: str = Field(
        default="prefix,suffix,full,infix,before,after",
        description=(
            "Comma-separated pattern types to use: prefix, suffix, full, infix, before, after. "
            "'before': n-gram of tokens immediately before the span (w_{-k}..w_{-1}, <B>). "
            "'after': n-gram of tokens immediately after the span (<A>, w_{+1}..w_{+k}). "
            "Use 'full' only for the safest setting (exact span match, lowest FN risk)."
        ),
    )
    max_tokens: int = Field(
        default=3,
        description="Maximum word tokens per pattern.",
    )
    train_split: float = Field(
        default=0.8,
        description=(
            "Fraction of training documents to use for statistics collection. "
            "The remaining documents form an internal dev split used for greedy selection, "
            "keeping the external dev_file completely untouched. "
            "Documents are shuffled before splitting (controlled by seed). "
            "Set to 1.0 to use all training documents for statistics and the external "
            "dev_file for the sweep."
        ),
    )
    seed: int = Field(
        default=42,
        description="Random seed for shuffling training documents before the stats/sweep split.",
    )
    target_fn: int = Field(
        default=2,
        description=(
            "Select the configuration with the highest pruning rate that has "
            "at most this many false negatives."
        ),
    )
    n_top: int = Field(
        default=15,
        description="Number of configurations to show in the sweep table.",
    )
    save: Optional[str] = Field(
        default=None,
        description="If set, save the learned pruner patterns to this JSON file.",
    )

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "FitRulebasedPrunerConfig":
        """Load config from a YAML file."""
        data: dict = load_yaml_strict(path)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Fit and evaluate the rule-based span pruner.

    Pass a YAML config as a positional argument or supply all parameters as
    CLI flags::

        gsapere-fit-rulebased-pruner configs/train/scier/fit_rulebased_pruner.yaml
        gsapere-fit-rulebased-pruner --train_file train.jsonl --dev_file dev.jsonl
    """
    if argv is None:
        argv = sys.argv[1:]

    config = load_config_from_argv(
        argv,
        FitRulebasedPrunerConfig,
        description="Fit and evaluate a rule-based span pruner from n-gram statistics.",
    )

    pattern_types = {t.strip() for t in config.pattern_types.split(",")}
    max_tokens = config.max_tokens
    max_prefix_tokens = max_tokens
    max_suffix_tokens = max_tokens
    max_infix_tokens = max_tokens if "infix" in pattern_types else None
    max_before_tokens = max_tokens if "before" in pattern_types else None
    max_after_tokens = max_tokens if "after" in pattern_types else None

    # 1. Load training documents and optionally split into stats-train / internal-dev
    logger.info("Loading training documents from %s ...", config.train_file)
    all_train_docs = load_docs(config.train_file)
    logger.info("  %d documents loaded.", len(all_train_docs))

    if config.train_split < 1.0:
        import random

        rng = random.Random(config.seed)
        all_train_docs = all_train_docs.copy()
        rng.shuffle(all_train_docs)
        split_idx = int(len(all_train_docs) * config.train_split)
        stats_docs = all_train_docs[:split_idx]
        sweep_docs = all_train_docs[split_idx:]
        logger.info(
            "  Shuffled (seed=%d), split: %d docs for statistics, %d docs for internal sweep.",
            config.seed,
            len(stats_docs),
            len(sweep_docs),
        )
        sweep_file = None  # signals to use sweep_docs directly
    else:
        stats_docs = all_train_docs
        sweep_docs = None
        sweep_file = config.dev_file

    # 2. Collect statistics from the stats split
    full_only = pattern_types == {"full"}
    logger.info("Collecting n-gram statistics (max_tokens=%d) ...", max_tokens)
    stats = collect_stats(
        stats_docs,
        max_span_length=config.max_span_length,
        max_ngram=config.max_ngram,
        full_only=full_only,
        max_prefix_tokens=max_prefix_tokens,
        max_suffix_tokens=max_suffix_tokens,
        max_infix_tokens=max_infix_tokens,
        max_before_tokens=max_before_tokens,
        max_after_tokens=max_after_tokens,
    )
    logger.info("  %d unique n-grams collected.", len(stats))

    # 3. Load sweep spans (internal split or external dev file)
    if sweep_docs is not None:
        logger.info(
            "Enumerating spans from internal dev split (%d docs) ...", len(sweep_docs)
        )
        gold_spans_sweep, all_sweep_spans = _load_dev_spans(
            sweep_docs, config.max_span_length
        )
    else:
        logger.info("Loading dev spans from %s ...", sweep_file)
        gold_spans_sweep, all_sweep_spans = _load_dev_spans(
            sweep_file, config.max_span_length
        )

    total_sweep = len(all_sweep_spans)
    logger.info(
        "  %d gold entities, %d candidate spans.", len(gold_spans_sweep), total_sweep
    )

    # 4. Load external dev spans for final evaluation (always the real dev set)
    logger.info("Loading external dev spans from %s ...", config.dev_file)
    gold_spans, all_dev_spans = _load_dev_spans(config.dev_file, config.max_span_length)
    total_candidates = len(all_dev_spans)
    logger.info(
        "  %d gold entities, %d candidate spans.", len(gold_spans), total_candidates
    )

    # 5. Determine min_count floor
    if config.min_count is not None:
        min_count = config.min_count
    else:
        min_count = min_count_from_entity_ratio(stats, config.min_count_ratio)
        logger.info(
            "  min_count derived from ratio %.4f → %d",
            config.min_count_ratio,
            min_count,
        )

    # 6. Greedy pattern selection on sweep split
    pruner = _greedy_select(
        stats=stats,
        gold_spans=gold_spans_sweep,
        all_candidate_spans=all_sweep_spans,
        pattern_types=pattern_types,
        max_entity_rate=config.max_entity_rate,
        min_count=min_count,
        target_fn=config.target_fn,
        max_prefix_tokens=max_prefix_tokens,
        max_suffix_tokens=max_suffix_tokens,
        max_infix_tokens=max_infix_tokens,
        max_before_tokens=max_before_tokens,
        max_after_tokens=max_after_tokens,
    )
    logger.info("  %d pruning patterns selected.", len(pruner))
    _print_stats_summary(stats, pruner, min_count, config.max_entity_rate)

    # 7. Evaluate on sweep set (same data used for greedy selection)
    sweep_label = (
        "sweep (internal train split)" if sweep_docs is not None else "sweep (dev file)"
    )
    kept_sweep = _apply_pruner(pruner, all_sweep_spans)
    metrics_sweep = _eval_metrics(gold_spans_sweep, kept_sweep, total_sweep)
    sweep_words = {
        (doc_idx, sent_idx, start, end): words
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_sweep_spans
    }
    sweep_context = {
        (doc_idx, sent_idx, start, end): (ctx_before, ctx_after)
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_sweep_spans
    }
    print(f"\n{sweep_label}:")
    _print_eval_results(
        gold_spans_sweep, kept_sweep, total_sweep, metrics_sweep, span_words=sweep_words
    )
    _print_pruned_gold_details(
        gold_spans_sweep, kept_sweep, sweep_words, sweep_context, pruner, stats
    )

    # 8. Evaluate final pruner on external dev set
    kept_spans = _apply_pruner(pruner, all_dev_spans)
    metrics = _eval_metrics(gold_spans, kept_spans, total_candidates)
    span_words = {
        (doc_idx, sent_idx, start, end): words
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_dev_spans
    }
    span_context = {
        (doc_idx, sent_idx, start, end): (ctx_before, ctx_after)
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_dev_spans
    }
    print("\nexternal dev:")
    _print_eval_results(
        gold_spans, kept_spans, total_candidates, metrics, span_words=span_words
    )
    _print_pruned_gold_details(
        gold_spans, kept_spans, span_words, span_context, pruner, stats
    )

    # 9. Optionally save pruner
    if config.save:
        pruner.max_span_len = config.max_span_length
        pruner.save(config.save)
        logger.info(
            "Pruner saved to %s (max_span_len=%d)", config.save, config.max_span_length
        )


cli = main


if __name__ == "__main__":
    main()
