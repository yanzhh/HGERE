"""CLI: fit and evaluate the rule-based span pruner.

Workflow
--------
1. Collect n-gram statistics from --train_file
2. Fit a RuleBasedPruner with the given thresholds
3. Enumerate all candidate spans in --dev_file and apply the pruner
4. Evaluate against gold entities and print stats + metrics

Example
-------
    uv run eval-rulebased-pruner \\
        --train_file data/scier/train.jsonl \\
        --dev_file   data/scier/dev.jsonl   \\
        --max_span_length 20                \\
        --min_count_ratio 0.005             \\
        --save rulebased_pruner.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

from tabulate import tabulate

from ..evaluation.pruner import compute_metrics
from ..span_classifier.rulebased import (
    RuleBasedPruner,
    _build_type_mask,
    collect_stats,
    fit,
    load_docs,
    min_count_from_entity_ratio,
)
from ..span_classifier.rulebased.statistics import (
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
    stats, pruner: RuleBasedPruner, min_count: int, purity_threshold: float
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
    print(f"  purity threshold                : {purity_threshold:>10.3f}")
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
        .nlargest(20, "frequency")[["ngram", "frequency", "n_tokens", "purity"]]
        .copy()
    )
    top["type"] = top["ngram"].apply(_pattern_type)
    top["ngram"] = top["ngram"].apply(lambda t: " ".join(t))
    top["purity"] = top["purity"].map("{:.3f}".format)

    print("\n── Top-20 pruning patterns (by frequency) ───────────────────")
    print(
        tabulate(
            top[["ngram", "type", "frequency", "n_tokens", "purity"]].values.tolist(),
            headers=["n-gram", "type", "freq", "n_tok", "purity"],
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


def _sweep(
    stats,
    gold_spans: List,
    all_candidate_spans,
    total_candidates: int,
    pattern_types,
    purity_threshold: float,
    max_ngram_tokens: int,
    target_fn: int,
    n_top: int,
    max_prefix_tokens=None,
    max_suffix_tokens=None,
    max_infix_tokens: int = None,
    max_before_tokens: int = None,
    max_after_tokens: int = None,
):
    """Sweep over min_count values, evaluate on pre-enumerated dev spans."""
    import numpy as np

    # full_only fast path only when no prefix/suffix/context tokens are requested
    full_only = (
        pattern_types == {"full"}
        and max_prefix_tokens is None
        and max_suffix_tokens is None
        and not max_before_tokens
        and not max_after_tokens
    )

    logger.info(
        "Precomputing span keys for sweep (%d spans) ...", len(all_candidate_spans)
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

    max_freq = int(stats["frequency"].max())
    candidates = sorted(
        set(
            [1, 2, 3, 5]
            + list(np.unique(np.round(np.geomspace(1, max_freq, num=60)).astype(int)))
        )
    )
    logger.info("Sweeping %d min_count values ...", len(candidates))

    # Pre-filter stats once: purity + n_tokens + pattern type (constant across the sweep)
    type_mask = _build_type_mask(stats, pattern_types)
    pre_filtered = stats[
        (stats["purity"] >= purity_threshold)
        & (stats["n_tokens"] <= max_ngram_tokens)
        & type_mask
    ].sort_values("frequency", ascending=True)
    freq_arr = pre_filtered["frequency"].to_numpy()
    ngram_arr = pre_filtered["ngram"].to_numpy()
    valid_pattern_set = set(ngram_arr)

    # Build inverted index: pattern → list of span indices that contain it
    from collections import defaultdict

    n_spans = len(precomputed)
    n_gold = len(gold_spans)
    ngram_to_span_idx = defaultdict(list)
    if full_only:
        for span_idx, (coord, key) in enumerate(precomputed):
            if key in valid_pattern_set:
                ngram_to_span_idx[key].append(span_idx)
    else:
        for span_idx, (coord, ngrams_fs) in enumerate(precomputed):
            for ng in ngrams_fs:
                if ng in valid_pattern_set:
                    ngram_to_span_idx[ng].append(span_idx)

    # Initialize per-span match count (number of active patterns matching each span)
    match_count = np.zeros(n_spans, dtype=np.int32)
    for indices in ngram_to_span_idx.values():
        for span_idx in indices:
            match_count[span_idx] += 1
    n_pruned = int(np.sum(match_count > 0))

    # Map gold span coords → span indices; init gold_match_count and fn.
    # Gold spans absent from the candidate set (e.g. longer than max_span_length)
    # are permanent FN regardless of min_count.
    coord_to_span_idx = {coord: i for i, (coord, _) in enumerate(precomputed)}
    trackable_indices = []
    always_fn = 0
    for c in gold_spans:
        idx = coord_to_span_idx.get(c)
        if idx is None:
            always_fn += 1
        else:
            trackable_indices.append(idx)
    n_trackable = len(trackable_indices)
    gold_span_indices = (
        np.array(trackable_indices, dtype=np.int32)
        if n_trackable > 0
        else np.empty(0, dtype=np.int32)
    )
    span_to_gold_idx = np.full(n_spans, -1, dtype=np.int32)
    if n_trackable > 0:
        span_to_gold_idx[gold_span_indices] = np.arange(n_trackable, dtype=np.int32)
    gold_match_count = (
        match_count[gold_span_indices].copy()
        if n_trackable > 0
        else np.empty(0, dtype=np.int32)
    )
    fn = always_fn + int(np.sum(gold_match_count > 0))

    ptr = 0
    rows = []
    for mc in candidates:
        # Drop patterns with freq < mc; update counts incrementally
        while ptr < len(freq_arr) and freq_arr[ptr] < mc:
            ng = ngram_arr[ptr]
            for span_idx in ngram_to_span_idx.get(ng, ()):
                if match_count[span_idx] == 1:
                    n_pruned -= 1
                match_count[span_idx] -= 1
                gold_idx = span_to_gold_idx[span_idx]
                if gold_idx >= 0:
                    if gold_match_count[gold_idx] == 1:
                        fn -= 1
                    gold_match_count[gold_idx] -= 1
            ptr += 1
        n_kept = n_spans - n_pruned
        tp = n_gold - fn
        rows.append(
            {
                "min_count": mc,
                "n_patterns": len(valid_pattern_set) - ptr,
                "recall": (1.0 - fn / n_gold) if n_gold > 0 else 1.0,
                "precision": tp / n_kept if n_kept > 0 else 0.0,
                "pruning_rate": n_pruned / n_spans if n_spans > 0 else 0.0,
                "fn": fn,
            }
        )

    feasible = [r for r in rows if r["fn"] <= target_fn]
    infeasible = [r for r in rows if r["fn"] > target_fn]
    feasible.sort(key=lambda r: r["pruning_rate"], reverse=True)
    infeasible.sort(key=lambda r: r["fn"])
    ranked = feasible + infeasible

    print(f"\n── Sweep (target FN ≤ {target_fn}) — top {n_top} ──────────────────────")
    top = ranked[:n_top]
    table = [
        [
            r["min_count"],
            r["n_patterns"],
            f"{r['recall']:.3f}" + (" ✓" if r["fn"] <= target_fn else ""),
            f"{r['precision']:.3f}",
            f"{r['pruning_rate']:.1%}",
            r["fn"],
        ]
        for r in top
    ]
    print(
        tabulate(
            table,
            headers=["min_count", "patterns", "recall", "precision", "pruning%", "FN"],
            tablefmt="simple",
        )
    )

    if feasible:
        best = feasible[0]
        print(
            f"\n  Best: min_count={best['min_count']}  "
            f"recall={best['recall']:.3f}  pruning={best['pruning_rate']:.1%}  "
            f"FN={best['fn']}"
        )
        return best["min_count"]
    else:
        print(
            f"\n  No configuration achieved FN ≤ {target_fn}. "
            f"Best: FN={ranked[0]['fn']} at min_count={ranked[0]['min_count']}"
        )
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit and evaluate a rule-based span pruner from n-gram statistics."
    )
    p.add_argument("--train_file", required=True, help="Training JSONL file")
    p.add_argument("--dev_file", required=True, help="Dev JSONL file")
    p.add_argument(
        "--max_span_length",
        type=int,
        default=20,
        help="Maximum span length in tokens (default: 20)",
    )
    p.add_argument(
        "--max_ngram",
        type=int,
        default=14,
        help="Maximum n-gram length including <S>/<E> markers (default: 14 = 12 words + 2 markers)",
    )
    p.add_argument(
        "--max_ngram_tokens",
        type=int,
        default=3,
        help="Maximum word tokens in a pruning pattern, boundary markers excluded (default: 3)",
    )
    p.add_argument(
        "--purity_threshold",
        type=float,
        default=1.0,
        help="Minimum NIL fraction for a pattern to be used (default: 1.0 = always NIL)",
    )
    p.add_argument(
        "--min_count",
        type=int,
        default=None,
        help="Minimum frequency for a pattern. If not set, derived from --min_count_ratio.",
    )
    p.add_argument(
        "--min_count_ratio",
        type=float,
        default=0.01,
        help=(
            "Derive min_count as this fraction of total entity n-gram occurrences "
            "(used when --min_count is not set, default: 0.01)"
        ),
    )
    p.add_argument(
        "--pattern_types",
        type=str,
        default="prefix,suffix,full,infix",
        help=(
            "Comma-separated pattern types to use: prefix, suffix, full, infix, before, after. "
            "'before': n-gram of tokens immediately before the span (w_{-k}..w_{-1}, <B>). "
            "'after': n-gram of tokens immediately after the span (<A>, w_{+1}..w_{+k}). "
            "Use 'full' only for the safest setting (exact span match, lowest FN risk). "
            "Default: prefix,suffix,full,infix."
        ),
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=3,
        help="Maximum word tokens per pattern (default: 3).",
    )
    p.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help=(
            "Fraction of training documents to use for statistics collection (default: 0.8). "
            "The remaining documents form an internal dev split used for the --target_recall "
            "sweep, keeping the external --dev_file completely untouched. "
            "Documents are split by index (no shuffling) to keep the split deterministic. "
            "If not set, all training documents are used for statistics and the external "
            "--dev_file is used for the sweep."
        ),
    )
    p.add_argument(
        "--target_fn",
        type=int,
        default=2,
        help=(
            "Sweep over min_count values and select the configuration with the highest "
            "pruning rate that has at most this many false negatives (default: 2)."
        ),
    )
    p.add_argument(
        "--n_top",
        type=int,
        default=15,
        help="Number of configurations to show in the sweep table (default: 15).",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save the learned pruner patterns to this JSON file.",
    )
    return p.parse_args()


def cli():
    args = parse_args()
    pattern_types = {t.strip() for t in args.pattern_types.split(",")}
    max_tokens = args.max_tokens
    max_prefix_tokens = max_tokens
    max_suffix_tokens = max_tokens
    max_infix_tokens = max_tokens if "infix" in pattern_types else None
    max_before_tokens = max_tokens if "before" in pattern_types else None
    max_after_tokens = max_tokens if "after" in pattern_types else None

    # 1. Load training documents and optionally split into stats-train / internal-dev
    logger.info("Loading training documents from %s ...", args.train_file)
    all_train_docs = load_docs(args.train_file)
    logger.info("  %d documents loaded.", len(all_train_docs))

    if args.train_split < 1.0:
        split_idx = int(len(all_train_docs) * args.train_split)
        stats_docs = all_train_docs[:split_idx]
        sweep_docs = all_train_docs[split_idx:]
        logger.info(
            "  Split: %d docs for statistics, %d docs for internal sweep dev.",
            len(stats_docs),
            len(sweep_docs),
        )
        sweep_file = None  # signals to use sweep_docs directly
    else:
        stats_docs = all_train_docs
        sweep_docs = None
        sweep_file = args.dev_file

    # 2. Collect statistics from the stats split
    full_only = pattern_types == {"full"}
    logger.info("Collecting n-gram statistics (max_tokens=%d) ...", max_tokens)
    stats = collect_stats(
        stats_docs,
        max_span_length=args.max_span_length,
        max_ngram=args.max_ngram,
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
            sweep_docs, args.max_span_length
        )
    else:
        logger.info("Loading dev spans from %s ...", sweep_file)
        gold_spans_sweep, all_sweep_spans = _load_dev_spans(
            sweep_file, args.max_span_length
        )

    total_sweep = len(all_sweep_spans)
    logger.info(
        "  %d gold entities, %d candidate spans.", len(gold_spans_sweep), total_sweep
    )

    # 4. Load external dev spans for final evaluation (always the real dev set)
    logger.info("Loading external dev spans from %s ...", args.dev_file)
    gold_spans, all_dev_spans = _load_dev_spans(args.dev_file, args.max_span_length)
    total_candidates = len(all_dev_spans)
    logger.info(
        "  %d gold entities, %d candidate spans.", len(gold_spans), total_candidates
    )

    # 4. Determine min_count
    if args.min_count is not None:
        min_count = args.min_count
    else:
        min_count = min_count_from_entity_ratio(stats, args.min_count_ratio)
        logger.info(
            "  min_count derived from ratio %.4f → %d", args.min_count_ratio, min_count
        )

    # 5. Sweep: search for best min_count on sweep split
    best_min_count = _sweep(
        stats=stats,
        gold_spans=gold_spans_sweep,
        all_candidate_spans=all_sweep_spans,
        total_candidates=total_sweep,
        pattern_types=pattern_types,
        purity_threshold=args.purity_threshold,
        max_ngram_tokens=args.max_ngram_tokens,
        target_fn=args.target_fn,
        n_top=args.n_top,
        max_prefix_tokens=max_prefix_tokens,
        max_suffix_tokens=max_suffix_tokens,
        max_infix_tokens=max_infix_tokens,
        max_before_tokens=max_before_tokens,
        max_after_tokens=max_after_tokens,
    )
    if best_min_count is not None:
        min_count = best_min_count
        logger.info("Using best min_count=%d from sweep.", min_count)

    # 6. Fit final pruner and show full stats
    pruner = fit(
        stats,
        min_count=min_count,
        purity_threshold=args.purity_threshold,
        max_ngram_tokens=args.max_ngram_tokens,
        pattern_types=pattern_types,
    )
    logger.info("  %d pruning patterns learned.", len(pruner))
    _print_stats_summary(stats, pruner, min_count, args.purity_threshold)

    # 7. Evaluate final pruner on external dev set
    kept_spans = _apply_pruner(pruner, all_dev_spans)
    metrics = _eval_metrics(gold_spans, kept_spans, total_candidates)
    span_words = {
        (doc_idx, sent_idx, start, end): words
        for doc_idx, sent_idx, start, end, words, ctx_before, ctx_after in all_dev_spans
    }
    _print_eval_results(
        gold_spans, kept_spans, total_candidates, metrics, span_words=span_words
    )

    # 8. Optionally save pruner
    if args.save:
        pruner.save(args.save)
        logger.info("Pruner saved to %s", args.save)


if __name__ == "__main__":
    cli()
