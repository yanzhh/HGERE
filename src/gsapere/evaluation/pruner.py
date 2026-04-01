"""
Pruner evaluation utilities.

The pruner is a **binary** entity/non-entity classifier.

Predictions file format (jsonl):
  Each line is a document dict with:
    - 'ner':                  list[list[[start, end, type]]]            gold entities per sentence
    - 'predicted_ner_proba':  list[list[[start, end, prob, gold_type]]] candidate spans with
                              the pruner's entity probability score and the *gold* label for reference.

Threshold logic
---------------
A span (doc_id, sent_idx, start, end) is predicted as an entity candidate when
its score prob >= threshold.  The gold_type stored alongside is NOT used for
filtering — it is only used for per-type recall breakdown.

Top-K selection (original paper approach)
-----------------------------------------
Spans are ranked by prob and the top K are kept per sentence, where:
    K = max(lmin, min(lambda * n, lmax))
with n = sentence length (number of tokens), lambda = topk_ratio,
lmin = min_mentions_num, lmax = max_mentions_num.

Matching is span-level (doc_id, sent_idx, start, end) because the pruner does
not predict entity types.
"""

import json
from itertools import product
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import precision_recall_fscore_support  # noqa: F401 (available for callers)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_predictions(filepath: str) -> List[dict]:
    """Load entity predictions from a jsonl file."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _gold_spans(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, start, end) for every gold entity."""
    spans = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_ner in enumerate(doc["ner"]):
            for start, end, _etype in sent_ner:
                spans.append((doc_id, sent_idx, start, end))
    return spans


def _gold_entities(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, start, end, type) for every gold entity (for per-type breakdown)."""
    entities = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_ner in enumerate(doc["ner"]):
            for start, end, etype in sent_ner:
                entities.append((doc_id, sent_idx, start, end, etype))
    return entities


def _all_candidate_spans(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, start, end) for every span in predicted_ner_proba (pool)."""
    return _predicted_spans(docs, threshold=0.0)


def _predicted_spans(docs: List[dict], threshold: float) -> List[Tuple]:
    """Return (doc_id, sent_idx, start, end) for all candidate spans above threshold.

    The pruner is a binary classifier: a span is predicted as an entity candidate
    when its score prob >= threshold, regardless of the stored gold_type label.
    """
    spans = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_preds in enumerate(doc["predicted_ner_proba"]):
            for start, end, prob, _gold_type in sent_preds:
                if prob >= threshold:
                    spans.append((doc_id, sent_idx, start, end))
    return spans


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _compute_counts(
    gold_spans: List[Tuple],
    predicted_spans: List[Tuple],
    all_candidate_spans: Optional[List[Tuple]] = None,
) -> Dict:
    """Span-level TP/FP/FN counts (no entity type involved).

    Parameters
    ----------
    all_candidate_spans : if provided, used to compute n_gold_in_pool — the number
        of gold spans that appear anywhere in the pruner's candidate pool, regardless
        of whether they pass the threshold.  This is needed to compute a meaningful TN:
            TN = n_total_candidates - n_pred - (n_gold_in_pool - TP)
        Without this, TN can be negative when some gold spans fall outside the pool
        (e.g. they exceed the pruner's maximum span length) and threshold=0 selects
        every candidate.
    """
    gold_set = set(gold_spans)
    pred_set = set(predicted_spans)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    counts = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gold": len(gold_set),
        "n_pred": len(pred_set),
    }
    if all_candidate_spans is not None:
        counts["n_gold_in_pool"] = len(gold_set & set(all_candidate_spans))
    return counts


def _metrics_from_counts(counts: Dict) -> Dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    n_pred = counts["n_pred"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # Share of predicted candidates that are false positives (FP / n_predicted).
    false_positive_share = fp / n_pred if n_pred > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_share": false_positive_share,
        **counts,
    }


def compute_metrics(
    gold_spans: List[Tuple],
    predicted_spans: List[Tuple],
    all_candidate_spans: Optional[List[Tuple]] = None,
) -> Dict:
    """Compute span-level precision, recall, F1 and false_positive_share.

    Pass *all_candidate_spans* (the full pruner pool at threshold=0) to also
    compute n_gold_in_pool, which enables a correct TN calculation.
    """
    return _metrics_from_counts(
        _compute_counts(gold_spans, predicted_spans, all_candidate_spans)
    )


def compute_metrics_per_type(
    gold_entities: List[Tuple],  # (doc_id, sent_idx, start, end, type)
    predicted_spans: List[Tuple],  # (doc_id, sent_idx, start, end)  — no type
    entity_types: List[str],
) -> Dict[str, Dict]:
    """Compute recall per gold entity type.

    Since the pruner is a binary classifier, FP spans carry no entity type.
    Precision, F1 and false_positive_share are therefore NOT included here —
    they are only meaningful at the overall (span-level) scope.
    Use compute_metrics() for those.
    """
    pred_set = set(predicted_spans)

    results = {}
    for etype in entity_types:
        gold_type = [(e[0], e[1], e[2], e[3]) for e in gold_entities if e[4] == etype]
        gold_type_set = set(gold_type)
        tp = len(gold_type_set & pred_set)
        fn = len(gold_type_set - pred_set)
        n_gold = len(gold_type_set)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results[etype] = {
            "recall": recall,
            "tp": tp,
            "fn": fn,
            "n_gold": n_gold,
        }
    return results


# ---------------------------------------------------------------------------
# Threshold-level API
# ---------------------------------------------------------------------------


def compute_metrics_at_threshold(
    docs: List[dict],
    threshold: float,
    entity_types: List[str] = None,
) -> Dict:
    """Compute all metrics at a single threshold."""
    gold_spans = _gold_spans(docs)
    all_candidates = _all_candidate_spans(docs)
    predicted = _predicted_spans(docs, threshold)
    metrics = compute_metrics(gold_spans, predicted, all_candidate_spans=all_candidates)
    metrics["threshold"] = threshold
    if entity_types:
        gold_entities = _gold_entities(docs)
        metrics["per_type"] = compute_metrics_per_type(
            gold_entities, predicted, entity_types
        )
    return metrics


def threshold_sweep(
    docs: List[dict],
    thresholds: List[float],
    entity_types: List[str] = None,
) -> List[Dict]:
    """Compute metrics for each value in *thresholds*.

    Gold spans are extracted once; only predictions change per threshold.
    """
    gold_spans = _gold_spans(docs)
    gold_entities = _gold_entities(docs) if entity_types else None
    results = []
    for t in thresholds:
        predicted = _predicted_spans(docs, t)
        m = compute_metrics(gold_spans, predicted)
        m["threshold"] = t
        if entity_types:
            m["per_type"] = compute_metrics_per_type(
                gold_entities, predicted, entity_types
            )
        results.append(m)
    return results


def find_threshold_for_recall(
    docs: List[dict],
    target_recall: float = 0.98,
    tol: float = 1e-5,
    max_iter: int = 60,
) -> Tuple[float, Dict]:
    """Binary-search for the *highest* threshold that still achieves *target_recall*.

    Starting point: 0.5, then bisecting toward the optimum.

    Returns
    -------
    best_threshold : float
    metrics        : dict
    """
    gold_spans = _gold_spans(docs)

    def recall_at(t: float) -> float:
        predicted = _predicted_spans(docs, t)
        counts = _compute_counts(gold_spans, predicted)
        tp, fn = counts["tp"], counts["fn"]
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    lo, hi = 0.0, 1.0
    mid = 0.5
    best_threshold = lo  # fallback: include everything

    for _ in range(max_iter):
        r = recall_at(mid)
        if r >= target_recall:
            best_threshold = mid
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

        mid = (lo + hi) / 2.0

    predicted = _predicted_spans(docs, best_threshold)
    metrics = compute_metrics(gold_spans, predicted)
    metrics["threshold"] = best_threshold
    return best_threshold, metrics


# ---------------------------------------------------------------------------
# Top-K selection (original paper approach)
# ---------------------------------------------------------------------------


def _sent_length(sent_preds: List) -> int:
    """Infer sentence length from the maximum token index in predicted spans + 1."""
    if not sent_preds:
        return 0
    return max(end for _, end, *_ in sent_preds) + 1


def _predicted_spans_topk(
    docs: List[dict],
    topk_ratio: float,
    min_mentions_num: int,
    max_mentions_num: int,
) -> List[Tuple]:
    """Return predicted spans using the paper's top-K selection strategy.

    Per sentence, K = max(lmin, min(lambda * n, lmax)) spans are kept,
    selected by ranking candidate spans by their pruner score (descending).

    Parameters
    ----------
    topk_ratio        : lambda — ratio of candidates to sentence length
    min_mentions_num  : l_min  — floor on K
    max_mentions_num  : l_max  — ceiling on K
    """
    spans = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_preds in enumerate(doc["predicted_ner_proba"]):
            if not sent_preds:
                continue
            n = _sent_length(sent_preds)
            k = max(min_mentions_num, min(int(topk_ratio * n), max_mentions_num))
            # Rank by prob descending and take top-k
            ranked = sorted(sent_preds, key=lambda x: x[2], reverse=True)
            for start, end, _prob, _gold_type in ranked[:k]:
                spans.append((doc_id, sent_idx, start, end))
    return spans


def compute_metrics_topk(
    docs: List[dict],
    topk_ratio: float,
    min_mentions_num: int,
    max_mentions_num: int,
    entity_types: Optional[List[str]] = None,
) -> Dict:
    """Compute metrics using the top-K selection strategy."""
    gold_spans = _gold_spans(docs)
    all_candidates = _all_candidate_spans(docs)
    predicted = _predicted_spans_topk(
        docs, topk_ratio, min_mentions_num, max_mentions_num
    )
    metrics = compute_metrics(gold_spans, predicted, all_candidate_spans=all_candidates)
    metrics.update(
        {
            "topk_ratio": topk_ratio,
            "min_mentions_num": min_mentions_num,
            "max_mentions_num": max_mentions_num,
        }
    )
    if entity_types:
        gold_entities = _gold_entities(docs)
        metrics["per_type"] = compute_metrics_per_type(
            gold_entities, predicted, entity_types
        )
    return metrics


def optimize_topk_params(
    docs: List[dict],
    target_recall: float = 0.99,
    topk_ratio_values: Optional[List[float]] = None,
    min_mentions_num_values: Optional[List[int]] = None,
    max_mentions_num_values: Optional[List[int]] = None,
    entity_types: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], List[Dict]]:
    """Grid-search for top-K parameters that minimise FP share at target recall.

    Searches every combination of the three parameter lists and returns
    the best configuration (highest threshold = fewest candidates) that
    still achieves *target_recall*, as well as all evaluated results.

    Parameters
    ----------
    docs                    : loaded prediction documents
    target_recall           : minimum acceptable recall (default 0.99)
    topk_ratio_values       : lambda values to try  (default: 0.1 … 1.0 in steps of 0.1)
    min_mentions_num_values : l_min values to try   (default: 1, 2, 3, 4, 5)
    max_mentions_num_values : l_max values to try   (default: 10, 15, 20, 25, 30)
    entity_types            : if given, compute per-type recall in each result

    Returns
    -------
    best   : dict with params + metrics for the best configuration, or None if no
             configuration achieves target_recall
    all_results : list of dicts with params + metrics for every evaluated combination,
                  sorted by false_positive_share ascending (feasible first)
    """
    if topk_ratio_values is None:
        topk_ratio_values = [round(v * 0.1, 1) for v in range(1, 11)]  # 0.1 … 1.0
    if min_mentions_num_values is None:
        min_mentions_num_values = [1, 2, 3, 4, 5]
    if max_mentions_num_values is None:
        max_mentions_num_values = [10, 15, 20, 25, 30]

    all_results = []
    for lam, lmin, lmax in product(
        topk_ratio_values, min_mentions_num_values, max_mentions_num_values
    ):
        if lmin > lmax:
            continue
        m = compute_metrics_topk(docs, lam, lmin, lmax, entity_types=entity_types)
        all_results.append(m)

    # Sort: feasible configs (recall >= target) first by fp_share, then infeasible by recall desc
    feasible = [r for r in all_results if r["recall"] >= target_recall]
    infeasible = [r for r in all_results if r["recall"] < target_recall]

    feasible.sort(key=lambda r: r["false_positive_share"])
    infeasible.sort(key=lambda r: r["recall"], reverse=True)

    best = feasible[0] if feasible else None
    return best, feasible + infeasible
