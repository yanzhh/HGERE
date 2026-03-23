"""HGERE evaluation utilities for NER and relation extraction.

Prediction file format (jsonl)
-------------------------------
Each line is a document dict with:

  - 'ner':                  list[list[[start, end, type]]]
        Gold entities per sentence.
  - 'relations':            list[list[[s_start, s_end, o_start, o_end, label]]]
        Gold relations per sentence.
  - 'predicted_ner':        list[list[[start, end, type]]]
        Predicted entities per sentence.
  - 'predicted_rel':        list[list[[s_start, s_end, o_start, o_end, label]]]
        Predicted relations per sentence.
  - 'predicted_ner_proba':  list[list[[start, end, type, pruner_score, hgere_score]]]
        Predicted entities with confidence scores (optional).
  - 'predicted_rel_proba':  list[list[[s_start, s_end, o_start, o_end, label, score]]]
        Predicted relations with confidence scores (optional).

Metrics
-------
- ``re``  : relation matching on (subj_span, obj_span, label) — entity types ignored.
- ``re+`` : relation matching on (subj_span+subj_type, obj_span+obj_type, label) — entity
            types from *predicted* NER must match those in the gold annotation.

This mirrors the evaluation logic in ``src/gsapere/hgere/evaluate.py``.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_predictions(filepath: str) -> List[dict]:
    """Load predictions from a JSONL file."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Gold extraction
# ---------------------------------------------------------------------------


def _gold_entities(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, start, end, type) for every gold entity."""
    entities = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_ner in enumerate(doc.get("ner", [])):
            for start, end, etype in sent_ner:
                entities.append((doc_id, sent_idx, start, end, etype))
    return entities


def _gold_relations(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, s_start, s_end, o_start, o_end, label) for every gold relation."""
    relations = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_rels in enumerate(doc.get("relations", [])):
            for s_start, s_end, o_start, o_end, label in sent_rels:
                relations.append(
                    (doc_id, sent_idx, s_start, s_end, o_start, o_end, label)
                )
    return relations


def _gold_relations_with_ner(docs: List[dict]) -> List[Tuple]:
    """Return gold relations augmented with gold entity types.

    Returns (doc_id, sent_idx, s_start, s_end, s_type, o_start, o_end, o_type, label).
    """
    ner_lookup: Dict[Tuple, str] = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_ner in enumerate(doc.get("ner", [])):
            for start, end, etype in sent_ner:
                ner_lookup[(doc_id, sent_idx, start, end)] = etype

    relations = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_rels in enumerate(doc.get("relations", [])):
            for s_start, s_end, o_start, o_end, label in sent_rels:
                s_type = ner_lookup.get((doc_id, sent_idx, s_start, s_end), "")
                o_type = ner_lookup.get((doc_id, sent_idx, o_start, o_end), "")
                relations.append(
                    (
                        doc_id,
                        sent_idx,
                        s_start,
                        s_end,
                        s_type,
                        o_start,
                        o_end,
                        o_type,
                        label,
                    )
                )
    return relations


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------


def _predicted_entities(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, start, end, type) for every predicted entity."""
    entities = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_ner in enumerate(doc.get("predicted_ner", [])):
            for start, end, etype in sent_ner:
                entities.append((doc_id, sent_idx, start, end, etype))
    return entities


def _predicted_relations(docs: List[dict]) -> List[Tuple]:
    """Return (doc_id, sent_idx, s_start, s_end, o_start, o_end, label) for every predicted relation."""
    relations = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_rels in enumerate(doc.get("predicted_rel", [])):
            for s_start, s_end, o_start, o_end, label in sent_rels:
                relations.append(
                    (doc_id, sent_idx, s_start, s_end, o_start, o_end, label)
                )
    return relations


def _predicted_relations_with_ner(docs: List[dict]) -> List[Tuple]:
    """Return predicted relations augmented with predicted entity types.

    Returns (doc_id, sent_idx, s_start, s_end, s_type, o_start, o_end, o_type, label).
    """
    ner_lookup: Dict[Tuple, str] = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_ner in enumerate(doc.get("predicted_ner", [])):
            for start, end, etype in sent_ner:
                ner_lookup[(doc_id, sent_idx, start, end)] = etype

    relations = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_rels in enumerate(doc.get("predicted_rel", [])):
            for s_start, s_end, o_start, o_end, label in sent_rels:
                s_type = ner_lookup.get((doc_id, sent_idx, s_start, s_end), "")
                o_type = ner_lookup.get((doc_id, sent_idx, o_start, o_end), "")
                relations.append(
                    (
                        doc_id,
                        sent_idx,
                        s_start,
                        s_end,
                        s_type,
                        o_start,
                        o_end,
                        o_type,
                        label,
                    )
                )
    return relations


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _prf1(tp: int, n_gold: int, n_pred: int) -> Tuple[float, float, float]:
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gold if n_gold > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _macro_weighted(
    per_type: Dict[str, Dict], weighted: bool
) -> Tuple[float, float, float]:
    """Compute macro or weighted-macro P/R/F1 from per-type metrics dicts.

    per_type values must have keys 'precision', 'recall', 'f1', 'n_gold'.
    Types with n_gold == 0 are excluded (undefined precision/recall).
    """
    types = [m for m in per_type.values() if m["n_gold"] > 0]
    if not types:
        return 0.0, 0.0, 0.0
    if weighted:
        total = sum(m["n_gold"] for m in types)
        weights = [m["n_gold"] / total for m in types]
    else:
        w = 1.0 / len(types)
        weights = [w] * len(types)
    p = sum(wt * m["precision"] for wt, m in zip(weights, types))
    r = sum(wt * m["recall"] for wt, m in zip(weights, types))
    f = sum(wt * m["f1"] for wt, m in zip(weights, types))
    return p, r, f


# ---------------------------------------------------------------------------
# NER metrics
# ---------------------------------------------------------------------------


def compute_ner_metrics(docs: List[dict]) -> Dict:
    """Compute overall NER metrics: micro, macro, and weighted-macro P/R/F1."""
    gold = set(_gold_entities(docs))
    pred = set(_predicted_entities(docs))
    tp = len(gold & pred)
    micro_p, micro_r, micro_f1 = _prf1(tp, len(gold), len(pred))

    per_type = compute_ner_metrics_per_type(docs)
    macro_p, macro_r, macro_f1 = _macro_weighted(per_type, weighted=False)
    wmacro_p, wmacro_r, wmacro_f1 = _macro_weighted(per_type, weighted=True)

    return {
        "ner_precision": micro_p,
        "ner_recall": micro_r,
        "ner_f1": micro_f1,
        "ner_macro_precision": macro_p,
        "ner_macro_recall": macro_r,
        "ner_macro_f1": macro_f1,
        "ner_wmacro_precision": wmacro_p,
        "ner_wmacro_recall": wmacro_r,
        "ner_wmacro_f1": wmacro_f1,
        "ner_tp": tp,
        "ner_fp": len(pred - gold),
        "ner_fn": len(gold - pred),
        "ner_n_gold": len(gold),
        "ner_n_pred": len(pred),
    }


def compute_ner_metrics_per_type(docs: List[dict]) -> Dict[str, Dict]:
    """Compute NER precision, recall, and F1 per entity type."""
    gold_entities = _gold_entities(docs)
    pred_entities = _predicted_entities(docs)
    all_types = sorted(set(e[4] for e in gold_entities))

    results: Dict[str, Dict] = {}
    for etype in all_types:
        gold_spans = {(d, s, st, en) for d, s, st, en, t in gold_entities if t == etype}
        pred_spans = {(d, s, st, en) for d, s, st, en, t in pred_entities if t == etype}
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        p, r, f1 = _prf1(tp, len(gold_spans), len(pred_spans))
        results[etype] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gold": len(gold_spans),
            "n_pred": len(pred_spans),
        }
    return results


# ---------------------------------------------------------------------------
# Relation metrics
# ---------------------------------------------------------------------------


def compute_rel_metrics(docs: List[dict]) -> Dict:
    """Compute relation P/R/F1 without entity-type matching (re metric).

    Returns micro, macro, and weighted-macro averages.
    """
    gold = set(_gold_relations(docs))
    pred = set(_predicted_relations(docs))
    tp = len(gold & pred)
    micro_p, micro_r, micro_f1 = _prf1(tp, len(gold), len(pred))

    per_type = compute_rel_metrics_per_type(docs)
    macro_p, macro_r, macro_f1 = _macro_weighted(per_type, weighted=False)
    wmacro_p, wmacro_r, wmacro_f1 = _macro_weighted(per_type, weighted=True)

    return {
        "re_precision": micro_p,
        "re_recall": micro_r,
        "re_f1": micro_f1,
        "re_macro_precision": macro_p,
        "re_macro_recall": macro_r,
        "re_macro_f1": macro_f1,
        "re_wmacro_precision": wmacro_p,
        "re_wmacro_recall": wmacro_r,
        "re_wmacro_f1": wmacro_f1,
        "re_tp": tp,
        "re_fp": len(pred - gold),
        "re_fn": len(gold - pred),
        "re_n_gold": len(gold),
        "re_n_pred": len(pred),
    }


def compute_rel_metrics_with_ner(docs: List[dict]) -> Dict:
    """Compute relation P/R/F1 where entity types must also match (re+ metric).

    Returns micro, macro, and weighted-macro averages.
    """
    gold_list = _gold_relations_with_ner(docs)
    pred_list = _predicted_relations_with_ner(docs)
    gold = set(gold_list)
    pred = set(pred_list)
    tp = len(gold & pred)
    micro_p, micro_r, micro_f1 = _prf1(tp, len(gold), len(pred))

    # Per-type (label is last element)
    all_types = sorted(set(r[-1] for r in gold_list))
    per_type: Dict[str, Dict] = {}
    for rtype in all_types:
        g = {r for r in gold_list if r[-1] == rtype}
        p_set = {r for r in pred_list if r[-1] == rtype}
        tp_t = len(g & p_set)
        pp, rr, ff = _prf1(tp_t, len(g), len(p_set))
        per_type[rtype] = {"precision": pp, "recall": rr, "f1": ff, "n_gold": len(g)}

    macro_p, macro_r, macro_f1 = _macro_weighted(per_type, weighted=False)
    wmacro_p, wmacro_r, wmacro_f1 = _macro_weighted(per_type, weighted=True)

    return {
        "re+_precision": micro_p,
        "re+_recall": micro_r,
        "re+_f1": micro_f1,
        "re+_macro_precision": macro_p,
        "re+_macro_recall": macro_r,
        "re+_macro_f1": macro_f1,
        "re+_wmacro_precision": wmacro_p,
        "re+_wmacro_recall": wmacro_r,
        "re+_wmacro_f1": wmacro_f1,
        "re+_tp": tp,
        "re+_fp": len(pred - gold),
        "re+_fn": len(gold - pred),
        "re+_n_gold": len(gold),
        "re+_n_pred": len(pred),
    }


def compute_rel_metrics_per_type(docs: List[dict]) -> Dict[str, Dict]:
    """Compute relation P/R/F1 per relation type."""
    gold_relations = _gold_relations(docs)
    pred_relations = _predicted_relations(docs)
    all_types = sorted(set(r[-1] for r in gold_relations))

    results: Dict[str, Dict] = {}
    for rtype in all_types:
        gold_type = {r for r in gold_relations if r[-1] == rtype}
        pred_type = {r for r in pred_relations if r[-1] == rtype}
        tp = len(gold_type & pred_type)
        fp = len(pred_type - gold_type)
        fn = len(gold_type - pred_type)
        p, r, f1 = _prf1(tp, len(gold_type), len(pred_type))
        results[rtype] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gold": len(gold_type),
            "n_pred": len(pred_type),
        }
    return results


# ---------------------------------------------------------------------------
# Pruner candidate metrics
# ---------------------------------------------------------------------------


def compute_pruner_metrics(docs: List[dict], threshold: float) -> Dict:
    """Evaluate pruner candidate recall/precision at a given threshold.

    Candidates are read from ``ner_candidates_proba`` (entries
    ``[start, end, prob, gold_label]``).  Gold is ``ner`` (span-level only,
    entity type is ignored).

    Args:
        docs: Prediction documents containing ``ner`` and
            ``ner_candidates_proba`` fields.
        threshold: Keep candidate spans with ``prob >= threshold``.

    Returns:
        Dict with keys ``pruner_precision``, ``pruner_recall``, ``pruner_f1``,
        ``pruner_n_gold``, ``pruner_n_pred``, ``pruner_tp``, ``pruner_fp``,
        ``pruner_fn``.
    """
    gold_spans: set = set()
    pred_spans: set = set()

    for doc in docs:
        doc_id = doc["doc_id"]
        for sent_idx, sent_gold in enumerate(doc.get("ner", [])):
            for start, end, _type in sent_gold:
                gold_spans.add((doc_id, sent_idx, start, end))

        for sent_idx, sent_cands in enumerate(doc.get("ner_candidates_proba", [])):
            for entry in sent_cands:
                start, end, prob = entry[0], entry[1], entry[2]
                if prob >= threshold:
                    pred_spans.add((doc_id, sent_idx, start, end))

    tp = len(gold_spans & pred_spans)
    p, r, f1 = _prf1(tp, len(gold_spans), len(pred_spans))
    return {
        "pruner_precision": p,
        "pruner_recall": r,
        "pruner_f1": f1,
        "pruner_tp": tp,
        "pruner_fp": len(pred_spans - gold_spans),
        "pruner_fn": len(gold_spans - pred_spans),
        "pruner_n_gold": len(gold_spans),
        "pruner_n_pred": len(pred_spans),
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate(docs: List[dict]) -> Dict:
    """Run full NER + RE evaluation and return all metrics in a flat dict."""
    metrics: Dict = {}
    metrics.update(compute_ner_metrics(docs))
    metrics.update(compute_rel_metrics(docs))
    metrics.update(compute_rel_metrics_with_ner(docs))
    return metrics
