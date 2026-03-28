"""Tests for evaluation/pruner.py.

The pruner is a binary entity/non-entity classifier.
- `predicted_ner_proba` stores [start, end, prob, gold_type] — the label is GOLD, not predicted.
- Threshold is applied to prob only; predicted spans have no type.
- Matching is span-level: (doc_id, sent_idx, start, end).
"""
import json
from pathlib import Path

import pytest

from gsapere.evaluation.pruner import (
    _compute_counts,
    _gold_entities,
    _gold_spans,
    _metrics_from_counts,
    _predicted_spans,
    _predicted_spans_topk,
    _sent_length,
    compute_metrics,
    compute_metrics_at_threshold,
    compute_metrics_per_type,
    compute_metrics_topk,
    find_threshold_for_recall,
    load_predictions,
    optimize_topk_params,
    threshold_sweep,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(doc_id, gold, predicted):
    """Build a minimal prediction document.

    gold      : list of [start, end, type] per sentence
    predicted : list of [start, end, prob, gold_type] per sentence
    """
    return {"doc_id": doc_id, "ner": gold, "predicted_ner_proba": predicted}


# DOC_A — two sentences.
# Sentence 0 gold: Method@[0,1], Dataset@[5,6]
# Sentence 1 gold: Task@[2,3]
# Predicted (with gold labels for reference):
#   prob >= 0.5 → spans [0,1], [5,6], [3,4](NIL→FP), [2,3], [9,9](NIL→FP)
#   prob < 0.5  → spans [7,8](NIL)
DOC_A = _make_doc(
    doc_id="A",
    gold=[
        [[0, 1, "Method"], [5, 6, "Dataset"]],  # sentence 0
        [[2, 3, "Task"]],                        # sentence 1
    ],
    predicted=[
        [
            [0, 1, 0.95, "Method"],   # TP  high confidence
            [5, 6, 0.60, "Dataset"],  # TP  medium confidence
            [3, 4, 0.80, "NIL"],      # FP  gold is NIL, but prob > threshold
            [7, 8, 0.20, "NIL"],      # below threshold → excluded
        ],
        [
            [2, 3, 0.40, "Task"],     # TP  (below 0.5 → excluded at that threshold)
            [9, 9, 0.70, "NIL"],      # FP  gold is NIL, prob > 0.5
        ],
    ],
)

# DOC_B — gold entity always predicted as NIL (FN), plus one FP
DOC_B = _make_doc(
    doc_id="B",
    gold=[[[1, 2, "Method"]]],
    predicted=[[
        [1, 2, 0.30, "Method"],  # FN at threshold 0.5 (prob too low)
        [3, 4, 0.90, "NIL"],     # FP (gold NIL but high prob)
    ]],
)

DOCS = [DOC_A, DOC_B]


# ---------------------------------------------------------------------------
# _gold_spans / _gold_entities
# ---------------------------------------------------------------------------

class TestGoldSpans:
    def test_extracts_all_gold_spans(self):
        spans = _gold_spans([DOC_A])
        assert len(spans) == 3

    def test_no_type_in_span(self):
        spans = _gold_spans([DOC_A])
        assert all(len(s) == 4 for s in spans)

    def test_correct_tuples(self):
        spans = _gold_spans([DOC_A])
        assert ("A", 0, 0, 1) in spans
        assert ("A", 0, 5, 6) in spans
        assert ("A", 1, 2, 3) in spans

    def test_multi_doc(self):
        spans = _gold_spans(DOCS)
        assert len(spans) == 4

    def test_empty_sentence(self):
        doc = _make_doc("E", [[]], [[]])
        assert _gold_spans([doc]) == []


class TestGoldEntities:
    def test_includes_type(self):
        entities = _gold_entities([DOC_A])
        assert ("A", 0, 0, 1, "Method") in entities

    def test_count(self):
        assert len(_gold_entities([DOC_A])) == 3


# ---------------------------------------------------------------------------
# _predicted_spans
# ---------------------------------------------------------------------------

class TestPredictedSpans:
    def test_threshold_filters_low_confidence(self):
        # At 0.5: [2,3] (prob=0.40) excluded, [7,8] (prob=0.20) excluded
        spans = _predicted_spans([DOC_A], threshold=0.5)
        assert ("A", 1, 2, 3) not in spans
        assert ("A", 0, 7, 8) not in spans

    def test_nil_gold_included_if_above_threshold(self):
        # Gold NIL spans with high prob MUST be included — pruner doesn't use gold labels
        spans = _predicted_spans([DOC_A], threshold=0.5)
        assert ("A", 0, 3, 4) in spans   # prob=0.80, gold=NIL → FP candidate
        assert ("A", 1, 9, 9) in spans   # prob=0.70, gold=NIL → FP candidate

    def test_threshold_zero_includes_all(self):
        # All 6 entries in DOC_A pass threshold 0
        spans = _predicted_spans([DOC_A], threshold=0.0)
        assert len(spans) == 6

    def test_threshold_one_excludes_all(self):
        assert _predicted_spans([DOC_A], threshold=1.0) == []

    def test_exact_boundary_included(self):
        spans = _predicted_spans([DOC_A], threshold=0.60)
        assert ("A", 0, 5, 6) in spans   # prob == 0.60

    def test_no_type_in_returned_tuples(self):
        spans = _predicted_spans([DOC_A], threshold=0.0)
        assert all(len(s) == 4 for s in spans)


# ---------------------------------------------------------------------------
# _compute_counts / _metrics_from_counts / compute_metrics
# ---------------------------------------------------------------------------

class TestComputeCounts:
    def test_perfect_match(self):
        gold = [("A", 0, 0, 1)]
        pred = [("A", 0, 0, 1)]
        c = _compute_counts(gold, pred)
        assert c == {"tp": 1, "fp": 0, "fn": 0, "n_gold": 1, "n_pred": 1}

    def test_false_positive(self):
        gold = []
        pred = [("A", 0, 0, 1)]
        c = _compute_counts(gold, pred)
        assert c["tp"] == 0 and c["fp"] == 1 and c["fn"] == 0

    def test_false_negative(self):
        gold = [("A", 0, 0, 1)]
        pred = []
        c = _compute_counts(gold, pred)
        assert c["tp"] == 0 and c["fp"] == 0 and c["fn"] == 1

    def test_duplicates_counted_once(self):
        gold = [("A", 0, 0, 1), ("A", 0, 0, 1)]
        pred = [("A", 0, 0, 1)]
        c = _compute_counts(gold, pred)
        assert c["n_gold"] == 1 and c["tp"] == 1


class TestMetricsFromCounts:
    def test_values(self):
        counts = {"tp": 2, "fp": 1, "fn": 1, "n_gold": 3, "n_pred": 3}
        m = _metrics_from_counts(counts)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"]    == pytest.approx(2 / 3)
        assert m["f1"]        == pytest.approx(2 / 3)

    def test_false_positive_share(self):
        counts = {"tp": 2, "fp": 2, "fn": 0, "n_gold": 2, "n_pred": 4}
        m = _metrics_from_counts(counts)
        assert m["false_positive_share"] == pytest.approx(0.5)

    def test_no_predictions(self):
        counts = {"tp": 0, "fp": 0, "fn": 3, "n_gold": 3, "n_pred": 0}
        m = _metrics_from_counts(counts)
        assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0

    def test_perfect(self):
        counts = {"tp": 5, "fp": 0, "fn": 0, "n_gold": 5, "n_pred": 5}
        m = _metrics_from_counts(counts)
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
        assert m["false_positive_share"] == 0.0


class TestComputeMetrics:
    def test_fp_nonzero_at_threshold_0_5(self):
        # At 0.5: TP=[0,1],[5,6] FP=[3,4],[9,9] FN=[2,3]
        gold   = _gold_spans([DOC_A])
        pred   = _predicted_spans([DOC_A], 0.5)
        m = compute_metrics(gold, pred)
        assert m["tp"] == 2
        assert m["fp"] == 2
        assert m["fn"] == 1

    def test_recall_at_threshold_0(self):
        gold = _gold_spans([DOC_A])
        pred = _predicted_spans([DOC_A], 0.0)
        m = compute_metrics(gold, pred)
        assert m["recall"] == pytest.approx(1.0)   # all gold spans covered


# ---------------------------------------------------------------------------
# compute_metrics_per_type
# ---------------------------------------------------------------------------

class TestComputeMetricsPerType:
    def setup_method(self):
        self.gold_ents = _gold_entities([DOC_A])
        self.pred      = _predicted_spans([DOC_A], 0.5)
        # At 0.5: pred = [0,1],[5,6],[3,4],[9,9]  (4 spans)
        # gold spans: [0,1],[5,6],[2,3]
        # TP=2, FP=2, FN=1

    def test_known_types_present(self):
        r = compute_metrics_per_type(self.gold_ents, self.pred, ["Method", "Dataset", "Task"])
        assert set(r.keys()) == {"Method", "Dataset", "Task"}

    def test_method_recall(self):
        r = compute_metrics_per_type(self.gold_ents, self.pred, ["Method"])["Method"]
        # gold Method: [0,1] → predicted (prob=0.95 >= 0.5) → TP=1, FN=0
        assert r["tp"] == 1 and r["fn"] == 0
        assert r["recall"] == pytest.approx(1.0)

    def test_task_recall(self):
        r = compute_metrics_per_type(self.gold_ents, self.pred, ["Task"])["Task"]
        # gold Task: [2,3] → prob=0.40 < 0.5 → not predicted → FN=1
        assert r["tp"] == 0 and r["fn"] == 1
        assert r["recall"] == pytest.approx(0.0)

    def test_no_precision_or_f1_in_per_type(self):
        # Precision/F1/fp_share are not meaningful per type for a binary pruner
        r = compute_metrics_per_type(self.gold_ents, self.pred, ["Method"])["Method"]
        assert "precision" not in r
        assert "f1" not in r
        assert "false_positive_share" not in r

    def test_absent_type_zero_recall(self):
        r = compute_metrics_per_type(self.gold_ents, self.pred, ["NonExistent"])["NonExistent"]
        assert r["tp"] == 0 and r["fn"] == 0 and r["recall"] == 0.0


# ---------------------------------------------------------------------------
# compute_metrics_at_threshold
# ---------------------------------------------------------------------------

class TestComputeMetricsAtThreshold:
    def test_threshold_stored(self):
        m = compute_metrics_at_threshold([DOC_A], 0.5)
        assert m["threshold"] == 0.5

    def test_fp_nonzero(self):
        m = compute_metrics_at_threshold([DOC_A], 0.5)
        assert m["fp"] > 0

    def test_per_type_included(self):
        m = compute_metrics_at_threshold([DOC_A], 0.5, entity_types=["Method"])
        assert "per_type" in m and "Method" in m["per_type"]

    def test_per_type_absent_without_arg(self):
        m = compute_metrics_at_threshold([DOC_A], 0.5)
        assert "per_type" not in m

    def test_higher_threshold_lowers_recall(self):
        m_low  = compute_metrics_at_threshold([DOC_A], 0.0)
        m_high = compute_metrics_at_threshold([DOC_A], 0.9)
        assert m_high["recall"] <= m_low["recall"]


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    def test_one_result_per_threshold(self):
        results = threshold_sweep([DOC_A], [0.0, 0.5, 1.0])
        assert len(results) == 3

    def test_recall_non_increasing(self):
        thresholds = [i / 10 for i in range(11)]
        results = threshold_sweep([DOC_A], thresholds)
        recalls = [r["recall"] for r in results]
        assert all(recalls[i] >= recalls[i + 1] for i in range(len(recalls) - 1))

    def test_threshold_key_correct(self):
        results = threshold_sweep([DOC_A], [0.3, 0.7])
        assert results[0]["threshold"] == pytest.approx(0.3)
        assert results[1]["threshold"] == pytest.approx(0.7)

    def test_fp_nonzero_at_mid_threshold(self):
        results = threshold_sweep([DOC_A], [0.5])
        assert results[0]["fp"] > 0


# ---------------------------------------------------------------------------
# find_threshold_for_recall
# ---------------------------------------------------------------------------

class TestFindThresholdForRecall:
    def test_achieves_target(self):
        _, metrics = find_threshold_for_recall([DOC_A], target_recall=0.98)
        assert metrics["recall"] >= 0.98

    def test_threshold_in_unit_interval(self):
        threshold, _ = find_threshold_for_recall([DOC_A], target_recall=0.5)
        assert 0.0 <= threshold <= 1.0

    def test_impossible_recall_falls_back(self):
        # Gold span [9,9] is not present in predicted_ner_proba at all →
        # recall can never reach 1.0 regardless of threshold
        doc = _make_doc("X",
            gold=[[[9, 9, "Method"]]],
            predicted=[[
                [1, 2, 0.99, "NIL"],   # high-prob FP, but not the gold span
            ]],
        )
        threshold, metrics = find_threshold_for_recall([doc], target_recall=1.0)
        assert threshold == pytest.approx(0.0, abs=1e-4)
        assert metrics["recall"] < 1.0

    def test_metrics_has_threshold_key(self):
        _, metrics = find_threshold_for_recall([DOC_A], target_recall=0.9)
        assert "threshold" in metrics


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------

class TestLoadPredictions:
    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "preds.json"
        p.write_text("\n".join(json.dumps(d) for d in [DOC_A, DOC_B]))
        loaded = load_predictions(str(p))
        assert len(loaded) == 2 and loaded[0]["doc_id"] == "A"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_predictions("/nonexistent/file.json")


# ---------------------------------------------------------------------------
# Top-K selection helpers
# ---------------------------------------------------------------------------

# Minimal doc for top-K tests: one sentence, 10 tokens
# Spans sorted by prob: [0,1]=0.95, [3,4]=0.80, [5,6]=0.60, [2,3]=0.40, [7,8]=0.20
DOC_TOPK = {
    "doc_id": "T",
    "ner": [[[0, 1, "Method"], [5, 6, "Dataset"]]],
    "predicted_ner_proba": [[
        [0, 1, 0.95, "Method"],
        [3, 4, 0.80, "NIL"],
        [5, 6, 0.60, "Dataset"],
        [2, 3, 0.40, "NIL"],
        [7, 8, 0.20, "NIL"],
    ]],
}


class TestSentLength:
    def test_basic(self):
        # max end = 8, so length = 9
        assert _sent_length(DOC_TOPK["predicted_ner_proba"][0]) == 9

    def test_empty(self):
        assert _sent_length([]) == 0


class TestPredictedSpansTopk:
    def test_takes_top_k_by_prob(self):
        # K = max(3, min(0.5*9, 18)) = max(3, 4) = 4 → top-4 spans
        spans = _predicted_spans_topk([DOC_TOPK], topk_ratio=0.5, min_mentions_num=3, max_mentions_num=18)
        assert len(spans) == 4
        assert ("T", 0, 0, 1) in spans   # prob=0.95
        assert ("T", 0, 3, 4) in spans   # prob=0.80
        assert ("T", 0, 5, 6) in spans   # prob=0.60
        assert ("T", 0, 2, 3) in spans   # prob=0.40
        assert ("T", 0, 7, 8) not in spans  # prob=0.20 (rank 5, excluded)

    def test_lmin_floor(self):
        # K = max(5, min(0.1*9, 18)) = max(5, 0) = 5 → all 5 spans
        spans = _predicted_spans_topk([DOC_TOPK], topk_ratio=0.1, min_mentions_num=5, max_mentions_num=18)
        assert len(spans) == 5

    def test_lmax_ceiling(self):
        # K = max(1, min(10*9, 2)) = max(1, 2) = 2 → top-2 spans only
        spans = _predicted_spans_topk([DOC_TOPK], topk_ratio=10.0, min_mentions_num=1, max_mentions_num=2)
        assert len(spans) == 2
        assert ("T", 0, 0, 1) in spans   # prob=0.95 (rank 1)
        assert ("T", 0, 3, 4) in spans   # prob=0.80 (rank 2)

    def test_no_type_in_tuples(self):
        spans = _predicted_spans_topk([DOC_TOPK], topk_ratio=0.5, min_mentions_num=1, max_mentions_num=18)
        assert all(len(s) == 4 for s in spans)

    def test_empty_sentence_skipped(self):
        doc = {"doc_id": "E", "ner": [[]], "predicted_ner_proba": [[]]}
        spans = _predicted_spans_topk([doc], topk_ratio=0.5, min_mentions_num=3, max_mentions_num=18)
        assert spans == []


# ---------------------------------------------------------------------------
# compute_metrics_topk
# ---------------------------------------------------------------------------

class TestComputeMetricsTopk:
    def test_params_stored(self):
        m = compute_metrics_topk([DOC_TOPK], topk_ratio=0.5, min_mentions_num=3, max_mentions_num=18)
        assert m["topk_ratio"] == 0.5
        assert m["min_mentions_num"] == 3
        assert m["max_mentions_num"] == 18

    def test_perfect_recall_when_k_large(self):
        # With very high lmin all spans are included → both gold spans covered
        m = compute_metrics_topk([DOC_TOPK], topk_ratio=0.5, min_mentions_num=10, max_mentions_num=20)
        assert m["recall"] == pytest.approx(1.0)

    def test_low_k_misses_gold(self):
        # K=1 (lmax=1): only [0,1] selected → [5,6] missed → recall < 1
        m = compute_metrics_topk([DOC_TOPK], topk_ratio=0.5, min_mentions_num=1, max_mentions_num=1)
        assert m["recall"] < 1.0

    def test_per_type_included_when_requested(self):
        m = compute_metrics_topk(
            [DOC_TOPK], topk_ratio=0.5, min_mentions_num=3, max_mentions_num=18,
            entity_types=["Method", "Dataset"],
        )
        assert "per_type" in m and "Method" in m["per_type"]

    def test_per_type_absent_without_arg(self):
        m = compute_metrics_topk([DOC_TOPK], topk_ratio=0.5, min_mentions_num=3, max_mentions_num=18)
        assert "per_type" not in m


# ---------------------------------------------------------------------------
# optimize_topk_params
# ---------------------------------------------------------------------------

class TestOptimizeTopkParams:
    def test_returns_best_and_all_results(self):
        best, all_results = optimize_topk_params(
            [DOC_TOPK],
            target_recall=1.0,
            topk_ratio_values=[0.3, 0.5, 1.0],
            min_mentions_num_values=[3, 5],
            max_mentions_num_values=[5, 10],
        )
        assert best is not None
        assert len(all_results) > 0

    def test_best_achieves_target_recall(self):
        best, _ = optimize_topk_params(
            [DOC_TOPK],
            target_recall=1.0,
            topk_ratio_values=[0.5],
            min_mentions_num_values=[5],
            max_mentions_num_values=[10],
        )
        assert best is not None
        assert best["recall"] >= 1.0

    def test_best_minimises_fp_share(self):
        best, all_results = optimize_topk_params(
            [DOC_TOPK],
            target_recall=0.5,
            topk_ratio_values=[0.3, 0.5, 1.0],
            min_mentions_num_values=[1, 3],
            max_mentions_num_values=[5, 10],
        )
        feasible = [r for r in all_results if r["recall"] >= 0.5]
        assert best["false_positive_share"] == min(r["false_positive_share"] for r in feasible)

    def test_infeasible_returns_none_as_best(self):
        # target_recall=1.0 but only K=1 available → gold span [5,6] never selected
        best, _ = optimize_topk_params(
            [DOC_TOPK],
            target_recall=1.0,
            topk_ratio_values=[0.05],
            min_mentions_num_values=[1],
            max_mentions_num_values=[1],
        )
        assert best is None

    def test_lmin_gt_lmax_skipped(self):
        # lmin=5, lmax=3 → invalid, should be skipped
        _, all_results = optimize_topk_params(
            [DOC_TOPK],
            target_recall=0.5,
            topk_ratio_values=[0.5],
            min_mentions_num_values=[5],
            max_mentions_num_values=[3],
        )
        assert all_results == []

    def test_all_results_sorted_feasible_first(self):
        best, all_results = optimize_topk_params(
            [DOC_TOPK],
            target_recall=0.5,
            topk_ratio_values=[0.3, 0.5, 1.0],
            min_mentions_num_values=[1, 3],
            max_mentions_num_values=[5, 10],
        )
        feasible   = [r for r in all_results if r["recall"] >= 0.5]
        infeasible = [r for r in all_results if r["recall"] <  0.5]
        # all_results has feasible first, then infeasible
        assert all_results[:len(feasible)] == feasible
        assert all_results[len(feasible):] == infeasible
