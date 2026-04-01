"""Tests for evaluation/hgere.py — NER and RE metrics with double annotations."""

from __future__ import annotations

import pytest

from gsapere.evaluation.hgere import (
    compute_ner_metrics,
    compute_rel_metrics,
    compute_rel_metrics_with_ner,
)

# A doc with span (1,2) doubly annotated as TypeA and TypeB, plus span (3,4) as TypeC
DOUBLE_ANN_DOC = {
    "doc_id": "doc0",
    "ner": [[[1, 2, "TypeA"], [1, 2, "TypeB"], [3, 4, "TypeC"]]],
    "predicted_ner": [[[1, 2, "TypeA"], [3, 4, "TypeC"]]],
    "relations": [[]],
    "predicted_rel": [[]],
}


class TestNerMetricsDoubleAnnotation:
    def test_recall_uses_unique_spans(self) -> None:
        """Recall denominator must be unique span positions (2), not annotation count (3)."""
        m = compute_ner_metrics([DOUBLE_ANN_DOC])
        # 2 unique spans predicted correctly → recall = 2/2 = 1.0
        assert m["ner_recall"] == pytest.approx(1.0)

    def test_tp_any_matching_label(self) -> None:
        """TP when predicted type matches any valid gold type for the span."""
        m = compute_ner_metrics([DOUBLE_ANN_DOC])
        # span (1,2): model predicts TypeA ∈ {TypeA, TypeB} → TP
        # span (3,4): model predicts TypeC ∈ {TypeC} → TP
        assert m["ner_tp"] == 2

    def test_n_gold_unique_spans_reported(self) -> None:
        m = compute_ner_metrics([DOUBLE_ANN_DOC])
        assert m["ner_n_gold_unique_spans"] == 2

    def test_n_gold_annotations_reported(self) -> None:
        m = compute_ner_metrics([DOUBLE_ANN_DOC])
        assert m["ner_n_gold_annotations"] == 3  # TypeA + TypeB + TypeC

    def test_wrong_type_is_fp(self) -> None:
        """Predicted type not in any gold type → FP (not TP)."""
        doc = {
            "doc_id": "doc0",
            "ner": [[[1, 2, "TypeA"]]],
            "predicted_ner": [[[1, 2, "TypeX"]]],  # wrong type
            "relations": [[]],
            "predicted_rel": [[]],
        }
        m = compute_ner_metrics([doc])
        assert m["ner_tp"] == 0
        assert m["ner_fp"] == 1

    def test_no_double_annotation_unchanged(self) -> None:
        """For data without double annotations, behavior is unchanged."""
        doc = {
            "doc_id": "doc0",
            "ner": [[[1, 2, "TypeA"], [3, 4, "TypeC"]]],
            "predicted_ner": [[[1, 2, "TypeA"], [3, 4, "TypeC"]]],
            "relations": [[]],
            "predicted_rel": [[]],
        }
        m = compute_ner_metrics([doc])
        assert m["ner_recall"] == pytest.approx(1.0)
        assert m["ner_tp"] == 2


class TestRePlusMetricsDoubleAnnotation:
    def test_tp_when_predicted_type_matches_any_gold_type(self) -> None:
        """RE+ TP when predicted entity type is any valid gold type."""
        doc = {
            "doc_id": "doc0",
            "ner": [[[1, 2, "TypeA"], [1, 2, "TypeB"], [3, 4, "TypeC"]]],
            "predicted_ner": [[[1, 2, "TypeA"], [3, 4, "TypeC"]]],
            "relations": [[[1, 2, 3, 4, "Rel1"]]],
            "predicted_rel": [[[1, 2, 3, 4, "Rel1"]]],
        }
        m = compute_rel_metrics_with_ner([doc])
        assert m["re+_tp"] == 1
        assert m["re+_recall"] == pytest.approx(1.0)

    def test_tp_with_second_valid_type(self) -> None:
        """RE+ TP even when model predicts the 'other' valid type."""
        doc = {
            "doc_id": "doc0",
            "ner": [[[1, 2, "TypeA"], [1, 2, "TypeB"], [3, 4, "TypeC"]]],
            "predicted_ner": [
                [[1, 2, "TypeB"], [3, 4, "TypeC"]]
            ],  # TypeB instead of TypeA
            "relations": [[[1, 2, 3, 4, "Rel1"]]],
            "predicted_rel": [[[1, 2, 3, 4, "Rel1"]]],
        }
        m = compute_rel_metrics_with_ner([doc])
        assert m["re+_tp"] == 1

    def test_fp_when_type_not_in_gold(self) -> None:
        """RE+ FP when predicted entity type is not any valid gold type."""
        doc = {
            "doc_id": "doc0",
            "ner": [[[1, 2, "TypeA"], [3, 4, "TypeC"]]],
            "predicted_ner": [[[1, 2, "TypeX"], [3, 4, "TypeC"]]],  # TypeX not in gold
            "relations": [[[1, 2, 3, 4, "Rel1"]]],
            "predicted_rel": [[[1, 2, 3, 4, "Rel1"]]],
        }
        m = compute_rel_metrics_with_ner([doc])
        assert m["re+_tp"] == 0
        assert m["re+_fn"] == 1

    def test_n_gold_equals_unique_relations(self) -> None:
        """RE+ n_gold is unique relations (not expanded by type combos)."""
        doc = {
            "doc_id": "doc0",
            "ner": [[[1, 2, "TypeA"], [1, 2, "TypeB"], [3, 4, "TypeC"]]],
            "predicted_ner": [[[1, 2, "TypeA"], [3, 4, "TypeC"]]],
            "relations": [[[1, 2, 3, 4, "Rel1"]]],
            "predicted_rel": [[[1, 2, 3, 4, "Rel1"]]],
        }
        m = compute_rel_metrics_with_ner([doc])
        assert (
            m["re+_n_gold"] == 1
        )  # 1 unique relation, not 2 (TypeA×TypeC + TypeB×TypeC)


# ---------------------------------------------------------------------------
# Symmetric relation tests
# ---------------------------------------------------------------------------

_SYM_LABELS = ("Similar",)

# Minimal doc with a symmetric relation in canonical direction (A→B, A < B)
_SYM_DOC_BASE = {
    "doc_id": "doc_sym",
    "ner": [[[1, 1, "Method"], [3, 3, "Method"]]],
    "predicted_ner": [[[1, 1, "Method"], [3, 3, "Method"]]],
    "relations": [[[1, 1, 3, 3, "Similar"]]],  # canonical: (1,1) < (3,3)
    "predicted_rel": [[[1, 1, 3, 3, "Similar"]]],
}

# Same relation but predicted in reverse direction
_SYM_DOC_REVERSED_PRED = {
    **_SYM_DOC_BASE,
    "predicted_rel": [[[3, 3, 1, 1, "Similar"]]],  # reversed direction
}

# Gold has reverse direction (both directions present in annotation)
_SYM_DOC_BOTH_GOLD = {
    **_SYM_DOC_BASE,
    "relations": [[[1, 1, 3, 3, "Similar"], [3, 3, 1, 1, "Similar"]]],
    "predicted_rel": [[[1, 1, 3, 3, "Similar"]]],
}


class TestSymmetricRelationEval:
    def test_canonical_direction_is_tp(self) -> None:
        """Prediction in canonical direction is TP when sym_labels provided."""
        m = compute_rel_metrics([_SYM_DOC_BASE], sym_labels=_SYM_LABELS)
        assert m["re_tp"] == 1
        assert m["re_recall"] == pytest.approx(1.0)

    def test_reversed_prediction_is_tp(self) -> None:
        """Reversed-direction prediction counts as TP after normalisation."""
        m = compute_rel_metrics([_SYM_DOC_REVERSED_PRED], sym_labels=_SYM_LABELS)
        assert m["re_tp"] == 1
        assert m["re_recall"] == pytest.approx(1.0)
        assert m["re_fp"] == 0

    def test_no_sym_labels_reversed_is_fp(self) -> None:
        """Without sym_labels, reversed prediction is a FP (direction matters)."""
        m = compute_rel_metrics([_SYM_DOC_REVERSED_PRED])  # no sym_labels
        assert m["re_tp"] == 0
        assert m["re_fp"] == 1

    def test_duplicate_gold_directions_count_once(self) -> None:
        """Gold with both A→B and B→A for a sym label deduplicates to 1 gold entry."""
        m = compute_rel_metrics([_SYM_DOC_BOTH_GOLD], sym_labels=_SYM_LABELS)
        assert m["re_n_gold"] == 1

    def test_asymmetric_label_direction_still_matters(self) -> None:
        """For non-symmetric labels, reversed direction is still a FP."""
        doc = {
            "doc_id": "doc_asym",
            "ner": [[[1, 1, "Method"], [3, 3, "Dataset"]]],
            "predicted_ner": [[[1, 1, "Method"], [3, 3, "Dataset"]]],
            "relations": [[[1, 1, 3, 3, "Used-for"]]],
            "predicted_rel": [[[3, 3, 1, 1, "Used-for"]]],  # wrong direction
        }
        m = compute_rel_metrics([doc], sym_labels=_SYM_LABELS)  # Used-for is NOT sym
        assert m["re_tp"] == 0
        assert m["re_fp"] == 1

    def test_re_plus_reversed_prediction_is_tp(self) -> None:
        """RE+ also treats reversed sym prediction as TP after normalisation."""
        m = compute_rel_metrics_with_ner(
            [_SYM_DOC_REVERSED_PRED], sym_labels=_SYM_LABELS
        )
        assert m["re+_tp"] == 1
        assert m["re+_recall"] == pytest.approx(1.0)
