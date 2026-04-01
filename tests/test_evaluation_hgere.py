"""Tests for evaluation/hgere.py — NER and RE metrics with double annotations."""

from __future__ import annotations

import pytest

from gsapere.evaluation.hgere import (
    compute_ner_metrics,
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
