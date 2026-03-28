from __future__ import annotations

import dataclasses

import pytest

from gsapere.data.data_types import (
    CandidateStats,
    ContextWindow,
    Document,
    DocumentModel,
    DocumentPredictionModel,
    HGEREEntityPrediction,
    HGERERelationPrediction,
    NerSpan,
    NerSpanModel,
    ObjectEntry,
    PrunerSample,
    PrunerSpanPrediction,
    Relation,
    RelationModel,
    RelationSample,
    RelationSentence,
    RelationSpanCandidate,
    Sentence,
    SentenceModel,
    SentenceSubjectCandidate,
    SubjectInfo,
    SubwordIndex,
)


# ---------------------------------------------------------------------------
# Section 1 — Pydantic boundary models
# ---------------------------------------------------------------------------


class TestNerSpanModel:
    def test_from_list(self) -> None:
        span = NerSpanModel.model_validate([0, 2, "Material"])
        assert span.start == 0
        assert span.end == 2
        assert span.label == "Material"

    def test_from_dict(self) -> None:
        span = NerSpanModel.model_validate({"start": 1, "end": 3, "label": "Task"})
        assert span.start == 1
        assert span.label == "Task"

    def test_to_domain(self) -> None:
        span = NerSpanModel.model_validate([0, 2, "Material"])
        domain = span.to_domain()
        assert isinstance(domain, NerSpan)
        assert domain.start == 0
        assert domain.end == 2
        assert domain.label == "Material"


class TestRelationModel:
    def test_from_list(self) -> None:
        rel = RelationModel.model_validate([0, 2, 3, 5, "Used-for"])
        assert rel.subj_start == 0
        assert rel.subj_end == 2
        assert rel.obj_start == 3
        assert rel.obj_end == 5
        assert rel.label == "Used-for"

    def test_to_domain(self) -> None:
        rel = RelationModel.model_validate([0, 2, 3, 5, "Used-for"])
        domain = rel.to_domain()
        assert isinstance(domain, Relation)
        assert domain.label == "Used-for"


class TestSentenceModel:
    def test_defaults_empty(self) -> None:
        sent = SentenceModel.model_validate({"tokens": ["Hello", "world"]})
        assert sent.ner == []
        assert sent.relations == []

    def test_with_ner_and_relations(self) -> None:
        data = {
            "tokens": ["A", "B", "C", "D"],
            "ner": [[0, 1, "Task"], [2, 3, "Method"]],
            "relations": [[0, 1, 2, 3, "Used-for"]],
        }
        sent = SentenceModel.model_validate(data)
        assert len(sent.ner) == 2
        assert sent.ner[0].label == "Task"
        assert len(sent.relations) == 1

    def test_to_domain(self) -> None:
        data = {
            "tokens": ["A", "B"],
            "ner": [[0, 1, "Task"]],
            "relations": [],
        }
        sent = SentenceModel.model_validate(data).to_domain()
        assert isinstance(sent, Sentence)
        assert sent.tokens == ("A", "B")
        assert len(sent.ner) == 1
        assert sent.ner[0].start == 0


class TestDocumentModel:
    def test_round_trip(self) -> None:
        raw = {
            "doc_key": "doc_001",
            "sentences": [
                {
                    "tokens": ["A", "B"],
                    "ner": [[0, 1, "Task"]],
                    "relations": [],
                }
            ],
        }
        doc = DocumentModel.model_validate(raw)
        assert doc.doc_key == "doc_001"
        assert len(doc.sentences) == 1

    def test_to_domain(self) -> None:
        raw = {
            "doc_key": "doc_001",
            "sentences": [
                {"tokens": ["A", "B"], "ner": [[0, 1, "Task"]], "relations": []}
            ],
        }
        doc = DocumentModel.model_validate(raw).to_domain()
        assert isinstance(doc, Document)
        assert doc.doc_key == "doc_001"
        assert isinstance(doc.sentences, tuple)
        assert len(doc.sentences) == 1
        assert isinstance(doc.sentences[0], Sentence)

    def test_missing_ner_key_defaults_empty(self) -> None:
        """JSONL files sometimes omit the ner/relations keys for sentences."""
        raw = {
            "doc_key": "doc_002",
            "sentences": [{"tokens": ["X"]}],
        }
        doc = DocumentModel.model_validate(raw).to_domain()
        assert doc.sentences[0].ner == ()


class TestDocumentPredictionModel:
    def test_construction(self) -> None:
        pred = DocumentPredictionModel(
            doc_key="doc_001",
            ner_candidates=[(0, 1, 0.9)],
            ner_predicted=[(0, 1, "Task")],
            ner_predicted_proba=[(0, 1, "Task", 0.9)],
            relations_predicted=[(0, 1, 2, 3, "Used-for")],
            relations_predicted_proba=[(0, 1, 2, 3, "Used-for", 0.8)],
        )
        assert pred.doc_key == "doc_001"
        assert pred.ner_candidates[0][2] == pytest.approx(0.9)

    def test_json_round_trip(self) -> None:
        pred = DocumentPredictionModel(
            doc_key="doc_001",
            ner_candidates=[],
            ner_predicted=[],
            ner_predicted_proba=[],
            relations_predicted=[],
            relations_predicted_proba=[],
        )
        json_str = pred.model_dump_json()
        restored = DocumentPredictionModel.model_validate_json(json_str)
        assert restored.doc_key == "doc_001"


# ---------------------------------------------------------------------------
# Section 2 — Domain types (frozen dataclasses)
# ---------------------------------------------------------------------------


class TestNerSpan:
    def test_frozen(self) -> None:
        span = NerSpan(start=0, end=2, label="Task")
        with pytest.raises(dataclasses.FrozenInstanceError):
            span.label = "Method"  # type: ignore[misc]

    def test_hashable(self) -> None:
        span = NerSpan(start=0, end=2, label="Task")
        assert hash(span) is not None
        s = {span}
        assert span in s

    def test_equality(self) -> None:
        assert NerSpan(0, 2, "Task") == NerSpan(0, 2, "Task")
        assert NerSpan(0, 2, "Task") != NerSpan(0, 3, "Task")


class TestRelation:
    def test_frozen(self) -> None:
        rel = Relation(0, 2, 3, 5, "Used-for")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rel.label = "Result"  # type: ignore[misc]

    def test_hashable(self) -> None:
        rel = Relation(0, 2, 3, 5, "Used-for")
        assert hash(rel) is not None


class TestSentence:
    def test_frozen(self) -> None:
        sent = Sentence(
            tokens=("A", "B"),
            ner=(NerSpan(0, 1, "Task"),),
            relations=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sent.tokens = ("X",)  # type: ignore[misc]

    def test_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(Sentence)}
        assert fields == {"tokens", "ner", "relations"}


class TestDocument:
    def test_frozen(self) -> None:
        doc = Document(doc_key="d1", sentences=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.doc_key = "d2"  # type: ignore[misc]

    def test_hashable(self) -> None:
        doc = Document(doc_key="d1", sentences=())
        assert hash(doc) is not None


# ---------------------------------------------------------------------------
# Section 3 — Processing artifacts (mutable dataclasses)
# ---------------------------------------------------------------------------


class TestSubwordIndex:
    def test_construction(self) -> None:
        idx = SubwordIndex(
            token2subword=[0, 1, 3],
            subword2token=[0, 1, 1, 2],
            subword_sentence_boundaries=[0, 4],
            subword_start_positions=frozenset({0, 1, 3}),
        )
        assert idx.token2subword[0] == 0
        assert isinstance(idx.subword_start_positions, frozenset)

    def test_mutable(self) -> None:
        idx = SubwordIndex([], [], [], frozenset())
        idx.token2subword.append(0)
        assert len(idx.token2subword) == 1


class TestContextWindow:
    def test_construction(self) -> None:
        cw = ContextWindow(
            target_tokens=["A", "B"],
            doc_offset=0,
            left_context_length=0,
            sentence_length=2,
        )
        assert cw.sentence_length == 2


# ---------------------------------------------------------------------------
# Section 4 — Model input types (spot-check fields exist)
# ---------------------------------------------------------------------------


class TestPrunerSampleFields:
    def test_fields_exist(self) -> None:
        expected = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "labels",
            "mention_pos",
            "sent_subword_length",
            "sent_length",
            "example_index",
            "mentions",
        }
        actual = {f.name for f in dataclasses.fields(PrunerSample)}
        assert actual == expected


class TestRelationSampleFields:
    def test_fields_exist(self) -> None:
        expected = {
            "indices",
            "input_ids",
            "attention_mask",
            "position_ids",
            "sub_positions",
            "rel_labels",
            "ner_labels",
            "obj_token_pos",
            "n_ent",
            "subtoken_len",
            "sub",
        }
        actual = {f.name for f in dataclasses.fields(RelationSample)}
        assert actual == expected


class TestRelationSpanCandidateFields:
    def test_frozen_and_fields(self) -> None:
        cand = RelationSpanCandidate(token_start=0, token_end=2, label="Method")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cand.label = "Task"  # type: ignore[misc]


class TestSubjectInfoFields:
    def test_fields_exist(self) -> None:
        expected = {"token_span", "subtoken_pos", "gold_label_idx"}
        actual = {f.name for f in dataclasses.fields(SubjectInfo)}
        assert actual == expected


class TestObjectEntryFields:
    def test_fields_exist(self) -> None:
        expected = {"subtoken_pos", "ner_label_idx", "rel_label", "token_pos"}
        actual = {f.name for f in dataclasses.fields(ObjectEntry)}
        assert actual == expected


class TestSentenceSubjectCandidateFields:
    def test_fields_exist(self) -> None:
        expected = {"index", "subject_marked_tokens", "subject", "relations"}
        actual = {f.name for f in dataclasses.fields(SentenceSubjectCandidate)}
        assert actual == expected


class TestRelationSentenceFields:
    def test_fields_exist(self) -> None:
        expected = {"index", "subject_candidates", "ner_labels", "subword_tokens"}
        actual = {f.name for f in dataclasses.fields(RelationSentence)}
        assert actual == expected


# ---------------------------------------------------------------------------
# Section 5 — Prediction types (frozen, hashable)
# ---------------------------------------------------------------------------


class TestPrunerSpanPrediction:
    def test_frozen_and_hashable(self) -> None:
        pred = PrunerSpanPrediction(token_start=0, token_end=2, probability=0.95)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pred.probability = 0.5  # type: ignore[misc]
        assert hash(pred) is not None


class TestHGEREEntityPrediction:
    def test_frozen_and_hashable(self) -> None:
        pred = HGEREEntityPrediction(
            token_start=0, token_end=2, entity_type="Task", probability=0.9
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pred.entity_type = "Method"  # type: ignore[misc]
        assert hash(pred) is not None


class TestHGERERelationPrediction:
    def test_frozen_and_hashable(self) -> None:
        pred = HGERERelationPrediction(
            subj_start=0,
            subj_end=2,
            obj_start=3,
            obj_end=5,
            relation_type="Used-for",
            probability=0.85,
        )
        assert hash(pred) is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            pred.relation_type = "Result"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 6 — CandidateStats
# ---------------------------------------------------------------------------


class TestCandidateStats:
    def test_recall(self) -> None:
        stats = CandidateStats(n_gold=10, n_candidates=8, n_tp=7, n_fp=1, n_fn=3)
        assert stats.recall == pytest.approx(0.7)

    def test_precision(self) -> None:
        stats = CandidateStats(n_gold=10, n_candidates=8, n_tp=7, n_fp=1, n_fn=3)
        assert stats.precision == pytest.approx(7 / 8)

    def test_perfect_recall_and_precision(self) -> None:
        stats = CandidateStats(n_gold=5, n_candidates=5, n_tp=5, n_fp=0, n_fn=0)
        assert stats.recall == pytest.approx(1.0)
        assert stats.precision == pytest.approx(1.0)

    def test_zero_gold_returns_zero(self) -> None:
        stats = CandidateStats(n_gold=0, n_candidates=0, n_tp=0, n_fp=0, n_fn=0)
        assert stats.recall == 0.0
        assert stats.precision == 0.0

    def test_zero_candidates_returns_zero_precision(self) -> None:
        stats = CandidateStats(n_gold=3, n_candidates=0, n_tp=0, n_fp=0, n_fn=3)
        assert stats.precision == 0.0
        assert stats.recall == 0.0

    def test_frozen(self) -> None:
        stats = CandidateStats(n_gold=5, n_candidates=5, n_tp=5, n_fp=0, n_fn=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.n_tp = 3  # type: ignore[misc]
