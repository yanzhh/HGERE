"""
data_types.py — Single source of truth for all data structures in the HGERE pipeline.

Organized into five sections by data-flow stage:
  1. Pydantic models     — I/O boundary (JSONL load, service output)
  2. Domain types        — frozen dataclasses used throughout processing
  3. Processing artifacts — mutable dataclasses produced by tokenization
  4. Model input types   — one item per dataset __getitem__ call
  5. Prediction types    — frozen output dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, model_validator


# ── 1. Pydantic models (I/O boundary) ─────────────────────────────────────


class NerSpanModel(BaseModel):
    start: int
    end: int
    label: str

    @model_validator(mode="before")
    @classmethod
    def from_list(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return {"start": v[0], "end": v[1], "label": v[2]}
        return v

    def to_domain(self) -> NerSpan:
        return NerSpan(start=self.start, end=self.end, label=self.label)


class RelationModel(BaseModel):
    subj_start: int
    subj_end: int
    obj_start: int
    obj_end: int
    label: str

    @model_validator(mode="before")
    @classmethod
    def from_list(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return {
                "subj_start": v[0],
                "subj_end": v[1],
                "obj_start": v[2],
                "obj_end": v[3],
                "label": v[4],
            }
        return v

    def to_domain(self) -> Relation:
        return Relation(
            subj_start=self.subj_start,
            subj_end=self.subj_end,
            obj_start=self.obj_start,
            obj_end=self.obj_end,
            label=self.label,
        )


class SentenceModel(BaseModel):
    tokens: list[str]
    ner: list[NerSpanModel] = []
    relations: list[RelationModel] = []

    def to_domain(self) -> Sentence:
        return Sentence(
            tokens=tuple(self.tokens),
            ner=tuple(s.to_domain() for s in self.ner),
            relations=tuple(r.to_domain() for r in self.relations),
        )


class DocumentModel(BaseModel):
    doc_key: str
    sentences: list[SentenceModel]

    def to_domain(self) -> Document:
        return Document(
            doc_key=self.doc_key,
            sentences=tuple(s.to_domain() for s in self.sentences),
        )


class DocumentPredictionModel(BaseModel):
    """Service output — validated and JSON-serializable."""

    doc_key: str
    ner_candidates: list[tuple[int, int, float]]
    ner_predicted: list[tuple[int, int, str]]
    ner_predicted_proba: list[tuple[int, int, str, float]]
    relations_predicted: list[tuple[int, int, int, int, str]]
    relations_predicted_proba: list[tuple[int, int, int, int, str, float]]


# ── 2. Domain types (frozen dataclasses) ──────────────────────────────────


@dataclass(frozen=True)
class NerSpan:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Relation:
    subj_start: int
    subj_end: int
    obj_start: int
    obj_end: int
    label: str


@dataclass(frozen=True)
class Sentence:
    tokens: tuple[str, ...]
    ner: tuple[NerSpan, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True)
class Document:
    doc_key: str
    sentences: tuple[Sentence, ...]


# ── 3. Processing artifacts (mutable dataclasses) ─────────────────────────


@dataclass
class SubwordIndex:
    """Token-to-subword and subword-to-token mapping for one document.
    Built once per document during tokenization; shared by all sentences."""

    token2subword: list[int]
    subword2token: list[int]
    subword_sentence_boundaries: list[int]
    subword_start_positions: frozenset[int]
    subword_tokens: list[str] = field(default_factory=list)


@dataclass
class ContextWindow:
    """Result of _get_context_window() for one sentence."""

    target_tokens: list[str]
    doc_offset: int
    left_context_length: int
    sentence_length: int


# ── 4. Model input types ───────────────────────────────────────────────────


@dataclass
class PrunerSample:
    """One training/eval item returned by PrunerDataset.__getitem__().
    Corresponds to one sentence window (may be a chunk of a long sentence)."""

    input_ids: list[int]
    attention_mask: Any  # torch.Tensor — 1D or 2D depending on model_type
    position_ids: list[int]
    labels: list[int]  # length max_pair_length; -1 = padding
    mention_pos: list[tuple[int, int]]  # length max_pair_length; (0,0) = padding
    sent_subword_length: int
    sent_length: int | None  # only when evaluate=True
    example_index: tuple[int, int]  # (doc_idx, sent_idx) — metadata
    mentions: list[tuple[int, int]]  # token-level (start, end) — metadata


@dataclass
class RelationSample:
    """One item returned by RelationDataset.__getitem__().
    All tensor fields stored as torch.Tensor; list fields are metadata."""

    indexs: tuple[int, int]
    input_ids: Any  # Tensor [n_ent, seq_len + 2*n_ent]
    attention_mask: Any  # Tensor [n_ent, seq_len+2*n_ent, seq_len+2*n_ent]
    position_ids: Any  # Tensor [n_ent, seq_len + 2*n_ent]
    sub_positions: Any  # Tensor [n_ent, 2]
    rel_labels: Any  # Tensor [n_ent, n_ent]
    ner_labels: Any  # Tensor [n_ent]
    obj_token_pos: list[tuple[int, int]]
    n_ent: int
    subtoken_len: int
    sub: list  # raw subject metadata (kept for decode)


@dataclass(frozen=True)
class RelationSpanCandidate:
    """One entity candidate as seen by RelationDataset."""

    token_start: int
    token_end: int
    label: str


@dataclass
class SubjectObjectPair:
    """One (subject, all-objects) group per subject entity.
    Replaces SentenceSubjectCandidate in relation_dataset.py."""

    doc_idx: int
    sent_idx: int
    subject_token_start: int
    subject_token_end: int
    subject_label: str
    subject_label_gold_id: int
    sub_tokens: list[str]
    sub_subtoken_pos: tuple[int, int]
    object_candidates: list[tuple[tuple[int, int, int], int, tuple[int, int]]]
    rel_labels: list[int]
    ner_labels_gold: list[int]
    n_ent: int


@dataclass
class RelationSentence:
    """All subject groups for one sentence.
    Replaces Sentence internal class in relation_dataset.py."""

    doc_idx: int
    sent_idx: int
    items: list[SubjectObjectPair]
    ner_labels_gold: list[int]
    words: list[str]


# ── 5. Prediction types (frozen dataclasses) ──────────────────────────────


@dataclass(frozen=True)
class PrunerSpanPrediction:
    """Output of pruner inference for one span."""

    token_start: int
    token_end: int
    probability: float


@dataclass(frozen=True)
class HGEREEntityPrediction:
    token_start: int
    token_end: int
    entity_type: str
    probability: float


@dataclass(frozen=True)
class HGERERelationPrediction:
    subj_start: int
    subj_end: int
    obj_start: int
    obj_end: int
    relation_type: str
    probability: float
