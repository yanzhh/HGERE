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

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


# ── 1. Pydantic models (I/O boundary) ─────────────────────────────────────


class NerSpanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    doc_key: str
    sentences: list[SentenceModel]

    def to_domain(self) -> Document:
        return Document(
            doc_key=self.doc_key,
            sentences=tuple(s.to_domain() for s in self.sentences),
        )


class DocumentPredictionModel(BaseModel):
    """Service output — validated and JSON-serializable."""

    model_config = ConfigDict(extra="forbid")

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
class SubjectInfo:
    """Subject entity info carried by a SentenceSubjectCandidate."""

    token_span: tuple[int, int, str]  # (token_begin, token_end, predicted_label)
    subtoken_pos: tuple[int, int]  # (subtoken_begin, subtoken_end) in sentence context
    gold_label_idx: int  # gold NER label index


@dataclass
class ObjectEntry:
    """One object candidate paired with a subject."""

    subtoken_pos: tuple[int, int]  # (subtoken_begin, subtoken_end) in sentence context
    ner_label_idx: int  # predicted NER label index
    rel_label: int  # relation label (0 = no relation)
    token_pos: tuple[int, int]  # (token_begin, token_end) in document


@dataclass(repr=False)
class SentenceSubjectCandidate:
    """One subject candidate for a sentence.

    For a sentence with m entities there are m subjects and up to m**2 relations.
    One instance records all object/relation pairs for a single subject.
    """

    index: tuple[int, int]  # (doc_idx, sent_idx)
    input_ids: (
        np.ndarray
    )  # token IDs (int32) with subject boundary markers, without padding
    subject: SubjectInfo
    relations: list[ObjectEntry]  # one entry per object candidate

    def __post_init__(self) -> None:
        self.obj_subtoken_pos: list[tuple[int, int]] = [
            e.subtoken_pos for e in self.relations
        ]
        self.obj_token_pos: list[tuple[int, int]] = [
            e.token_pos for e in self.relations
        ]

    @property
    def n_ent(self) -> int:
        return len(self.relations)

    @property
    def rel_labels(self) -> list[int]:
        return [e.rel_label for e in self.relations]

    def __repr__(self) -> str:
        if self.n_ent == 0:
            return "<0 relations; ...>"
        return (
            f"<{self.n_ent} relations; subject: {self.subject.token_span}; "
            f"objs: obj_subtokens-{self.obj_subtoken_pos}, obj_tokens-{self.obj_token_pos}>"
        )


@dataclass(repr=False)
class RelationSentence:
    """All subject candidates for one sentence.

    Acts as the top-level data unit for RelationDataset: one __getitem__ call
    returns tensors derived from one RelationSentence.
    """

    index: tuple[int, int]  # (doc_idx, sent_idx)
    subject_candidates: list[SentenceSubjectCandidate]
    ner_labels: list[int]  # gold NER label indices for each object candidate position
    subword_tokens: list[
        str
    ]  # plain (unmarked) subword token sequence; used for repr/debugging

    def __post_init__(self) -> None:
        # All subject candidates share the same object set, so token positions
        # of objects are taken from the first candidate.
        self.obj_token_pos: list[tuple[int, int]] = (
            self.subject_candidates[0].obj_token_pos if self.subject_candidates else []
        )

    @property
    def n_subjects(self) -> int:
        """Number of subject candidates (= entity candidates) in this sentence."""
        return len(self.subject_candidates)

    def __repr__(self) -> str:
        _PREVIEW = 6
        tokens = self.subword_tokens
        preview = tokens[:_PREVIEW]
        suffix = f" +{len(tokens) - _PREVIEW}" if len(tokens) > _PREVIEW else ""
        return (
            f"<RelationSentence {self.index}; {self.n_subjects} subjects; "
            f"tokens: {preview}{suffix}; obj_positions: {self.obj_token_pos}>"
        )


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

    indices: tuple[int, int]
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


# ── 5. Candidate quality statistics ───────────────────────────────────────


@dataclass(frozen=True)
class CandidateStats:
    """Pruner candidate quality statistics computed during RelationDataset._build_index().

    Measures how well the pruner candidates cover the gold entities for one
    dataset split (train / dev / test).  Matching is span-level only
    (token_start, token_end) — labels are ignored, as the pruner predicts no types.
    """

    n_gold: int  # total gold entity mentions
    n_candidates: int  # total candidate spans fed to HGERE
    n_tp: int  # candidates whose span matches a gold entity
    n_fp: int  # candidates with no matching gold entity
    n_fn: int  # gold entities not covered by any candidate

    @property
    def recall(self) -> float:
        return self.n_tp / self.n_gold if self.n_gold > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.n_tp / self.n_candidates if self.n_candidates > 0 else 0.0


@dataclass(frozen=True)
class TruncationStats:
    """Entity-level truncation counts accumulated during RelationDataset._build_index().

    Two independent truncation sources are tracked:

    max_ents truncation (applied first, before tokenisation):
      n_maxents_sents      — sentences where the candidate list was capped by max_ents.
      n_maxents_dropped    — total candidates removed by the cap.

    Sequence-length truncation (applied during tokenisation):
      n_seqlen_subjects    — subject candidates dropped because their subtoken end
                             position fell outside max_seq_length.
      n_seqlen_objects     — object candidates dropped for the same reason
                             (invisible in entity counts, but reduces relation pairs).
    """

    n_maxents_sents: int
    n_maxents_dropped: int
    n_seqlen_subjects: int
    n_seqlen_objects: int


# ── 6. Prediction types (frozen dataclasses) ──────────────────────────────


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
