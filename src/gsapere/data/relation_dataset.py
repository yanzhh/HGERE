from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import numpy as np

import torch
from torch.utils.data import DataLoader

from ..data.base_dataset import DocumentDataset
from ..data.collators import _collate_relation_batch
from ..data.config import RelationDatasetParams
from ..data.data_types import (
    CandidateStats,
    ObjectEntry,
    RelationSentence,
    SentenceSubjectCandidate,
    SubjectInfo,
    TruncationStats,
)
from ..data.samplers import (
    BucketSampler,
    DistributedBucketSampler,
    create_batches,
    create_shuffled_batches,
    create_size_sorted_batches,
)
from ..pruner.evaluate import _select_sent_spans


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------


class RelationDataset(DocumentDataset):
    """HGERE relation-extraction dataset.

    One dataset item = one sentence (all subject/object pairs for that sentence).
    Inherits shared tokenization and context-window infrastructure from
    DocumentDataset.

    Constructor signature is identical to the original RelationDataset to preserve
    backward compatibility with run_hgnn.py, hgere/inference.py, and
    infer_pruner_augmented.py.
    """

    def __init__(
        self,
        logger: Any,
        tokenizer: Any,
        labels: Any,
        file_path: str,
        params: RelationDatasetParams,
    ) -> None:
        self.logger = logger
        self.max_ents = params.max_ents

        self.use_typemarker = params.use_typemarker
        self.model_type = params.model_type
        self.no_sym = params.no_sym
        self.nocross = params.nocross
        self.local_rank = params.local_rank
        self.split = params.split
        self.use_gold_ner = params.use_gold_ner
        self.dataset_id = params.dataset_id
        # Pre-resolved [unusedX] token ID to substitute at position 0 instead of [CLS].
        # None means no substitution (standard behaviour).
        self._dataset_cls_token_id = params.dataset_cls_token_id

        self.ner_label_list: list[str] = labels.ner
        self.sym_labels: list[str] = labels.rel.symmetric(only_nil=self.no_sym)
        self.label_list: list[str] = labels.rel.all

        # Populated by _build_index()
        self.data: list[RelationSentence] = []
        self.sizes: list[int] = []
        self.ner_golden_labels: set[tuple[Any, ...]] = set()
        self.ner_golden_span_types: dict[tuple, set[str]] = {}
        self.golden_labels: set[tuple[Any, ...]] = set()
        self.golden_labels_with_ner: set[tuple[Any, ...]] = set()
        # Maps (doc_idx, sent_idx) → candidate NER spans fed into HGERE.
        # Used during evaluation to compute NER recall against gold entities.
        self.global_predicted_ners: dict[
            tuple[int, int], list[tuple[int, int, str]]
        ] = {}
        # Running totals of gold relation/entity instances (for recall denominator).
        self.ner_tot_recall: int = 0
        self.tot_recall: int = 0
        # Populated by _build_index() after processing all documents.
        self.candidate_stats: CandidateStats = CandidateStats(
            n_gold=0, n_candidates=0, n_tp=0, n_fp=0, n_fn=0
        )
        self.truncation_stats: TruncationStats = TruncationStats(
            n_maxents_sents=0,
            n_maxents_dropped=0,
            n_seqlen_subjects=0,
            n_seqlen_objects=0,
        )
        # Mutable accumulators updated during _build_index(); frozen into
        # truncation_stats at the end of build().
        self._trunc_maxents_sents: int = 0
        self._trunc_maxents_dropped: int = 0
        self._trunc_seqlen_subjects: int = 0
        self._trunc_seqlen_objects: int = 0

        # Populated lazily by _preload() when params.preload is True.
        self._data_tokenized: list[dict[str, Any]] = []

        # Base __init__ indexes the file (byte offsets) and sets self._file_path.
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            max_seq_length=params.max_seq_length,
            lazy=False,
            doc_limit=params.doc_limit,
            pre_filter_params=params.pre_filter_params,
        )
        self._build_index()

        self.is_preloaded = params.preload
        if params.preload:
            self.logger.info("Load whole dataset ...")
            self._preload()
            self.logger.info("Load whole dataset ended.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.is_preloaded:
            return self._data_tokenized[idx]
        return self.prepare_item(idx)

    def _preload(self) -> None:
        """Eagerly convert all sentences to tensor dicts and cache them in memory."""
        self._data_tokenized = [self.prepare_item(idx) for idx in range(len(self.data))]

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build the in-memory index of RelationSentence objects from the JSONL file.

        For each document and sentence:
        1. Tokenises the document and computes the context window.
        2. Collects gold entity labels (for supervision) and candidate NER spans
           (from the pruner or gold annotations, depending on config).
        3. Builds a relation label lookup (pos2label) from gold relations,
           including reverse directions for symmetric/asymmetric relations.
        4. For every candidate entity acting as subject, injects subject boundary
           markers into the subword token sequence and pairs it with every other
           candidate entity as object, recording the gold relation label for each
           pair.
        5. Assembles a RelationSentence dataclass and appends it to self.data.
        """
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}
        # Reserve 4 slots for [CLS], [SEP], and the two subject marker tokens.
        max_num_subwords = self.max_seq_length - 4

        n_predicted_ner_total = 0  # spans from predicted_ner (final_pruning output)

        # Dataset-level drop accumulators — emitted as single warnings after the loop.
        ner_dropped_total: Counter[str] = Counter()
        self_rel_total: Counter[tuple[str, str]] = Counter()
        missing_cand_total: Counter[tuple[str, str, str]] = Counter()

        for doc_idx in range(len(self._offsets)):
            doc = self._load_raw_document(doc_idx)
            tok_idx = self._tokenize_document(doc)
            token2subword = tok_idx.token2subword
            subword_sentence_boundaries = tok_idx.subword_sentence_boundaries

            # Load raw JSON to access predicted_ner / predicted_ner_proba fields
            # (these are not stored in the frozen domain objects from the base class).
            with open(self._file_path, "rb") as f:
                f.seek(self._offsets[doc_idx])
                raw = json.loads(f.readline())

            ner_candidates_all = self._select_candidate_ner(raw)

            if self.do_pre_filter:
                for sent_cands in raw.get("predicted_ner", []):
                    n_predicted_ner_total += len(sent_cands)

            n_sents = len(doc.sentences)
            ner_gold_all = raw.get("ner", [[] for _ in range(n_sents)])
            relations_all = raw.get("relations", [[] for _ in range(n_sents)])

            for sent_rel in relations_all:
                self.tot_recall += len(sent_rel)

            all_subwords: list[str] = tok_idx.subword_tokens

            for sentence_idx, sent in enumerate(doc.sentences):
                ner_candidates_sent = list(ner_candidates_all[sentence_idx])
                if self.max_ents is not None:
                    dropped = max(0, len(ner_candidates_sent) - self.max_ents)
                    if dropped:
                        self._trunc_maxents_sents += 1
                        self._trunc_maxents_dropped += dropped
                    ner_candidates_sent = ner_candidates_sent[: self.max_ents]
                sentence_relations = relations_all[sentence_idx]
                ner_sent_gold = ner_gold_all[sentence_idx]

                self.ner_tot_recall += len(ner_sent_gold)

                (
                    entity_labels_gold,
                    gold_span_multi_types,
                    ner_golden_new,
                    ner_dropped_sent,
                ) = self._collect_gold_entity_labels(
                    ner_sent_gold=ner_sent_gold,
                    ner_label_map=ner_label_map,
                    doc_idx=doc_idx,
                    sentence_idx=sentence_idx,
                    sentence_relations=sentence_relations,
                )
                ner_dropped_total.update(ner_dropped_sent)
                self.ner_golden_labels.update(ner_golden_new)

                # Store multi-type lookup for evaluation
                for span, labels in gold_span_multi_types.items():
                    key = (doc_idx, sentence_idx) + span
                    self.ner_golden_span_types[key] = labels

                self.global_predicted_ners[(doc_idx, sentence_idx)] = list(
                    ner_candidates_sent
                )

                doc_sent_start = subword_sentence_boundaries[sentence_idx]
                doc_sent_end = subword_sentence_boundaries[sentence_idx + 1]

                cw = self._get_context_window(
                    subword_tokens=all_subwords,
                    doc_sent_start=doc_sent_start,
                    doc_sent_end=doc_sent_end,
                    max_num_subwords=max_num_subwords,
                    nocross=self.nocross,
                )
                doc_offset = cw.doc_offset

                target_tokens = (
                    [self.tokenizer.cls_token]
                    + cw.target_tokens[: self.max_seq_length - 4]
                    + [self.tokenizer.sep_token]
                )

                candidate_span_set = {
                    (start, end) for start, end, _ in ner_candidates_sent
                }

                (
                    pos2label,
                    golden_new,
                    golden_with_ner_new,
                    self_rel_sent,
                    missing_cand_sent,
                ) = self._build_pos2label(
                    sentence_relations=sentence_relations,
                    label_map=label_map,
                    sym_labels=self.sym_labels,
                    entity_labels_gold=entity_labels_gold,
                    gold_span_multi_types=gold_span_multi_types,
                    doc_idx=doc_idx,
                    sentence_idx=sentence_idx,
                    candidate_span_set=candidate_span_set,
                )
                self_rel_total.update(self_rel_sent)
                missing_cand_total.update(missing_cand_sent)
                self.golden_labels.update(golden_new)
                self.golden_labels_with_ner.update(golden_with_ner_new)

                subject_candidates: list[SentenceSubjectCandidate] = []
                ner_labels: list[int] = []

                for subj_begin, subj_end, subj_label in ner_candidates_sent:
                    if subj_begin == len(token2subword):
                        continue

                    subj_begin_sent = token2subword[subj_begin] - doc_offset + 1
                    subj_end_sent = token2subword[subj_end + 1] - doc_offset

                    subj_label_gold = ner_label_map.get(
                        entity_labels_gold.get((subj_begin, subj_end), "NIL"), 0
                    )

                    subj_marker_begin, subj_marker_end = self._subject_markers(
                        subj_label_gold
                    )

                    subject_marked_tokens = (
                        target_tokens[:subj_begin_sent]
                        + [subj_marker_begin]
                        + target_tokens[subj_begin_sent : subj_end_sent + 1]
                        + [subj_marker_end]
                        + target_tokens[subj_end_sent + 1 :]
                    )
                    subj_end_sent += 2  # account for the two inserted marker tokens

                    if subj_end_sent >= self.max_seq_length - 1:
                        self._trunc_seqlen_subjects += 1
                        continue

                    object_candidates, ner_labels = self._build_object_candidates(
                        subj_begin=subj_begin,
                        subj_end=subj_end,
                        ner_candidates_sent=ner_candidates_sent,
                        token2subword=token2subword,
                        doc_offset=doc_offset,
                        pos2label=pos2label,
                        ner_label_map=ner_label_map,
                        entity_labels_gold=entity_labels_gold,
                    )

                    subject_candidates.append(
                        SentenceSubjectCandidate(
                            index=(doc_idx, sentence_idx),
                            subject_marked_tokens=subject_marked_tokens,
                            subject=SubjectInfo(
                                token_span=(subj_begin, subj_end, subj_label),
                                subtoken_pos=(subj_begin_sent, subj_end_sent),
                                gold_label_idx=subj_label_gold,
                            ),
                            relations=object_candidates,
                        )
                    )

                # ner_labels from the last subject candidate is used for the sentence.
                # All subject candidates share the same object set, so ner_labels is
                # identical across iterations — only rel_labels differ per subject.
                ner_labels_sent: list[int] = ner_labels if subject_candidates else []

                new_sent = RelationSentence(
                    index=(doc_idx, sentence_idx),
                    subject_candidates=subject_candidates,
                    ner_labels=ner_labels_sent,
                    subword_tokens=target_tokens,
                )
                self.data.append(new_sent)
                self.sizes.append(new_sent.n_subjects)

        # --- Dataset-level drop warnings (one per drop type, training only) ---
        if self.split == "train" and ner_dropped_total:
            summary = ", ".join(
                f"{lbl}:{cnt}" for lbl, cnt in sorted(ner_dropped_total.items())
            )
            self.logger.warning(
                "NER double-annotation: %d label(s) dropped across dataset "
                "(span with multiple gold labels — training uses most-connected label, "
                "alphabetical as tie-breaker). Dropped label counts: %s",
                sum(ner_dropped_total.values()),
                summary,
            )

        if self.split == "train" and self_rel_total:
            summary = ", ".join(
                f"({ent},{rel}):{cnt}"
                for (ent, rel), cnt in sorted(self_rel_total.items())
            )
            self.logger.warning(
                "Self-relations: %d dropped (model cannot predict self-relations). "
                "Counts by (entity_label, rel_label): %s",
                sum(self_rel_total.values()),
                summary,
            )

        if self.split == "train" and missing_cand_total:
            summary = ", ".join(
                f"({s},{r},{o}):{cnt}"
                for (s, r, o), cnt in sorted(missing_cand_total.items())
            )
            self.logger.warning(
                "Relations not trainable: %d dropped (span absent from candidate set). "
                "Counts by (subj_label, rel_label, obj_label): %s",
                sum(missing_cand_total.values()),
                summary,
            )

        self.candidate_stats = self._compute_candidate_stats(
            self.global_predicted_ners, self.ner_golden_labels
        )
        self.truncation_stats = TruncationStats(
            n_maxents_sents=self._trunc_maxents_sents,
            n_maxents_dropped=self._trunc_maxents_dropped,
            n_seqlen_subjects=self._trunc_seqlen_subjects,
            n_seqlen_objects=self._trunc_seqlen_objects,
        )
        self.logger.info(
            "Truncation stats: "
            "max_ents: %d sents affected, %d candidates dropped | "
            "seq_len: %d subjects dropped, %d objects dropped",
            self._trunc_maxents_sents,
            self._trunc_maxents_dropped,
            self._trunc_seqlen_subjects,
            self._trunc_seqlen_objects,
        )
        if self.do_pre_filter:
            self.logger.info(
                "Candidate stats (pre_filter applied): n_gold=%d  "
                "n_predicted_ner(final_pruning)=%d  n_candidates(pre_filter)=%d  "
                "TP=%d  FP=%d  FN=%d  recall=%.3f  precision=%.3f",
                self.candidate_stats.n_gold,
                n_predicted_ner_total,
                self.candidate_stats.n_candidates,
                self.candidate_stats.n_tp,
                self.candidate_stats.n_fp,
                self.candidate_stats.n_fn,
                self.candidate_stats.recall,
                self.candidate_stats.precision,
            )
        else:
            self.logger.info(
                "Candidate stats: n_gold=%d  n_candidates=%d  "
                "TP=%d  FP=%d  FN=%d  recall=%.3f  precision=%.3f",
                self.candidate_stats.n_gold,
                self.candidate_stats.n_candidates,
                self.candidate_stats.n_tp,
                self.candidate_stats.n_fp,
                self.candidate_stats.n_fn,
                self.candidate_stats.recall,
                self.candidate_stats.precision,
            )

    @staticmethod
    def _compute_candidate_stats(
        global_predicted_ners: dict[tuple[int, int], list[tuple[int, int, str]]],
        ner_golden_labels: set[tuple[Any, ...]],
    ) -> CandidateStats:
        """Compute TP/FP/FN counts comparing candidate spans against gold entities.

        Matching is span-level only (token_start, token_end) — labels are ignored,
        as the pruner produces no type predictions.
        """
        gold_spans: set[tuple] = {
            (sent_id, span) for sent_id, span, _ in ner_golden_labels
        }
        candidate_spans: set[tuple] = {
            (sent_id, (start, end))
            for sent_id, spans in global_predicted_ners.items()
            for start, end, _ in spans
        }
        n_tp = len(gold_spans & candidate_spans)
        n_fp = len(candidate_spans - gold_spans)
        n_fn = len(gold_spans - candidate_spans)
        return CandidateStats(
            n_gold=len(gold_spans),
            n_candidates=len(candidate_spans),
            n_tp=n_tp,
            n_fp=n_fp,
            n_fn=n_fn,
        )

    @staticmethod
    def _collect_gold_entity_labels(
        ner_sent_gold: list[tuple[int, int, str]],
        ner_label_map: dict[str, int],
        doc_idx: int,
        sentence_idx: int,
        sentence_relations: list[tuple[int, int, int, int, str]],
    ) -> tuple[
        dict[tuple[int, int], str],
        dict[tuple[int, int], set[str]],
        set[tuple[Any, ...]],
        Counter[str],
    ]:
        """Build gold entity label maps for one sentence.

        For spans with multiple annotations (different labels at the same position),
        picks the label with the highest relation connectivity (number of non-self
        relations, at the sentence level, involving spans with that label — counted
        only over single-label spans to avoid circular dependency).  Alphabetical
        order is the tie-breaker when connectivity is equal.

        Connectivity is computed from single-label spans only: for each non-self
        relation endpoint that belongs to a singly-annotated span, every label
        of that span contributes +1 to its label's connectivity score.  This
        ensures that multi-label spans do not bias the count for themselves.

        Returns:
        - entity_labels_gold: span → single training label (non-NIL).
        - gold_span_multi_types: span → set of all valid gold labels.
        - ner_golden_additions: set entries for ner_golden_labels (all annotations).
        - dropped_by_label: Counter of labels dropped due to double annotations
          (to be accumulated by the caller for a dataset-level warning).
        """
        gold_span_multi_types: dict[tuple[int, int], set[str]] = {}
        ner_golden_additions: set[tuple[Any, ...]] = set()

        for start, end, label in ner_sent_gold:
            label = "NIL" if label not in ner_label_map else label
            if label == "NIL":
                continue
            span = (start, end)
            gold_span_multi_types.setdefault(span, set()).add(label)
            ner_golden_additions.add(((doc_idx, sentence_idx), span, label))

        # Count relation connectivity per label from single-label spans only.
        # Multi-label spans are excluded to avoid a circular dependency (we need
        # connectivity to choose a label, but their label isn't resolved yet).
        multi_ann_spans = {
            span for span, lbls in gold_span_multi_types.items() if len(lbls) > 1
        }
        conn_per_label: Counter[str] = Counter()
        for subj_b, subj_e, obj_b, obj_e, _ in sentence_relations:
            if (subj_b, subj_e) == (obj_b, obj_e):
                continue  # skip self-relations
            for span in ((subj_b, subj_e), (obj_b, obj_e)):
                if span not in multi_ann_spans:
                    for lbl in gold_span_multi_types.get(span, set()):
                        conn_per_label[lbl] += 1

        entity_labels_gold: dict[tuple[int, int], str] = {}
        dropped_by_label: Counter[str] = Counter()
        for span, labels in gold_span_multi_types.items():
            if len(labels) == 1:
                entity_labels_gold[span] = next(iter(labels))
                continue
            # Pick the label most connected in the sentence relation graph;
            # fall back to alphabetical order when connectivity is equal.
            chosen = sorted(labels, key=lambda lbl: (-conn_per_label[lbl], lbl))[0]
            entity_labels_gold[span] = chosen
            for dropped in labels - {chosen}:
                dropped_by_label[dropped] += 1

        return (
            entity_labels_gold,
            gold_span_multi_types,
            ner_golden_additions,
            dropped_by_label,
        )

    @staticmethod
    def _build_pos2label(
        sentence_relations: list[tuple[int, int, int, int, str]],
        label_map: dict[str, int],
        sym_labels: list[str],
        entity_labels_gold: dict[tuple[int, int], str],
        gold_span_multi_types: dict[tuple[int, int], set[str]],
        doc_idx: int,
        sentence_idx: int,
        candidate_span_set: set[tuple[int, int]],
    ) -> tuple[
        dict[tuple[int, int, int, int], int],
        set[tuple[Any, ...]],
        set[tuple[Any, ...]],
        Counter[tuple[str, str]],
        Counter[tuple[str, str, str]],
    ]:
        """Build a (subj_begin, subj_end, obj_begin, obj_end) → label_idx lookup.

        Returns a tuple of:
        - pos2label: span-pair → label index dict used during candidate construction.
        - golden_additions: set entries to merge into self.golden_labels.
        - golden_with_ner_additions: set entries for golden_labels_with_ner (RE+).
        - self_rel_counter: Counter[(span_label, rel_label)] for dropped self-relations,
          to be accumulated by the caller for a dataset-level warning.
        - missing_cand_counter: Counter[(subj_label, rel_label, obj_label)] for
          relations not trainable because a span is absent from the candidate set,
          to be accumulated by the caller for a dataset-level warning.

        For RE+ evaluation, golden_with_ner_additions is expanded to include ALL
        valid (s_type, o_type) combinations from gold_span_multi_types so that the
        model is credited for predicting any valid entity type for a doubly-annotated
        span.
        """
        pos2label: dict[tuple[int, int, int, int], int] = {}
        golden_additions: set[tuple[Any, ...]] = set()
        golden_with_ner_additions: set[tuple[Any, ...]] = set()
        self_rel_counter: Counter[tuple[str, str]] = Counter()
        missing_cand_counter: Counter[tuple[str, str, str]] = Counter()

        for (
            subj_begin,
            subj_end,
            obj_begin,
            obj_end,
            relation_label,
        ) in sentence_relations:
            relation_label = (
                "NIL" if relation_label not in label_map else relation_label
            )

            # Self-relation: same span as both subject and object
            if subj_begin == obj_begin and subj_end == obj_end:
                subj_types = gold_span_multi_types.get((subj_begin, subj_end), {"NIL"})
                span_label = min(subj_types)  # canonical label for counting
                self_rel_counter[(span_label, relation_label)] += 1
                continue

            # Track relations whose spans are absent from the candidate set
            subj_in = (subj_begin, subj_end) in candidate_span_set
            obj_in = (obj_begin, obj_end) in candidate_span_set
            if not subj_in or not obj_in:
                subj_label = entity_labels_gold.get((subj_begin, subj_end), "NIL")
                obj_label = entity_labels_gold.get((obj_begin, obj_end), "NIL")
                missing_cand_counter[(subj_label, relation_label, obj_label)] += 1

            pos2label[(subj_begin, subj_end, obj_begin, obj_end)] = label_map[
                relation_label
            ]

            # For symmetric relations, normalise to canonical direction
            # (lexicographically smaller span as subject) so that both
            # A→B and B→A annotations in the data collapse to one gold entry.
            sym_label_set = frozenset(sym_labels[1:])
            if relation_label in sym_label_set and (obj_begin, obj_end) < (
                subj_begin,
                subj_end,
            ):
                c_subj = (obj_begin, obj_end)
                c_obj = (subj_begin, subj_end)
            else:
                c_subj = (subj_begin, subj_end)
                c_obj = (obj_begin, obj_end)

            golden_additions.add(
                (
                    (doc_idx, sentence_idx),
                    c_subj,
                    c_obj,
                    relation_label,
                )
            )

            # Expand RE+ gold to all valid (s_type, o_type) combos
            s_types = gold_span_multi_types.get(
                c_subj,
                {entity_labels_gold.get(c_subj, "NIL")},
            )
            o_types = gold_span_multi_types.get(
                c_obj,
                {entity_labels_gold.get(c_obj, "NIL")},
            )
            for s_type in s_types:
                for o_type in o_types:
                    golden_with_ner_additions.add(
                        (
                            (doc_idx, sentence_idx),
                            c_subj + (s_type,),
                            c_obj + (o_type,),
                            relation_label,
                        )
                    )

        # Add reverse directions to pos2label for all (non-self) relations
        for (
            subj_begin,
            subj_end,
            obj_begin,
            obj_end,
            relation_label,
        ) in sentence_relations:
            if subj_begin == obj_begin and subj_end == obj_end:
                continue
            relation_label = (
                "NIL" if relation_label not in label_map else relation_label
            )
            reverse_key = (obj_begin, obj_end, subj_begin, subj_end)
            if reverse_key not in pos2label:
                if relation_label in sym_labels[1:]:
                    pos2label[reverse_key] = label_map[relation_label]
                else:
                    pos2label[reverse_key] = (
                        label_map[relation_label] + len(label_map) - len(sym_labels)
                    )

        return (
            pos2label,
            golden_additions,
            golden_with_ner_additions,
            self_rel_counter,
            missing_cand_counter,
        )

    def _subject_markers(self, subj_label_gold: int) -> tuple[str, str]:
        """Return the (begin_marker, end_marker) token pair for a subject entity.

        Without type markers: [unused0] / [unused1] — type-agnostic boundaries.
        With type markers: [unusedN] / [unusedM] where N/M encode the NER type,
        allowing the model to condition on the subject entity type at encoding time.
        """
        if self.use_typemarker:
            begin_idx = 2 + subj_label_gold
            end_idx = 2 + subj_label_gold + len(self.ner_label_list)
            return f"[unused{begin_idx}]", f"[unused{end_idx}]"
        return "[unused0]", "[unused1]"

    def _build_object_candidates(
        self,
        subj_begin: int,
        subj_end: int,
        ner_candidates_sent: list[tuple[int, int, str]],
        token2subword: list[int],
        doc_offset: int,
        pos2label: dict[tuple[int, int, int, int], int],
        ner_label_map: dict[str, int],
        entity_labels_gold: dict[tuple[int, int], str],
    ) -> tuple[list[ObjectEntry], list[int]]:
        """Build the list of ObjectEntry candidates for one subject.

        Converts every NER candidate in the sentence into an object candidate
        paired with the given subject.  Returns both the ObjectEntry list and
        ner_labels (gold NER label indices per object), which is later stored on
        the RelationSentence for evaluation.

        Subtoken positions of each object are adjusted to account for the two
        subject marker tokens inserted into the sequence:
        - Every object token at or after subj_begin is shifted +1.
        - Every object token strictly after subj_end is shifted +1 more.
        The same +1/+2 logic applies independently to begin and end positions.
        """
        object_candidates: list[ObjectEntry] = []
        ner_labels: list[int] = []

        for obj_begin, obj_end, obj_label in ner_candidates_sent:
            if obj_begin == len(token2subword):
                continue
            obj_label = "NIL" if obj_label not in ner_label_map else obj_label
            obj_label_gold = entity_labels_gold.get((obj_begin, obj_end), "NIL")

            obj_begin_sent = token2subword[obj_begin] - doc_offset + 1
            obj_end_sent = token2subword[obj_end + 1] - doc_offset

            if obj_begin >= subj_begin:
                obj_begin_sent += 1
                if obj_begin > subj_end:
                    obj_begin_sent += 1
            if obj_end >= subj_begin:
                obj_end_sent += 1
                if obj_begin > subj_end:
                    obj_end_sent += 1

            if obj_end_sent >= self.max_seq_length - 1:
                self._trunc_seqlen_objects += 1
                continue

            object_candidates.append(
                ObjectEntry(
                    subtoken_pos=(obj_begin_sent, obj_end_sent),
                    ner_label_idx=ner_label_map[obj_label],
                    rel_label=pos2label.get(
                        (subj_begin, subj_end, obj_begin, obj_end), 0
                    ),
                    token_pos=(obj_begin, obj_end),
                )
            )
            ner_labels.append(ner_label_map[obj_label_gold])

        return object_candidates, ner_labels

    def _select_candidate_ner(
        self, raw_doc: dict[str, Any]
    ) -> list[list[tuple[int, int, str]]]:
        """Return per-sentence NER candidate spans for one document.

        Three modes (checked in priority order):
        1. Pre-filter mode (predicted_ner_proba present): apply probability
           threshold via _select_sent_spans and convert to (start, end, label).
        2. Predicted NER mode (predicted_ner present): use pre-computed span
           predictions directly (e.g. from a first-pass NER model or the pruner).
        3. Gold mode (fallback): use gold ner annotations as candidates, giving
           an oracle upper bound for the HGERE stage.
        """
        if self.use_gold_ner:
            return raw_doc.get("ner", [[] for _ in raw_doc.get("sentences", [])])
        if self.do_pre_filter:
            assert "predicted_ner_proba" in raw_doc, (
                "pre_filter requires 'predicted_ner_proba' in the input file"
            )
            ner_candidates_all: list[list[tuple[int, int, str]]] = []
            for sent_candidates in raw_doc["predicted_ner_proba"]:
                filtered = _select_sent_spans(sent_candidates, self.pre_filter_params)
                ner_candidates_all.append(
                    [(start, end, label) for start, end, _, label in filtered]
                )
            return ner_candidates_all
        elif "predicted_ner" in raw_doc:
            return raw_doc["predicted_ner"]
        else:
            return raw_doc.get("ner", [[] for _ in raw_doc.get("sentences", [])])

    # ------------------------------------------------------------------
    # Tensor construction
    # ------------------------------------------------------------------

    def prepare_item(self, idx: int) -> dict[str, Any]:
        """Convert a preprocessed RelationSentence into the tensor dict consumed by the model.

        For each subject candidate in the sentence the method:
        - Converts the subject-marked subword token sequence to input_ids.
        - Pads input_ids to max_seq_length, then appends 2 × n_ent virtual tokens
          (one head/tail pair per object entity).  These virtual tokens allow the
          transformer to attend to each object's span boundaries independently.
        - Builds a 2-D attention mask: real tokens attend to each other; each
          virtual head/tail pair attends to real tokens and to its own pair.
        - Builds position_ids: real tokens get sequential positions; virtual tokens
          borrow the subtoken positions of their corresponding object entity so
          the model's position embeddings correctly encode entity span boundaries.

        Returns a dict with keys:
            indices, input_ids, attention_mask, position_ids, sub_positions,
            sub, rel_labels, ner_labels, obj_token_pos, n_ent, subtoken_len
        """
        sent = self.data[idx]
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[torch.Tensor] = []
        position_ids_list: list[list[int]] = []
        subject_subtoken_pos_list: list[tuple[int, int]] = []
        rel_labels_list: list[list[int]] = []
        subject_token_spans: list[tuple[int, int, str]] = []

        for entry in sent.subject_candidates:
            input_ids = self.tokenizer.convert_tokens_to_ids(
                entry.subject_marked_tokens
            )
            if self._dataset_cls_token_id is not None:
                input_ids[0] = self._dataset_cls_token_id
            input_seq_len = len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * (
                self.max_seq_length - input_seq_len
            )

            n_ent = entry.n_ent

            # Append virtual head/tail tokens for each object entity.
            # Token IDs 3/4 (or Albert equivalents) are placeholder embeddings;
            # their position_ids will be overwritten to point to the entity's span.
            if self.model_type.startswith("albert"):
                input_ids = input_ids + [30002] * n_ent + [30003] * n_ent
            else:
                input_ids = input_ids + [3] * n_ent + [4] * n_ent

            tot_seq_len = len(input_ids)
            attention_mask = torch.zeros((tot_seq_len, tot_seq_len), dtype=torch.int64)
            attention_mask[:input_seq_len, :input_seq_len] = 1

            position_ids = list(range(self.max_seq_length)) + [0] * (
                tot_seq_len - self.max_seq_length
            )

            for obj_idx in range(n_ent):
                obj_sub_head, obj_sub_tail = entry.obj_subtoken_pos[obj_idx]
                # Virtual token layout: heads at [max_seq_length, max_seq_length + n_ent),
                # tails at [max_seq_length + n_ent, max_seq_length + 2*n_ent).
                virtual_head_idx = obj_idx + self.max_seq_length
                virtual_tail_idx = virtual_head_idx + n_ent
                position_ids[virtual_head_idx] = obj_sub_head
                position_ids[virtual_tail_idx] = obj_sub_tail
                for row in [virtual_head_idx, virtual_tail_idx]:
                    for col in [virtual_head_idx, virtual_tail_idx]:
                        attention_mask[row, col] = 1
                    attention_mask[row, :input_seq_len] = 1

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            subject_subtoken_pos_list.append(entry.subject.subtoken_pos)
            rel_labels_list.append(entry.rel_labels)
            subject_token_spans.append(entry.subject.token_span)

        n_subject_candidates = sent.n_subjects
        if n_subject_candidates == 0:
            input_ids_t = torch.tensor([])
            attention_mask_t = torch.tensor([])
            position_ids_t = torch.tensor([])
            subject_subtoken_pos_t = torch.tensor([])
            rel_labels_t = torch.tensor([], dtype=torch.int64)
            ner_labels_t = torch.tensor([], dtype=torch.int64)
        else:
            input_ids_t = torch.tensor(input_ids_list)
            attention_mask_t = torch.stack(attention_mask_list)
            position_ids_t = torch.tensor(position_ids_list)
            subject_subtoken_pos_t = torch.tensor(subject_subtoken_pos_list)
            rel_labels_t = torch.tensor(rel_labels_list, dtype=torch.int64)
            ner_labels_t = torch.tensor(sent.ner_labels, dtype=torch.int64)

        item = dict(
            indices=sent.index,
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            position_ids=position_ids_t,
            sub_positions=subject_subtoken_pos_t,
            sub=subject_token_spans,
            rel_labels=rel_labels_t,
            ner_labels=ner_labels_t,
            obj_token_pos=sent.obj_token_pos,
            n_ent=n_subject_candidates,
            subtoken_len=self.max_seq_length,
        )
        if self.dataset_id is not None:
            item["dataset_id"] = self.dataset_id
        return item

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def build(
        self,
        batch_size: int,
        shuffle: bool = False,
        batch_by_size: bool = False,
        n_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Build and return a DataLoader over this dataset.

        Args:
            batch_size: Maximum total entity candidates per batch.
            shuffle: Randomly order sentences within each batch.
            batch_by_size: Sort sentences by entity count before batching to
                minimise padding (recommended for training efficiency).
            n_workers: Number of DataLoader worker processes.
            pin_memory: Pin memory for faster GPU transfer.
        """
        if batch_by_size:
            self.buckets = create_size_sorted_batches(self.sizes, batch_size)
        elif shuffle:
            self.buckets = create_shuffled_batches(self.sizes, batch_size)
        else:
            self.buckets = create_batches(self.sizes, batch_size)

        sampler_cls = (
            BucketSampler if self.local_rank == -1 else DistributedBucketSampler
        )

        def _worker_init_fn(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            random.seed(worker_seed + worker_id)
            np.random.seed(worker_seed + worker_id)

        self.loader = DataLoader(
            dataset=self,
            batch_sampler=sampler_cls(buckets=self.buckets),
            num_workers=n_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            worker_init_fn=_worker_init_fn if n_workers > 0 else None,
            persistent_workers=n_workers > 0,
            prefetch_factor=4 if n_workers > 0 else None,
        )
        return self.loader

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of sentence items into padded tensors."""
        return _collate_relation_batch(
            batch, self.tokenizer.pad_token_id, self.max_seq_length
        )
