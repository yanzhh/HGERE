from __future__ import annotations

import json
from typing import Any

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
)
from ..data.samplers import (
    BucketSampler,
    DistributedBucketSampler,
    create_batches,
    create_shuffled_batches,
    create_size_sorted_batches,
)
from ..span_classifier.neural.evaluate import _select_sent_spans


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
        self.max_pair_length = params.max_pair_length
        self.max_entity_length = params.max_pair_length * 2
        self.max_ents = params.max_ents

        self.use_typemarker = params.use_typemarker
        self.model_type = params.model_type
        self.no_sym = params.no_sym
        self.nocross = params.nocross
        self.local_rank = params.local_rank

        self.ner_label_list: list[str] = labels.ner
        self.sym_labels: list[str] = labels.rel.symmetric(only_nil=self.no_sym)
        self.label_list: list[str] = labels.rel.all

        # Populated by _build_index()
        self.data: list[RelationSentence] = []
        self.sizes: list[int] = []
        self.ner_golden_labels: set[tuple[Any, ...]] = set()
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

            n_sents = len(doc.sentences)
            ner_gold_all = raw.get("ner", [[] for _ in range(n_sents)])
            relations_all = raw.get("relations", [[] for _ in range(n_sents)])

            for sent_rel in relations_all:
                self.tot_recall += len(sent_rel)

            all_subwords: list[str] = tok_idx.subword_tokens

            for sentence_idx, sent in enumerate(doc.sentences):
                ner_candidates_sent = list(ner_candidates_all[sentence_idx])
                if self.max_ents is not None:
                    ner_candidates_sent = ner_candidates_sent[: self.max_ents]
                sentence_relations = relations_all[sentence_idx]
                ner_sent_gold = ner_gold_all[sentence_idx]

                self.ner_tot_recall += len(ner_sent_gold)

                entity_labels_gold, ner_golden_new = self._collect_gold_entity_labels(
                    ner_sent_gold=ner_sent_gold,
                    ner_label_map=ner_label_map,
                    doc_idx=doc_idx,
                    sentence_idx=sentence_idx,
                )
                self.ner_golden_labels.update(ner_golden_new)

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

                pos2label, golden_new, golden_with_ner_new = self._build_pos2label(
                    sentence_relations=sentence_relations,
                    label_map=label_map,
                    sym_labels=self.sym_labels,
                    entity_labels_gold=entity_labels_gold,
                    doc_idx=doc_idx,
                    sentence_idx=sentence_idx,
                )
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

        self.candidate_stats = self._compute_candidate_stats(
            self.global_predicted_ners, self.ner_golden_labels
        )
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
    ) -> tuple[dict[tuple[int, int], str], set[tuple[Any, ...]]]:
        """Build a (token_begin, token_end) → label map for gold entities in one sentence.

        Returns a tuple of:
        - entity_labels_gold: span → label dict for non-NIL gold entities.
        - ner_golden_additions: set entries to merge into self.ner_golden_labels
          for final NER evaluation.

        Labels not in the NER label set are mapped to NIL and excluded from the dict.
        """
        entity_labels_gold: dict[tuple[int, int], str] = {}
        ner_golden_additions: set[tuple[Any, ...]] = set()
        for start, end, label in ner_sent_gold:
            label = "NIL" if label not in ner_label_map else label
            if label != "NIL":
                entity_labels_gold[(start, end)] = label
                ner_golden_additions.add(((doc_idx, sentence_idx), (start, end), label))
        return entity_labels_gold, ner_golden_additions

    @staticmethod
    def _build_pos2label(
        sentence_relations: list[tuple[int, int, int, int, str]],
        label_map: dict[str, int],
        sym_labels: list[str],
        entity_labels_gold: dict[tuple[int, int], str],
        doc_idx: int,
        sentence_idx: int,
    ) -> tuple[
        dict[tuple[int, int, int, int], int],
        set[tuple[Any, ...]],
        set[tuple[Any, ...]],
    ]:
        """Build a (subj_begin, subj_end, obj_begin, obj_end) → label_idx lookup.

        Returns a tuple of:
        - pos2label: span-pair → label index dict used during candidate construction.
        - golden_additions: set entries to merge into self.golden_labels.
        - golden_with_ner_additions: set entries to merge into self.golden_labels_with_ner.

        For symmetric relations, adds the reverse direction with the same label.
        For asymmetric relations, the reverse direction is added with an index
        offset so the model can score (A→B) and (B→A) independently:
            reverse_idx = original_idx + n_labels - n_sym_labels
        """
        pos2label: dict[tuple[int, int, int, int], int] = {}
        golden_additions: set[tuple[Any, ...]] = set()
        golden_with_ner_additions: set[tuple[Any, ...]] = set()

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
            pos2label[(subj_begin, subj_end, obj_begin, obj_end)] = label_map[
                relation_label
            ]
            golden_additions.add(
                (
                    (doc_idx, sentence_idx),
                    (subj_begin, subj_end),
                    (obj_begin, obj_end),
                    relation_label,
                )
            )
            golden_with_ner_additions.add(
                (
                    (doc_idx, sentence_idx),
                    (
                        subj_begin,
                        subj_end,
                        entity_labels_gold.get((subj_begin, subj_end), "NIL"),
                    ),
                    (
                        obj_begin,
                        obj_end,
                        entity_labels_gold.get((obj_begin, obj_end), "NIL"),
                    ),
                    relation_label,
                )
            )
            if relation_label in sym_labels[1:]:
                # Symmetric: reverse direction carries the same label.
                golden_additions.add(
                    (
                        (doc_idx, sentence_idx),
                        (obj_begin, obj_end),
                        (subj_begin, subj_end),
                        relation_label,
                    )
                )

        # Add reverse directions to pos2label for all relations so the model
        # receives a training signal for both (A→B) and (B→A).
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
            reverse_key = (obj_begin, obj_end, subj_begin, subj_end)
            if reverse_key not in pos2label:
                if relation_label in sym_labels[1:]:
                    pos2label[reverse_key] = label_map[relation_label]
                else:
                    pos2label[reverse_key] = (
                        label_map[relation_label] + len(label_map) - len(sym_labels)
                    )

        return pos2label, golden_additions, golden_with_ner_additions

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

        return dict(
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
        self.loader = DataLoader(
            dataset=self,
            batch_sampler=sampler_cls(buckets=self.buckets),
            num_workers=n_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
        )
        return self.loader

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of sentence items into padded tensors."""
        return _collate_relation_batch(
            batch, self.tokenizer.pad_token_id, self.max_seq_length
        )
