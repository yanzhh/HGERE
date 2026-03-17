from __future__ import annotations

from copy import deepcopy
import json
import math
import random
from collections import Counter
from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from hgere.data.base_dataset import DocumentDataset
from hgere.data.samplers import (
    BucketSampler,
    DistributedBucketSampler,
    create_batches,
    create_shuffled_batches,
    create_size_sorted_batches,
)


class SentenceSubjectCandidate:
    # for each sentence with m entities, there are m subjects, objects and m**2 relations.
    # a sent_item corresponds to one subject, it records m relations related to this subject
    # a sentence contains m sent_items
    def __init__(self, id_tuple, sub_tokens, examples, sub, rel_labels):
        self.index = id_tuple  # e.g.: (0,0)
        self.sub_tokens = sub_tokens
        self.relations = examples
        # e.g.: [((2, 2, 5), 0, (0, 0)), ((17, 18, 2), 0, (10, 10)),
        #  ((25, 30, 2), 0, (17, 20))],  formation: ((left, right, ner_label),
        #  rel_label, obj_token_index_tuple);
        #  left/right: obj subtoken index; obj token index tuples: [(0,0), (10,10), (17,20)]
        self.sub = sub  # e.g: ([0, 0, 'Material'], (1, 3), 5)
        self.rel_labels = rel_labels  # e.g.: [0,0,0]
        if len(examples) > 0:
            self.build()
        else:
            self.n_ent = 0
            (
                self.subject,
                self.sub_subtoken_pos,
                self.sub_label,
                self.obj_subtoken_pos,
                self.obj_token_pos,
            ) = [], [], [], [], []

    def build(self):
        self.subject, self.sub_subtoken_pos, self.sub_label = self.sub
        self.obj_subtoken_pos = []  # e.g.: [(2, 2), (17, 18), (25, 30)]
        self.obj_token_pos = []  # e.g.: [(0, 0), (10, 10), (17, 20)]
        for entry in self.relations:
            (obj_sub_head, obj_sub_tail, _), _, (obj_head, obj_tail) = entry
            self.obj_subtoken_pos.append((obj_sub_head, obj_sub_tail))
            self.obj_token_pos.append((obj_head, obj_tail))
        self.n_ent = len(self.relations)  # e.g.: 3

    def __repr__(self):
        if self.n_ent > 0:
            repr_str = f"<{self.n_ent} relations; sub: {self.subject}; objs: obj_subtokens-{self.obj_subtoken_pos}, obj_tokens-{self.obj_token_pos}>"
        else:
            repr_str = "<0 relations; ...>"
        return repr_str


class Sentence:
    def __init__(self, id_tuple, item_list, ner_labels, words):
        self.index = id_tuple
        self.items = item_list
        self.n_items = len(item_list)
        self.ner_labels = ner_labels
        self.words = words
        if self.n_items == 0:
            self.obj_token_pos = []
        else:
            self.obj_token_pos = self.items[0].obj_token_pos

    @property
    # number of entities
    def size(self):
        return len(self.items)


class Sampler(torch.utils.data.Sampler):
    def __init__(
        self,
        buckets: List,
        shuffle: bool = False,
    ):
        self.shuffle = shuffle
        self.buckets = buckets
        self.n_samples = len(buckets)

        # self.epoch = 1

    def __iter__(self):
        indices = [self.buckets[i] for i in torch.arange(self.n_samples).tolist()]
        return iter(indices)

    def __len__(self):
        return self.n_samples

    # def set_epoch(self, epoch: int) -> None:
    #     self.epoch = epoch


class DistrSampler(DistributedSampler):
    def __init__(
        self,
        buckets: List,
        shuffle: bool = False,
    ):
        self.n_samples = len(buckets)
        # super().__init__(self, num_replicas=None,rank=None, drop_last=False)
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                f"Invalid rank {self.rank}, rank should be in the interval"
                " [0, {self.num_replicas - 1}]"
            )
        self.drop_last = False
        if self.drop_last and len(self) % self.num_replicas != 0:
            # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        self.shuffle = shuffle
        self.buckets = buckets

    def __iter__(self):
        # batches = []
        # for i, bucket in enumerate(self.buckets):
        #     batches.append([bucket[j] for j in range_fn(len(bucket)).tolist()])
        # self.epoch += 1

        indices = [self.buckets[i] for i in torch.arange(self.n_samples).tolist()]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.n_samples


class RelationDataset(DocumentDataset):
    """HGERE relation-extraction dataset.

    One dataset item = one sentence (all subject/object pairs for that sentence).
    Inherits shared tokenization and context-window infrastructure from DocumentDataset.

    Constructor signature is identical to the original RelationDataset to preserve
    backward compatibility with run_hgnn.py, infer_fixed_spans.py, and
    infer_pruner_augmented.py.
    """

    def __init__(
        self,
        logger: Any,
        tokenizer: Any,
        labels: Any,
        file_path: str,
        args: Any = None,
        max_pair_length: int | None = None,
        max_ents: int = 18,
        doc_limit: int | None = None,
        preload: bool = False,
    ) -> None:
        self.logger = logger
        self.max_pair_length = max_pair_length
        self.max_entity_length = (max_pair_length or 0) * 2
        self.max_ents = max_ents
        self.args = args

        self.use_typemarker = args.use_typemarker if args else False
        self.model_type = args.model_type if args else "bert"
        self.no_sym = args.no_sym if args else True
        self.nocross = getattr(args, "nocross", False)
        self.local_rank = getattr(args, "local_rank", -1)

        self.ner_label_list = labels.ner
        self.sym_labels = labels.rel.symmetric(only_nil=self.no_sym)
        self.label_list = labels.rel.all

        # Populated by _build_index()
        self.data: list[Any] = []
        self.sizes: list[int] = []
        self.ner_golden_labels: set[tuple[Any, ...]] = set()
        self.golden_labels: set[tuple[Any, ...]] = set()
        self.golden_labels_with_ner: set[tuple[Any, ...]] = set()
        self.global_predicted_ners: dict[tuple[int, int], list[Any]] = {}
        self.ner_tot_recall: int = 0
        self.tot_recall: int = 0

        # Base __init__ indexes the file (byte offsets)
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length if args else 512,
            lazy=False,
            doc_limit=doc_limit,
        )
        self._build_index()

        self.is_preloaded = preload
        if preload:
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
        self._data_tokenized = []
        for idx in range(len(self.data)):
            self._data_tokenized.append(self.prepare_item(idx))

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Replaces initialize(). Uses DocumentDataset helpers for tokenization
        and context-window computation."""
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}
        max_num_subwords = self.max_seq_length - 4  # for two type markers

        for doc_idx in range(len(self._offsets)):
            doc = self._load_raw_document(doc_idx)
            tok_idx = self._tokenize_document(doc)
            token2subword = tok_idx.token2subword
            subword_sentence_boundaries = tok_idx.subword_sentence_boundaries

            # Load raw JSON for predicted_ner (not in frozen domain objects)
            with open(self._file_path, "rb") as f:
                f.seek(self._offsets[doc_idx])
                raw = json.loads(f.readline())

            if "predicted_ner" in raw:
                ner_candidates_all = raw["predicted_ner"]
            else:
                ner_candidates_all = raw["ner"]
            ner_gold_all = raw["ner"]
            relations_all = raw["relations"]

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

                entity_labels_gold: dict[tuple[int, int], str] = {}
                for start, end, label in ner_sent_gold:
                    label = "NIL" if label not in ner_label_map else label
                    if label != "NIL":
                        entity_labels_gold[(start, end)] = label
                        self.ner_golden_labels.add(
                            ((doc_idx, sentence_idx), (start, end), label)
                        )

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

                # Build pos2label for relation labels
                pos2label: dict[tuple[int, int, int, int], int] = {}
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
                    pos_key = (subj_begin, subj_end, obj_begin, obj_end)
                    pos2label[pos_key] = label_map[relation_label]
                    self.golden_labels.add(
                        (
                            (doc_idx, sentence_idx),
                            (subj_begin, subj_end),
                            (obj_begin, obj_end),
                            relation_label,
                        )
                    )
                    self.golden_labels_with_ner.add(
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
                    if relation_label in self.sym_labels[1:]:
                        self.golden_labels.add(
                            (
                                (doc_idx, sentence_idx),
                                (obj_begin, obj_end),
                                (subj_begin, subj_end),
                                relation_label,
                            )
                        )

                # Add reverse relations to pos2label
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
                    rel_pos_key = (obj_begin, obj_end, subj_begin, subj_end)
                    if rel_pos_key not in pos2label:
                        if relation_label in self.sym_labels[1:]:
                            pos2label[rel_pos_key] = label_map[relation_label]
                        else:
                            pos2label[rel_pos_key] = (
                                label_map[relation_label]
                                + len(label_map)
                                - len(self.sym_labels)
                            )

                subject_candidates = []
                for subj_begin, subj_end, subj_label in ner_candidates_sent:
                    if subj_begin == len(token2subword):
                        continue

                    subj_begin_sent = token2subword[subj_begin] - doc_offset + 1
                    subj_end_sent = token2subword[subj_end + 1] - doc_offset

                    subj_label_gold = ner_label_map.get(
                        entity_labels_gold.get((subj_begin, subj_end), "NIL"), 0
                    )

                    if self.use_typemarker:
                        type_marker_begin_idx = 2 + subj_label_gold
                        subj_marker_begin = f"[unused{type_marker_begin_idx}]"
                        type_marker_end_idx = (
                            2 + subj_label_gold + len(self.ner_label_list)
                        )
                        subj_marker_end = f"[unused{type_marker_end_idx}]"
                    else:
                        subj_marker_begin = "[unused0]"
                        subj_marker_end = "[unused1]"

                    token_before_subj = target_tokens[:subj_begin_sent]
                    token_subj = target_tokens[subj_begin_sent : subj_end_sent + 1]
                    token_after_subj = target_tokens[subj_end_sent + 1 :]
                    sub_tokens = (
                        token_before_subj
                        + [subj_marker_begin]
                        + token_subj
                        + [subj_marker_end]
                        + token_after_subj
                    )
                    subj_end_sent += 2

                    if subj_end_sent >= self.max_seq_length - 1:
                        continue

                    object_candidates = []
                    rel_labels: list[int] = []
                    ner_labels: list[int] = []

                    for obj_begin, obj_end, obj_label in ner_candidates_sent:
                        if obj_begin == len(token2subword):
                            continue
                        obj_label = (
                            "NIL" if obj_label not in ner_label_map else obj_label
                        )
                        obj_label_gold = entity_labels_gold.get(
                            (obj_begin, obj_end), "NIL"
                        )

                        obj_begin_doc = token2subword[obj_begin]
                        obj_end_doc = token2subword[obj_end + 1]
                        obj_begin_sent = obj_begin_doc - doc_offset + 1
                        obj_end_sent = obj_end_doc - doc_offset

                        if obj_begin >= subj_begin:
                            obj_begin_sent += 1
                            if obj_begin > subj_end:
                                obj_begin_sent += 1
                        if obj_end >= subj_begin:
                            obj_end_sent += 1
                            if obj_begin > subj_end:
                                obj_end_sent += 1

                        relation_label = pos2label.get(
                            (subj_begin, subj_end, obj_begin, obj_end), 0
                        )

                        if obj_end_sent >= self.max_seq_length - 1:
                            continue

                        obj_label_idx = ner_label_map[obj_label]
                        obj_in_sentence = (obj_begin_sent, obj_end_sent, obj_label_idx)
                        object_candidates.append(
                            (obj_in_sentence, relation_label, (obj_begin, obj_end))
                        )
                        rel_labels.append(relation_label)
                        ner_labels.append(ner_label_map[obj_label_gold])

                    subject_candidate = SentenceSubjectCandidate(
                        id_tuple=(doc_idx, sentence_idx),
                        sub_tokens=sub_tokens,
                        examples=object_candidates,
                        sub=(
                            (subj_begin, subj_end, subj_label),
                            (subj_begin_sent, subj_end_sent),
                            subj_label_gold,
                        ),
                        rel_labels=rel_labels,
                    )
                    subject_candidates.append(subject_candidate)

                if len(subject_candidates) == 0:
                    ner_labels_sent: list[int] = []
                else:
                    ner_labels_sent = ner_labels  # type: ignore[assignment]

                new_sent = Sentence(
                    id_tuple=(doc_idx, sentence_idx),
                    item_list=subject_candidates,
                    ner_labels=ner_labels_sent,
                    words=target_tokens,
                )
                self.data.append(new_sent)
                self.sizes.append(new_sent.size)

    # ------------------------------------------------------------------
    # Tensor construction (unchanged logic from original prepare_item)
    # ------------------------------------------------------------------

    def prepare_item(self, idx: int) -> dict[str, Any]:
        """Convert a preprocessed Sentence item to tensor dict."""
        sent = self.data[idx]
        sent_index = sent.index
        ner_labels = sent.ner_labels
        obj_token_pos = sent.obj_token_pos
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[Any] = []
        position_ids_list: list[list[int]] = []
        sub_subtoken_pos_list: list[tuple[int, int]] = []
        rel_labels_list: list[list[int]] = []
        subs: list[Any] = []

        for entry in sent.items:
            rel_labels = entry.rel_labels
            sub_subtoken_pos = entry.sub_subtoken_pos
            input_tokens = entry.sub_tokens

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            L = len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * (
                self.max_seq_length - len(input_ids)
            )

            n_ent = entry.n_ent

            if self.model_type.startswith("albert"):
                input_ids = input_ids + [30002] * n_ent + [30003] * n_ent
            else:
                input_ids = input_ids + [3] * n_ent + [4] * n_ent

            tot_seq_len = len(input_ids)
            attention_mask = torch.zeros((tot_seq_len, tot_seq_len), dtype=torch.int64)
            attention_mask[:L, :L] = 1

            obj_subtoken_pos = entry.obj_subtoken_pos
            position_ids = list(range(self.max_seq_length)) + [0] * (
                tot_seq_len - self.max_seq_length
            )
            num_pair = n_ent

            for x_idx in range(n_ent):
                obj_sub_head, obj_sub_tail = obj_subtoken_pos[x_idx]
                w1 = x_idx + self.max_seq_length
                w2 = w1 + num_pair
                position_ids[w1] = obj_sub_head
                position_ids[w2] = obj_sub_tail
                for xx in [w1, w2]:
                    for yy in [w1, w2]:
                        attention_mask[xx, yy] = 1
                    attention_mask[xx, :L] = 1

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            sub_subtoken_pos_list.append(sub_subtoken_pos)
            rel_labels_list.append(rel_labels)
            subs.append(entry.subject)

        n_ent = sent.size
        if n_ent == 0:
            input_ids_t = torch.tensor([])
            attention_mask_t = torch.tensor([])
            position_ids_t = torch.tensor([])
            sub_subtoken_pos_t = torch.tensor([])
            rel_labels_t = torch.tensor([], dtype=torch.int64)
            ner_labels_t = torch.tensor([], dtype=torch.int64)
        else:
            input_ids_t = torch.tensor(input_ids_list)
            attention_mask_t = torch.stack(attention_mask_list)
            position_ids_t = torch.tensor(position_ids_list)
            sub_subtoken_pos_t = torch.tensor(sub_subtoken_pos_list)
            rel_labels_t = torch.tensor(rel_labels_list, dtype=torch.int64)
            ner_labels_t = torch.tensor(ner_labels, dtype=torch.int64)

        return dict(
            indexs=sent_index,
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            position_ids=position_ids_t,
            sub_positions=sub_subtoken_pos_t,
            sub=subs,
            rel_labels=rel_labels_t,
            ner_labels=ner_labels_t,
            obj_token_pos=obj_token_pos,
            n_ent=n_ent,
            subtoken_len=self.max_seq_length,
        )

    # ------------------------------------------------------------------
    # DataLoader construction (backward compat with run_hgnn.py)
    # ------------------------------------------------------------------

    def build(
        self,
        batch_size: int,
        shuffle: bool = False,
        batch_by_size: bool = False,
        n_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
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
        """Collate a batch of relation items. Preserved for backward compat."""
        return _collate_relation_batch(
            batch, self.tokenizer.pad_token_id, self.max_seq_length
        )


def _collate_relation_batch(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    max_seq_length: int,
) -> dict[str, Any]:
    """Standalone collate function, also used by RelationCollator."""
    ent_numbers = torch.tensor([sent["n_ent"] for sent in batch])
    n_sent = len(batch)

    if len(ent_numbers) == 1:
        items = deepcopy(batch[0])
        items.pop("n_ent")
        items.pop("subtoken_len")
        items["ent_numbers"] = ent_numbers
        for k in ["rel_labels", "ner_labels"]:
            items[k] = items[k].unsqueeze(0)
        for k in ["indexs", "sub", "obj_token_pos"]:
            items[k] = [items[k]]
        return items

    subs: list[Any] = []
    obj_token_pos: list[Any] = []
    indexs: list[Any] = []
    max_ent_num = int(max(ent_numbers))
    seqlen = batch[0]["subtoken_len"]
    bs = int(sum(ent_numbers))

    batch_input_ids = (
        torch.zeros((bs, seqlen + 2 * max_ent_num), dtype=torch.int) + pad_token_id
    )
    batch_pos_ids = torch.zeros((bs, seqlen + 2 * max_ent_num), dtype=torch.int)
    batch_sub_subtoken_pos = torch.zeros((bs, 2), dtype=torch.long)
    batch_attn_mask = torch.zeros(
        (bs, seqlen + 2 * max_ent_num, seqlen + 2 * max_ent_num), dtype=torch.int
    )
    batch_rel_labels = (
        torch.zeros((n_sent, max_ent_num, max_ent_num), dtype=torch.int64) - 1
    )
    batch_ner_labels = torch.zeros((n_sent, max_ent_num), dtype=torch.int64) - 1

    n_ent_0 = 0
    for i, items_i in enumerate(batch):
        indexs.append(items_i["indexs"])
        n_ent = int(ent_numbers[i])
        if n_ent > 0:
            n_ent_1 = n_ent_0 + n_ent
            input_ids_split = items_i["input_ids"].split([seqlen, n_ent, n_ent], dim=-1)
            batch_input_ids[n_ent_0:n_ent_1, :seqlen] = input_ids_split[0]
            for j in range(2):
                batch_input_ids[
                    n_ent_0:n_ent_1,
                    seqlen + j * max_ent_num : seqlen + j * max_ent_num + n_ent,
                ] = input_ids_split[j + 1]

            pos_ids_split = items_i["position_ids"].split(
                [seqlen, n_ent, n_ent], dim=-1
            )
            batch_pos_ids[n_ent_0:n_ent_1, :seqlen] = pos_ids_split[0]
            for j in range(2):
                batch_pos_ids[
                    n_ent_0:n_ent_1,
                    seqlen + j * max_ent_num : seqlen + j * max_ent_num + n_ent,
                ] = pos_ids_split[j + 1]

            batch_sub_subtoken_pos[n_ent_0:n_ent_1, :] = items_i["sub_positions"]

            attn_mask_i = items_i["attention_mask"]
            batch_attn_mask[n_ent_0:n_ent_1, :seqlen, :seqlen] = attn_mask_i[
                :, :seqlen, :seqlen
            ]
            for j in range(2):
                batch_attn_mask[
                    n_ent_0:n_ent_1,
                    seqlen + j * max_ent_num : seqlen + n_ent + j * max_ent_num,
                    :seqlen,
                ] = attn_mask_i[
                    :, seqlen + j * n_ent : seqlen + (j + 1) * n_ent, :seqlen
                ]
                for k in range(2):
                    batch_attn_mask[
                        n_ent_0:n_ent_1,
                        seqlen + j * max_ent_num : seqlen + n_ent + j * max_ent_num,
                        seqlen + k * max_ent_num : seqlen + n_ent + k * max_ent_num,
                    ] = attn_mask_i[
                        :,
                        seqlen + j * n_ent : seqlen + (j + 1) * n_ent,
                        seqlen + k * n_ent : seqlen + (k + 1) * n_ent,
                    ]

            batch_rel_labels[i, :n_ent, :n_ent] = items_i["rel_labels"]
            batch_ner_labels[i, :n_ent] = items_i["ner_labels"]
            subs.append(items_i["sub"])
            obj_token_pos.append(items_i["obj_token_pos"])
            n_ent_0 = n_ent_1
        else:
            obj_token_pos.append([])
            subs.append([])

    return dict(
        indexs=indexs,
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
        position_ids=batch_pos_ids,
        sub_positions=batch_sub_subtoken_pos,
        rel_labels=batch_rel_labels,
        ner_labels=batch_ner_labels,
        sub=subs,
        obj_token_pos=obj_token_pos,
        ent_numbers=ent_numbers,
    )


def _create_batches(n_ents_by_sent, batch_size):
    sort_index = np.arange(len(n_ents_by_sent))
    sorted_sizes = n_ents_by_sent
    batch_sizes = []
    batches = []
    missing_in_batches = []

    current_sizes = []
    current_batch = []
    for sent_idx, n_sent_ents in enumerate(sorted_sizes):
        if (sum(current_sizes) + n_sent_ents <= batch_size) or (
            len(current_sizes) == -1 and n_sent_ents >= batch_size
        ):
            # current batch has room for more sentences
            current_sizes.append(n_sent_ents)
            current_batch.append(sort_index[sent_idx])
        else:
            # current batch is full, open new batch
            missing = batch_size - sum(current_sizes)
            missing_in_batches.append(missing)
            batch_sizes.append(current_sizes)
            batches.append(current_batch)
            current_sizes = [n_sent_ents]
            current_batch = [sort_index[sent_idx]]
    if len(current_sizes) > -1:
        batch_sizes.append(current_sizes)
        batches.append(current_batch)
    # print("missing in batches:", sum(missing_in_batches))
    return batches


def _create_size_sorted_batches(n_ents_by_sent, batch_size):
    """Sort sentences by entity count, pack consecutive (similar-size) sentences into
    batches, then shuffle batch order.  Within each batch all sentences have similar
    entity counts, minimising max_ent_num padding and making hypergraph iteration
    uniform across the batch."""
    ordered = sorted(enumerate(n_ents_by_sent), key=lambda x: x[1])
    batches = []
    current_batch = []
    current_total = 0
    for idx, size in ordered:
        size = max(size, 1)
        if current_total + size > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_total = size
        else:
            current_batch.append(idx)
            current_total += size
    if current_batch:
        batches.append(current_batch)
    random.shuffle(batches)
    return batches


def _create_shuffled_batches(n_ents_by_sent, batch_size):
    ordered = [(idx, max(size, 1)) for idx, size in enumerate(n_ents_by_sent)]
    random.shuffle(ordered)
    ordered.sort(key=lambda x: x[1])
    batches = []
    sizes = []
    while ordered:
        batch, size = _get_batch_filled_up_smallest(ordered, batch_size)
        if batch is not None:
            batches.append(batch)
            sizes.append(size)
    print("Reals batch_size counts:", Counter(sizes).most_common())
    random.shuffle(batches)
    return batches


def _get_batch_filled_up_smallest(ordered, batch_size):
    current_batch = None
    current_size = None
    if ordered:
        sent_id, size = ordered.pop()
        current_batch = [sent_id]
        current_size = size
    while ordered:
        next_sent_id, next_size = ordered[0]
        if current_size + next_size > batch_size:
            return current_batch, current_size
        else:
            sent_id, size = ordered.pop(0)
            current_batch.append(sent_id)
            current_size += size
            if len(ordered) == 0:
                return current_batch, current_size
    return current_batch, current_size
