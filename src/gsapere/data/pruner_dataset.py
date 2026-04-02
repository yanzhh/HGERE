"""
pruner_dataset.py — Binary entity-pruner dataset.

Replaces ACEDatasetNER from pruner_data_utils.py.

One dataset item = one chunk of span candidates for one sentence window.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gsapere.data.base_dataset import DocumentDataset
from gsapere.labels import load_label_scheme
from gsapere.pre_filter import RuleBasedPruner


class PrunerDataset(DocumentDataset):
    """Binary entity-pruner dataset.

    Enumerates all valid n-gram span candidates for each sentence, assigns
    binary labels (0=NIL, >0=entity class), and chunks them by max_pair_length.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Any,
        max_seq_length: int,
        max_pair_length: int,
        max_mention_ori_length: int,
        model_type: str,
        label_set: str,
        evaluate: bool = False,
        rulebased_pruner_file: str | None = None,
        shuffle: bool = False,
        group_sort: bool = False,
        group_edge: bool = False,
        group_axis: int = -1,
        nocross: bool = False,
        doc_limit: int | None = None,
    ) -> None:
        self.max_pair_length = max_pair_length
        self.max_entity_length = max_pair_length * 2
        self.max_mention_ori_length = max_mention_ori_length
        self.model_type = model_type
        self.evaluate = evaluate
        self.shuffle = shuffle
        self.group_sort = group_sort
        self.group_edge = group_edge
        self.group_axis = group_axis
        self.nocross = nocross

        self.ner_label_list = load_label_scheme(label_set).ner

        if rulebased_pruner_file:
            self.rulebased_pruner: RuleBasedPruner | None = RuleBasedPruner.load(
                rulebased_pruner_file
            )
            if self.rulebased_pruner.max_span_len is not None:
                self.max_mention_ori_length = self.rulebased_pruner.max_span_len
        else:
            self.rulebased_pruner = None

        # Populated by _build_index()
        self.data: list[dict[str, Any]] = []
        self.tot_recall: int = 0
        self.n_gold_in_candidates: int = 0
        self.ner_golden_labels: set[tuple[Any, ...]] = set()

        # Call base __init__ which runs _index_file(); then we run _build_index()
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            lazy=False,
            doc_limit=doc_limit,
        )
        self._build_index()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> list[Any]:
        return self._item_to_tensors(self.data[idx])

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Iterate all documents, tokenize, enumerate span candidates."""
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}
        max_num_subwords = self.max_seq_length - 2  # for CLS + SEP

        for line_idx in range(len(self._offsets)):
            doc = self._load_raw_document(line_idx)
            idx = self._tokenize_document(doc)
            token2subword = idx.token2subword
            subword2token = idx.subword2token
            subword_sentence_boundaries = idx.subword_sentence_boundaries
            subword_start_positions = idx.subword_start_positions

            # Flatten tokens for rulebased pruner context lookup
            all_tokens: list[str] = []
            sentence_boundaries_tok: list[int] = [0]
            for sent in doc.sentences:
                all_tokens.extend(sent.tokens)
                sentence_boundaries_tok.append(len(all_tokens))

            all_subwords: list[str] = idx.subword_tokens

            for sentence_idx, sent in enumerate(doc.sentences):
                sentence_ners = sent.ner
                self.tot_recall += len(sentence_ners)

                entity_labels: dict[tuple[int, int], int] = {}
                for span in sentence_ners:
                    key = token2subword[span.start], token2subword[span.end + 1]
                    entity_labels[key] = ner_label_map.get(span.label, 0)
                    self.ner_golden_labels.add(
                        ((line_idx, sentence_idx), (span.start, span.end), span.label)
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
                sentence_length = cw.sentence_length
                left_context_length = cw.left_context_length
                doc_offset = cw.doc_offset

                target_tokens = (
                    [self.tokenizer.cls_token]
                    + cw.target_tokens
                    + [self.tokenizer.sep_token]
                )

                entity_infos: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
                sentence_begin_in_context = left_context_length
                sentence_end_in_context = left_context_length + sentence_length

                for context_entity_start in range(
                    sentence_begin_in_context, sentence_end_in_context
                ):
                    doc_entity_start = doc_offset + context_entity_start
                    if doc_entity_start not in subword_start_positions:
                        continue
                    for context_entity_end in range(
                        context_entity_start + 1,
                        sentence_end_in_context + 1,
                    ):
                        doc_entity_end = doc_offset + context_entity_end
                        if doc_entity_end not in subword_start_positions:
                            continue
                        token_start = subword2token[doc_entity_start]
                        token_end = subword2token[doc_entity_end - 1]
                        if token_end - token_start + 1 > self.max_mention_ori_length:
                            break

                        if self.rulebased_pruner is not None:
                            sent_tok_start = sentence_boundaries_tok[sentence_idx]
                            sent_tok_end = sentence_boundaries_tok[sentence_idx + 1]
                            span_words = all_tokens[token_start : token_end + 1]
                            ctx_before = all_tokens[sent_tok_start:token_start]
                            ctx_after = all_tokens[token_end + 1 : sent_tok_end]
                            if self.rulebased_pruner.should_prune(
                                span_words, ctx_before, ctx_after
                            ):
                                continue

                        label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                        entity_labels.pop((doc_entity_start, doc_entity_end), None)
                        entity_infos.append(
                            (
                                (context_entity_start + 1, context_entity_end),
                                label,
                                (token_start, token_end),
                            )
                        )
                        if label > 0:
                            self.n_gold_in_candidates += 1

                dL = self.max_pair_length
                if self.shuffle:
                    random.shuffle(entity_infos)
                if self.group_sort:
                    group_axis = np.random.randint(2)
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(
                        key=lambda x: (x[0][group_axis], x[0][1 - group_axis]),
                        reverse=sort_dir,
                    )

                sent_len = len(sent.tokens)

                if not self.group_edge:
                    for i in range(0, max(len(entity_infos), 1), dL):
                        examples = entity_infos[i : i + dL]
                        item: dict[str, Any] = {
                            "sentence": target_tokens,
                            "examples": examples,
                            "example_index": (line_idx, sentence_idx),
                            "example_L": len(entity_infos),
                            "sent_length": sent_len,
                        }
                        self.data.append(item)
                else:
                    ga = (
                        self.group_axis
                        if self.group_axis != -1
                        else np.random.randint(2)
                    )
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(
                        key=lambda x: (x[0][ga], x[0][1 - ga]),
                        reverse=sort_dir,
                    )
                    _start = 0
                    if len(entity_infos) == 0:
                        self.data.append(
                            {
                                "sentence": target_tokens,
                                "examples": [],
                                "example_index": (line_idx, sentence_idx),
                                "example_L": 0,
                                "sent_length": sent_len,
                            }
                        )
                    while _start < len(entity_infos):
                        _end = _start + dL
                        if _end >= len(entity_infos):
                            _end = len(entity_infos)
                        else:
                            while (
                                entity_infos[_end - 1][0][ga]
                                == entity_infos[_end][0][ga]
                                and _end > _start
                            ):
                                _end -= 1
                            if _start == _end:
                                _end = _start + dL
                        examples = entity_infos[_start:_end]
                        self.data.append(
                            {
                                "sentence": target_tokens,
                                "examples": examples,
                                "example_index": (line_idx, sentence_idx),
                                "example_L": len(entity_infos),
                                "sent_length": sent_len,
                            }
                        )
                        _start = _end

    # ------------------------------------------------------------------
    # Tensor construction (ported from ACEDatasetNER.__getitem__)
    # ------------------------------------------------------------------

    def _item_to_tensors(self, entry: dict[str, Any]) -> list[Any]:
        """Convert a preprocessed entry dict to a list of tensors + metadata."""
        input_ids = self.tokenizer.convert_tokens_to_ids(entry["sentence"])
        L = len(input_ids)

        if not self.model_type.startswith("modernbert"):
            input_ids += [0] * (self.max_seq_length - len(input_ids))

        position_plus_pad = int(self.model_type.find("roberta") != -1) * 2

        if self.model_type not in ["bertspan", "robertaspan", "albertspan"]:
            if self.model_type.startswith("albert"):
                input_ids = (
                    input_ids
                    + [30000] * len(entry["examples"])
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                    + [30001] * len(entry["examples"])
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )
            elif self.model_type.startswith("modernbert"):
                unused0_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
                unused1_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
                n_active = len(entry["examples"])
                input_ids = (
                    input_ids
                    + [unused0_id] * n_active
                    + [0] * (self.max_pair_length - n_active)
                    + [unused1_id] * n_active
                    + [0] * (self.max_pair_length - n_active)
                    + [0] * (self.max_seq_length - L)
                )
            elif self.model_type.startswith("roberta"):
                input_ids = (
                    input_ids
                    + [50261] * len(entry["examples"])
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                    + [50262] * len(entry["examples"])
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )
            else:  # BERT
                input_ids = (
                    input_ids
                    + [1] * len(entry["examples"])
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                    + [2] * len(entry["examples"])
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )

            attention_mask = torch.zeros(
                (self.max_entity_length + self.max_seq_length,) * 2,
                dtype=torch.int64,
            )
            attention_mask[:L, :L] = 1
            attention_mask.fill_diagonal_(1)

            if self.model_type.startswith("modernbert"):
                position_ids = list(range(L)) + [0] * (
                    self.max_seq_length + self.max_entity_length - L
                )
                marker_offset = L
            else:
                position_ids = (
                    list(
                        range(
                            position_plus_pad, position_plus_pad + self.max_seq_length
                        )
                    )
                    + [0] * self.max_entity_length
                )
                marker_offset = self.max_seq_length

        else:
            attention_mask_list = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.int64)
            position_ids = (
                list(range(position_plus_pad, position_plus_pad + self.max_seq_length))
                + [0] * self.max_entity_length
            )
            marker_offset = self.max_seq_length

        labels: list[int] = []
        mentions: list[tuple[int, int]] = []
        mention_pos: list[tuple[int, int]] = []
        num_pair = self.max_pair_length

        for x_idx, x in enumerate(entry["examples"]):
            m1 = x[0]
            label = x[1]
            mentions.append(x[2])
            mention_pos.append((m1[0], m1[1]))
            labels.append(label)

            if self.model_type in ["bertspan", "robertaspan", "albertspan"]:
                continue

            w1 = x_idx
            w2 = w1 + num_pair
            w1 += marker_offset
            w2 += marker_offset
            position_ids[w1] = m1[0]
            position_ids[w2] = m1[1]

            for xx in [w1, w2]:
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1

        if self.model_type.startswith("modernbert") and attention_mask.dim() == 2:
            attention_mask = torch.ones(
                attention_mask.shape[0], dtype=attention_mask.dtype
            )

        labels += [-1] * (num_pair - len(labels))
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))

        item: list[Any] = [
            torch.tensor(input_ids),
            attention_mask,
            torch.tensor(position_ids),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(mention_pos),
            torch.tensor(L, dtype=torch.int64),
        ]

        if self.evaluate:
            item.append(torch.tensor(entry["sent_length"]))

        item.append(entry["example_index"])
        item.append(mentions)
        return item
