"""
collators.py — Collator classes for PrunerDataset and RelationDataset.

Replaces the static/instance collate_fn methods embedded in the dataset classes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PrunerCollator:
    """Collator for PrunerDataset batches.

    Stacks all tensor fields; preserves the last `num_metadata_fields` fields
    as plain Python lists (example_index and mentions).

    Replaces ACEDatasetNER.collate_fn (static method).
    """

    num_metadata_fields: int = 2

    def __call__(self, batch: list[list[Any]]) -> list[Any]:
        fields = [x for x in zip(*batch)]
        stacked = [torch.stack(f) for f in fields[: -self.num_metadata_fields]]
        stacked.extend(fields[-self.num_metadata_fields :])
        return stacked


@dataclass
class RelationCollator:
    """Collator for RelationDataset batches.

    Pads batches to the maximum number of entities in the batch and stacks
    all tensor fields into a dict keyed by the batch field names used by
    run_hgnn.py.

    Replaces RelationDataset.collate_fn (instance method).
    """

    tokenizer_pad_id: int
    max_seq_length: int

    def __call__(self, batch: list[Any]) -> dict[str, Any]:
        return _collate_relation_batch(
            batch, self.tokenizer_pad_id, self.max_seq_length
        )


def _collate_relation_batch(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    max_seq_length: int,
) -> dict[str, Any]:
    """Collate variable-length sentence items into padded batch tensors.

    Each sentence item has a different number of subject candidates (n_ent).
    The collated tensors use two indexing dimensions:
    - Sentence dimension (n_sent): for label tensors (rel_labels, ner_labels).
    - Flat subject-candidate dimension (total_subject_candidates): for input_ids,
      attention_mask, position_ids, and sub_positions. Candidates from different
      sentences are concatenated along this axis.

    Attention masks and position_ids in the flat dimension are padded to the
    maximum entity count (max_ent_num) within the batch to support uniform
    matrix operations in the hypergraph encoder.
    """
    ent_numbers = torch.tensor([sent["n_ent"] for sent in batch])
    n_sent = len(batch)

    if n_sent == 1 and int(ent_numbers[0]) > 0:
        # Fast path: single sentence with entities needs no padding, just wrapping.
        # Skipped when n_ent == 0: torch.tensor([]) is 1D and would break BERT.
        item = deepcopy(batch[0])
        item.pop("n_ent")
        item.pop("subtoken_len")
        item["ent_numbers"] = ent_numbers
        for k in ["rel_labels", "ner_labels"]:
            item[k] = item[k].unsqueeze(0)
        for k in ["indices", "sub", "obj_token_pos"]:
            item[k] = [item[k]]
        return item

    subject_token_spans: list[Any] = []
    obj_token_pos: list[Any] = []
    indices: list[Any] = []
    max_ent_num = int(max(ent_numbers))
    seqlen = batch[0]["subtoken_len"]
    total_subject_candidates = int(sum(ent_numbers))

    batch_input_ids = (
        torch.zeros(
            (total_subject_candidates, seqlen + 2 * max_ent_num), dtype=torch.int
        )
        + pad_token_id
    )
    batch_pos_ids = torch.zeros(
        (total_subject_candidates, seqlen + 2 * max_ent_num), dtype=torch.int
    )
    batch_subject_subtoken_pos = torch.zeros(
        (total_subject_candidates, 2), dtype=torch.long
    )
    batch_attn_mask = torch.zeros(
        (total_subject_candidates, seqlen + 2 * max_ent_num, seqlen + 2 * max_ent_num),
        dtype=torch.int,
    )
    batch_rel_labels = (
        torch.zeros((n_sent, max_ent_num, max_ent_num), dtype=torch.int64) - 1
    )
    batch_ner_labels = torch.zeros((n_sent, max_ent_num), dtype=torch.int64) - 1

    subject_offset = 0
    for i, sent_item in enumerate(batch):
        indices.append(sent_item["indices"])
        n_ent = int(ent_numbers[i])
        if n_ent > 0:
            subject_offset_end = subject_offset + n_ent

            input_ids_parts = sent_item["input_ids"].split(
                [seqlen, n_ent, n_ent], dim=-1
            )
            batch_input_ids[subject_offset:subject_offset_end, :seqlen] = (
                input_ids_parts[0]
            )
            for j in range(2):
                batch_input_ids[
                    subject_offset:subject_offset_end,
                    seqlen + j * max_ent_num : seqlen + j * max_ent_num + n_ent,
                ] = input_ids_parts[j + 1]

            pos_ids_parts = sent_item["position_ids"].split(
                [seqlen, n_ent, n_ent], dim=-1
            )
            batch_pos_ids[subject_offset:subject_offset_end, :seqlen] = pos_ids_parts[0]
            for j in range(2):
                batch_pos_ids[
                    subject_offset:subject_offset_end,
                    seqlen + j * max_ent_num : seqlen + j * max_ent_num + n_ent,
                ] = pos_ids_parts[j + 1]

            batch_subject_subtoken_pos[subject_offset:subject_offset_end, :] = (
                sent_item["sub_positions"]
            )

            sent_attn_mask = sent_item["attention_mask"]
            batch_attn_mask[subject_offset:subject_offset_end, :seqlen, :seqlen] = (
                sent_attn_mask[:, :seqlen, :seqlen]
            )
            for j in range(2):
                batch_attn_mask[
                    subject_offset:subject_offset_end,
                    seqlen + j * max_ent_num : seqlen + n_ent + j * max_ent_num,
                    :seqlen,
                ] = sent_attn_mask[
                    :, seqlen + j * n_ent : seqlen + (j + 1) * n_ent, :seqlen
                ]
                for k in range(2):
                    batch_attn_mask[
                        subject_offset:subject_offset_end,
                        seqlen + j * max_ent_num : seqlen + n_ent + j * max_ent_num,
                        seqlen + k * max_ent_num : seqlen + n_ent + k * max_ent_num,
                    ] = sent_attn_mask[
                        :,
                        seqlen + j * n_ent : seqlen + (j + 1) * n_ent,
                        seqlen + k * n_ent : seqlen + (k + 1) * n_ent,
                    ]

            batch_rel_labels[i, :n_ent, :n_ent] = sent_item["rel_labels"]
            batch_ner_labels[i, :n_ent] = sent_item["ner_labels"]
            subject_token_spans.append(sent_item["sub"])
            obj_token_pos.append(sent_item["obj_token_pos"])
            subject_offset = subject_offset_end
        else:
            obj_token_pos.append([])
            subject_token_spans.append([])

    return dict(
        indices=indices,
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
        position_ids=batch_pos_ids,
        sub_positions=batch_subject_subtoken_pos,
        rel_labels=batch_rel_labels,
        ner_labels=batch_ner_labels,
        sub=subject_token_spans,
        obj_token_pos=obj_token_pos,
        ent_numbers=ent_numbers,
    )
