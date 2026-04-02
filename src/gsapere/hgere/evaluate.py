"""Evaluation functions for the HGERE model.

Moved verbatim from run_hgnn.py (lines 440-445, 488-777, 1714-1724).
"""

from __future__ import annotations

import glob
import json
import os
import timeit
from collections import defaultdict
from itertools import combinations
from typing import Any

import torch
from tqdm import tqdm

from gsapere.evaluation.hgere import (
    compute_ner_metrics,
    compute_rel_metrics,
    compute_rel_metrics_with_ner,
)

WEIGHTS_NAME = "pytorch_model.bin"
SAFETENSORS_NAME = "model.safetensors"

EVAL_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ent_numbers",
]


def get_gold_ner_with_nolabel(ner_golden_labels: set) -> set:
    ner_golden_nolabels = set()
    for ner in ner_golden_labels:
        ner_nolabel = (ner[0], ner[1])
        ner_golden_nolabels.add(ner_nolabel)
    return ner_golden_nolabels


def _build_eval_docs_from_file(
    ner_predictions: dict,
    rel_predictions: dict,
    file_path: str,
) -> list:
    """Merge in-memory model predictions with gold data from the original JSONL file.

    Reading gold data from the original file (not from the dataset's filtered
    in-memory structures) ensures the recall denominator is correct (all gold
    relations, not only those whose spans were included as candidates).

    Each doc in the returned list contains all original fields plus:
    ``doc_id``, ``predicted_ner``, ``predicted_ner_proba``,
    ``predicted_rel``, ``predicted_rel_proba``.
    """
    docs = []
    with open(file_path) as f:
        for doc_idx, line in enumerate(f):
            raw = json.loads(line)
            n_sents = len(raw["sentences"])
            pred_ner: list = []
            pred_ner_proba: list = []
            pred_rel: list = []
            pred_rel_proba: list = []
            for si in range(n_sents):
                sent_id = (doc_idx, si)
                sent_ents = ner_predictions.get(sent_id, {})
                sent_rels = sorted(
                    rel_predictions.get(sent_id, []), key=lambda x: -x[-1]
                )
                pred_ner.append(
                    sorted(
                        [
                            list(span) + [label]
                            for span, (label, _score) in sent_ents.items()
                            if label != "NIL"
                        ]
                    )
                )
                pred_ner_proba.append(
                    sorted(
                        [
                            list(span) + [label, score]
                            for span, (label, score) in sent_ents.items()
                            if label != "NIL"
                        ]
                    )
                )
                pred_rel.append(
                    sorted(
                        [
                            list(subj_span) + list(obj_span) + [label]
                            for subj_span, _sl, obj_span, _ol, label, _sc in sent_rels
                        ]
                    )
                )
                pred_rel_proba.append(
                    sorted(
                        [
                            list(subj_span) + list(obj_span) + [label, score]
                            for subj_span, _sl, obj_span, _ol, label, score in sent_rels
                        ]
                    )
                )
            docs.append(
                {
                    **raw,
                    "doc_id": raw.get("doc_id", doc_idx),
                    "predicted_ner": pred_ner,
                    "predicted_ner_proba": pred_ner_proba,
                    "predicted_rel": pred_rel,
                    "predicted_rel_proba": pred_rel_proba,
                }
            )
    return docs


def evaluate(
    model: Any,
    eval_dataset: Any,
    args: Any,
    logger: Any,
    prefix: str = "",
    persist_predictions: bool = False,
) -> dict:

    eval_output_dir = args.model_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    logger.info(f"***** Running evaluation {prefix} *****")

    model.eval()

    start_time = timeit.default_timer()

    # ---------------------------------------------------

    ner_predictions = {}
    rel_predictions = defaultdict(list)

    rel_label_list = list(eval_dataset.label_list)
    n_rel_label = len(rel_label_list)
    sym_labels = list(eval_dataset.sym_labels)
    n_syms = len(sym_labels)
    n_unsyms = n_rel_label - n_syms

    with torch.no_grad():
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in tqdm(eval_dataset.loader, desc="Evaluating"):
            sent_indices = batch["indices"]
            obj_mentions = batch["obj_token_pos"]
            # subjs = batch["sub"]
            # print(subjs)

            # rel_labels = batch["rel_labels"]
            # ner_labels = batch["ner_labels"]
            ent_counts = batch["ent_numbers"]
            inputs = {}
            input_keys = EVAL_KEYS

            for k, v in batch.items():
                if k in input_keys:
                    v = v.to(args.device)
                    inputs[k] = v

            outputs = model(**inputs)

            rel_logits = outputs[0]
            # print("# bs * n_ent * n_ent * num_rel_labels")
            # print(rel_logits.shape)
            ner_logits = outputs[1]

            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1)

            # print("rel_logits")
            # 2 * 10 * 10 * 23
            # n_sents, n_ents, n_ents, n_rel_label
            # print(rel_logits.shape)
            # print(rel_logits)
            ner_preds = torch.argmax(ner_logits, dim=-1)

            # print("ner")
            # print(ner_preds_label)
            # print(sent_indices)
            # rel_logits = (
            #    rel_logits.cpu().numpy()
            # )  # for plmk, n_ent_total * max_n_ent * num_rel_labels
            # ner_preds = (
            #    ner_preds.cpu().numpy()
            # )  # for plmk, n_ent_total * num_ner_labels
            # print(f'indexs:{indexs}')
            # print(f'ner_labels: {ner_labels}')
            # if args.baseline not in {'firstorder', 'mfvi', 'gnn'}:
            #     rel_logits_split = torch.split(rel_logits, ent_numbers)
            #     rel_logits = pad_sequence(rel_logits_split, batch_first=True, padding_value=0)

            # NER Label for entities
            for sample_idx, sent_id in enumerate(sent_indices):
                n_ent = ent_counts[sample_idx]

                sent_id = tuple(sent_id)
                sample_obj_mentions = obj_mentions[sample_idx]
                sample_ner_preds = ner_preds[sample_idx][: len(sample_obj_mentions)]
                sample_ner_logits = ner_logits[sample_idx][: len(sample_obj_mentions)]
                sample_ner_softmax = torch.nn.functional.softmax(
                    sample_ner_logits, dim=-1
                )
                sample_ner_probs = sample_ner_softmax.gather(
                    -1, sample_ner_preds.unsqueeze(-1)
                ).squeeze(-1)
                sample_ner_probs = sample_ner_probs.cpu().numpy()
                sample_ner_labels = [
                    eval_dataset.ner_label_list[label_idx]
                    for label_idx in ner_preds[sample_idx]
                ]
                ner_predictions[sent_id] = {}
                for ent_span, label, score in zip(
                    sample_obj_mentions, sample_ner_labels, sample_ner_probs
                ):
                    ent_span = tuple(ent_span)
                    score = float(score)
                    ner_predictions[sent_id][ent_span] = label, score

            # Relations
            for sample_idx, sent_id in enumerate(sent_indices):
                """
                sent_rel_logits = rel_logits[sample_idx]
                sent_rel_logits_T = sent_rel_logits.T
                # Rearange labels to combine label scores with inverse label scores
                x, y = n_syms, n_label
                # inverse dim: n_label * n_label (after n_label are the inverse versions of the labels)
                sent_rel_logits_T_rearranged = torch.concat([sent_rel_logits_T[:n_syms], sent_rel_logits_T[n_label:]]])
                sent_rel_logits = torch.add(sent_rel_logits[:n_label], sent_rel_logits_T_rearranged)
                # Calculate prob
                sent_rel_logits = torch.nn.softmax(sent_rel_logits)
                """
                n_ent = ent_counts[sample_idx]
                sent_id = tuple(sent_id)
                # obj tokens, e.g.: [(2, 3), (3, 4), (6, 6), (6, 7), (10, 10), (10, 11), (13, 14)]
                # go through all unique entity combinations
                for subj_idx, obj_idx in combinations(list(range(n_ent)), 2):
                    subj_span = tuple(obj_mentions[sample_idx][subj_idx])
                    obj_span = tuple(obj_mentions[sample_idx][obj_idx])
                    # Calc best score (incl. inverse label)
                    sample_rel_scores = rel_logits[sample_idx, subj_idx, obj_idx]
                    sample_rel_scores_inv = rel_logits[sample_idx, obj_idx, subj_idx]
                    sample_rel_scores_inv = torch.concat(
                        [
                            sample_rel_scores_inv[:n_syms],
                            sample_rel_scores_inv[n_rel_label:],
                            sample_rel_scores_inv[n_syms:n_rel_label],
                        ]
                    )
                    sample_rel_scores = torch.add(
                        sample_rel_scores, sample_rel_scores_inv
                    )
                    sample_rel_probs = torch.nn.functional.softmax(
                        sample_rel_scores, dim=-1
                    )
                    best_rel_label_idx = torch.argmax(sample_rel_probs)
                    score = sample_rel_probs[best_rel_label_idx].cpu().item()
                    # inverse = False
                    if best_rel_label_idx >= n_rel_label:
                        # the inverse is better!
                        # inverse = True
                        best_rel_label_idx -= n_unsyms
                        subj_span, obj_span = obj_span, subj_span
                    label = rel_label_list[best_rel_label_idx]
                    if label != "NIL":
                        subj_label, subj_score = ner_predictions[sent_id][subj_span]
                        obj_label, obj_score = ner_predictions[sent_id][obj_span]
                        rel_predictions[sent_id].append(
                            (subj_span, subj_label, obj_span, obj_label, label, score)
                        )

    # ---------------------------------------------------
    # Build evaluation docs and compute metrics
    docs = _build_eval_docs_from_file(
        ner_predictions, rel_predictions, eval_dataset.file_path
    )
    sym_labels_tuple = tuple(sym_labels[1:])  # exclude NIL prefix

    ner_metrics = compute_ner_metrics(docs)
    re_metrics = compute_rel_metrics(docs, sym_labels=sym_labels_tuple)
    re_plus_metrics = compute_rel_metrics_with_ner(docs, sym_labels=sym_labels_tuple)

    evalTime = timeit.default_timer() - start_time
    global_predicted_ners = eval_dataset.global_predicted_ners
    logger.info(
        "  Evaluation done in total %f secs (%f example per second)",
        evalTime,
        len(global_predicted_ners) / evalTime,
    )

    # Upper recall bounds from candidate span coverage
    gold_ners = eval_dataset.ner_golden_labels
    gold_rels = set(eval_dataset.golden_labels)
    n_gold_unique_ner = ner_metrics["ner_n_gold"]
    tot_recall = eval_dataset.tot_recall

    gold_ner_span_positions = {(sent_id, span) for sent_id, span, _ in gold_ners}
    candidate_span_positions = {
        (sent_id, (start, end))
        for sent_id, cands in global_predicted_ners.items()
        for start, end, _ in cands
    }
    n_gold_ner_in_cands = len(gold_ner_span_positions & candidate_span_positions)
    ner_upper_recall = (
        n_gold_ner_in_cands / n_gold_unique_ner if n_gold_unique_ner > 0 else 1.0
    )

    n_re_plus_achievable = sum(
        1
        for sent_id, subj_span, obj_span, _label in gold_rels
        if (sent_id, subj_span) in candidate_span_positions
        and (sent_id, obj_span) in candidate_span_positions
    )
    re_plus_upper_recall = n_re_plus_achievable / tot_recall if tot_recall > 0 else 1.0

    results = {
        **ner_metrics,
        **re_metrics,
        **re_plus_metrics,
        "ner_upper_recall": ner_upper_recall,
        "re+_upper_recall": re_plus_upper_recall,
    }

    logger.info(f"Result: {json.dumps(results, indent=4)}")

    if persist_predictions and getattr(args, "save_results", True):
        target_fn = os.path.split(eval_dataset.file_path)[-1]
        out_path = os.path.join(args.model_dir, target_fn)
        with open(out_path, "w") as output_w:
            for doc in docs:
                output_w.write(json.dumps(doc) + "\n")
        logger.info("Predictions written to %s", out_path)

    return results


def get_checkpoints(args: Any) -> list:
    checkpoints = [args.model_dir]

    if args.eval_all_checkpoints:
        found = sorted(
            glob.glob(args.model_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            + glob.glob(args.model_dir + "/**/" + SAFETENSORS_NAME, recursive=True)
        )
        checkpoints = list(dict.fromkeys(os.path.dirname(c) for c in found))
    return checkpoints
