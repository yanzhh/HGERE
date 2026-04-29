"""Evaluation functions for the HGERE model.

Moved verbatim from run_hgnn.py (lines 440-445, 488-777, 1714-1724).
"""

from __future__ import annotations

import glob
import json
import os
import timeit
from collections import defaultdict
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

# Keys that are passed to the model but must NOT be moved to a CUDA device.
_NON_TENSOR_KEYS = {"dataset_id"}


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

    eval_output_dir = getattr(args, "output_dir", None) or args.model_dir
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

    # Cache triu indices per n_ent to avoid recomputing each batch
    _triu_cache: dict[int, tuple] = {}

    with torch.inference_mode():
        for batch in tqdm(eval_dataset.loader, desc="Evaluating"):
            sent_indices = batch["indices"]
            obj_mentions = batch["obj_token_pos"]
            ent_counts = batch["ent_numbers"]
            inputs = {}

            for k, v in batch.items():
                if k in EVAL_KEYS:
                    inputs[k] = v.to(args.device)
                elif k in _NON_TENSOR_KEYS:
                    inputs[k] = v

            if inputs["ent_numbers"].sum() == 0:
                for sent_id in sent_indices:
                    ner_predictions[tuple(sent_id)] = {}
                continue

            outputs = model(**inputs)

            rel_logits = outputs[0]
            ner_logits = outputs[1]

            # Apply log-softmax for relations, then move everything to CPU once
            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1).cpu()
            ner_logits = ner_logits.cpu()
            ner_preds = torch.argmax(ner_logits, dim=-1)

            # Batch NER softmax for all sentences at once
            ner_softmax = torch.nn.functional.softmax(ner_logits, dim=-1)
            ner_probs = ner_softmax.gather(-1, ner_preds.unsqueeze(-1)).squeeze(-1)

            # NER decoding
            for sample_idx, sent_id in enumerate(sent_indices):
                sent_id = tuple(sent_id)
                sample_obj_mentions = obj_mentions[sample_idx]
                n_ent = len(sample_obj_mentions)
                sample_ner_preds = ner_preds[sample_idx][:n_ent].tolist()
                sample_ner_probs = ner_probs[sample_idx][:n_ent].tolist()
                ner_predictions[sent_id] = {
                    tuple(span): (eval_dataset.ner_label_list[lbl], float(prob))
                    for span, lbl, prob in zip(
                        sample_obj_mentions, sample_ner_preds, sample_ner_probs
                    )
                }

            # Relation decoding — vectorized over all pairs per sentence
            for sample_idx, sent_id in enumerate(sent_indices):
                n_ent = int(ent_counts[sample_idx])
                sent_id = tuple(sent_id)

                if n_ent < 2:
                    continue

                if n_ent not in _triu_cache:
                    _triu_cache[n_ent] = torch.triu_indices(n_ent, n_ent, offset=1)
                subj_idxs, obj_idxs = _triu_cache[n_ent]

                # Extract scores for all pairs at once: (n_pairs, n_labels)
                scores = rel_logits[sample_idx, subj_idxs, obj_idxs]
                scores_inv_raw = rel_logits[sample_idx, obj_idxs, subj_idxs]
                scores_inv = torch.cat(
                    [
                        scores_inv_raw[:, :n_syms],
                        scores_inv_raw[:, n_rel_label:],
                        scores_inv_raw[:, n_syms:n_rel_label],
                    ],
                    dim=-1,
                )
                combined = scores + scores_inv
                probs = torch.nn.functional.softmax(combined, dim=-1)
                best_idxs = probs.argmax(dim=-1)
                best_scores = probs.gather(1, best_idxs.unsqueeze(1)).squeeze(1)

                best_idxs = best_idxs.tolist()
                best_scores = best_scores.tolist()
                subj_idxs = subj_idxs.tolist()
                obj_idxs = obj_idxs.tolist()

                for pair_idx, (si, oi) in enumerate(zip(subj_idxs, obj_idxs)):
                    best_rel_label_idx = best_idxs[pair_idx]
                    score = best_scores[pair_idx]
                    subj_span = tuple(obj_mentions[sample_idx][si])
                    obj_span = tuple(obj_mentions[sample_idx][oi])
                    if best_rel_label_idx >= n_rel_label:
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
        out_path = os.path.join(
            getattr(args, "output_dir", None) or args.model_dir, target_fn
        )
        with open(out_path, "w") as output_w:
            for doc in docs:
                output_w.write(json.dumps(doc) + "\n")
        logger.info("Predictions written to %s", out_path)

    return results


def get_checkpoints(args: Any) -> list:
    if args.eval_all_checkpoints:
        found = sorted(
            glob.glob(args.model_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            + glob.glob(args.model_dir + "/**/" + SAFETENSORS_NAME, recursive=True)
        )
        return list(dict.fromkeys(os.path.dirname(c) for c in found))

    # Default: only the most recent (best-F1) checkpoint, identified by the
    # highest step number in the checkpoint directory name.
    ckpt_dirs = sorted(
        glob.glob(os.path.join(args.model_dir, "checkpoint-*")),
        key=lambda p: (
            int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1
        ),
    )
    if ckpt_dirs:
        return [ckpt_dirs[-1]]
    return [args.model_dir]
