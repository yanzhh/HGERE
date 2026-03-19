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

WEIGHTS_NAME = "pytorch_model.bin"

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


def evaluate(
    model: Any,
    eval_dataset: Any,
    args: Any,
    logger: Any,
    prefix: str = "",
    persist_predictions: bool = False,
) -> dict:

    eval_output_dir = args.output_dir
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
            sent_indices = batch["indexs"]
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
    # decode
    global_predicted_ners = eval_dataset.global_predicted_ners
    gold_rels = set(eval_dataset.golden_labels)
    gold_rels_with_ner = set(eval_dataset.golden_labels_with_ner)
    gold_ners = eval_dataset.ner_golden_labels

    tot_recall = eval_dataset.tot_recall
    n_pred_ner = 0
    n_tp_ner = 0
    n_pred_rel = 0
    n_tp_rel = 0
    n_tp_rel_with_ner = 0

    tot_predicted_relations = {}
    tot_predicted_ners = {}
    tot_predicted_relations_proba = {}
    tot_predicted_ners_proba = {}

    for sent_id, sent_ents_pred in ner_predictions.items():
        sent_relations = rel_predictions[sent_id]
        # sort by prob
        sent_relations = sorted(sent_relations, key=lambda x: -x[-1])

        sent_relation_keys = set()
        sent_relation_keys_with_ner = set()
        sent_ent_keys = set()
        output_pred_rels = []
        output_pred_rels_proba = []
        output_pred_ner = []
        output_pred_ner_proba = []
        for subj_span, subj_label, obj_span, obj_label, label, score in sent_relations:
            sent_relation_keys.add((sent_id, subj_span, obj_span, label))
            sent_relation_keys_with_ner.add(
                (sent_id, subj_span + (subj_label,), obj_span + (obj_label,), label)
            )
            output_pred_rels.append(subj_span + obj_span + (label,))
            output_pred_rels_proba.append(subj_span + obj_span + (label, score))
        for ent_span, (label, score) in sent_ents_pred.items():
            if label == "NIL":
                continue
            sent_ent_keys.add((sent_id, ent_span, label))
            output_pred_ner.append(ent_span + (label,))
            output_pred_ner_proba.append(ent_span + (label, score))

        sent_tp_rel = sent_relation_keys & gold_rels
        sent_tp_rel_with_ner = sent_relation_keys_with_ner & gold_rels_with_ner
        sent_tp_ner = sent_ent_keys & gold_ners

        n_pred_rel += len(sent_relation_keys)
        n_tp_rel += len(sent_tp_rel)
        n_tp_rel_with_ner += len(sent_tp_rel_with_ner)

        n_pred_ner += len(sent_ent_keys)
        n_tp_ner += len(sent_tp_ner)

        # pdb.set_trace()
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # TODO add tot_predicted by doc_id and sent_nr to save
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        tot_predicted_relations[sent_id] = output_pred_rels
        tot_predicted_relations_proba[sent_id] = output_pred_rels_proba
        tot_predicted_ners[sent_id] = output_pred_ner
        tot_predicted_ners_proba[sent_id] = output_pred_ner_proba

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f example per second)",
        evalTime,
        len(global_predicted_ners) / evalTime,
    )

    ner_p = n_tp_ner / n_pred_ner if n_pred_ner > 0 else 0
    ner_r = n_tp_ner / len(gold_ners) if gold_ners else 0.0
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if n_tp_ner > 0 else 0.0

    p = n_tp_rel / n_pred_rel if n_pred_rel > 0 else 0
    r = n_tp_rel / tot_recall if tot_recall > 0.0 else 0.0
    f1 = 2 * (p * r) / (p + r) if n_tp_rel > 0 else 0.0

    # assert(tot_recall==len(golden_labels))

    p_with_ner = n_tp_rel_with_ner / n_pred_rel if n_pred_rel > 0 else 0
    r_with_ner = n_tp_rel_with_ner / tot_recall if tot_recall else 0.0
    # assert(tot_recall==len(golden_labels_with_ner))
    f1_with_ner = (
        2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner)
        if n_tp_rel_with_ner > 0
        else 0.0
    )

    results = {
        "ner_precision": ner_p,
        "ner_recall": ner_r,
        "ner_f1": ner_f1,
        "re_precision": p,
        "re_recall": r,
        "re_f1": f1,
        "re+_precision": p_with_ner,
        "re+_recall": r_with_ner,
        "re+_f1": f1_with_ner,
    }

    logger.info(f"Result: {json.dumps(results, indent=4)}")
    # dump predictions
    if persist_predictions:
        target_fn = os.path.split(eval_dataset.file_path)[-1]
        output_w = open(os.path.join(args.output_dir, target_fn), "w")
        file_raw_data = open(eval_dataset.file_path)
        for l_idx, line in enumerate(file_raw_data):
            data = json.loads(line)
            num_sents = len(data["sentences"])
            predicted_ner = []
            predicted_ner_proba = []
            predicted_rel = []
            predicted_rel_proba = []
            for n in range(num_sents):
                ner_item = tot_predicted_ners.get((l_idx, n), [])
                ner_item.sort()
                predicted_ner.append(ner_item)
                ner_item = tot_predicted_ners_proba.get((l_idx, n), [])
                ner_item.sort()
                predicted_ner_proba.append(ner_item)
                rel_item = tot_predicted_relations.get((l_idx, n), [])
                rel_item.sort()
                predicted_rel.append(rel_item)
                rel_item = tot_predicted_relations_proba.get((l_idx, n), [])
                rel_item.sort()
                predicted_rel_proba.append(rel_item)
            data["predicted_ner"] = predicted_ner
            data["predicted_rel"] = predicted_rel
            data["predicted_ner_proba"] = predicted_ner_proba
            data["predicted_rel_proba"] = predicted_rel_proba
            # pdb.set_trace()
            output_w.write(json.dumps(data) + "\n")
            # json.dump(tot_output_results, output_w)

    return results


def get_checkpoints(args: Any) -> list:
    checkpoints = [args.output_dir]

    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(
                glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            )
        )
    return checkpoints
