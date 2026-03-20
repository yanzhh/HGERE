"""
Inference with fixed (gold) spans.

Given gold NER spans as entity candidates, predicts the highest-probability
non-NIL entity type for each span (NIL is masked out from the NER logits).
Relations are predicted with the standard HGERE logic.

Intended use: evaluate a model trained on one dataset on another dataset's
gold spans, to test cross-dataset entity type prediction.
"""

import json
import timeit
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm

_EVAL_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ent_numbers",
]


def infer_fixed_spans(
    model,
    eval_dataset,
    args,
    logger,
    source_file_path,
    output_path,
    gold_only: bool = False,
    disable_progress: bool = False,
):
    """Run inference using fixed (gold) entity spans as candidates.

    For NER: masks the NIL class from the logits so every span receives the
    highest-probability *non-NIL* label.
    For relations: standard HGERE decoding (NIL filtered out as usual).

    Args:
        model: loaded HGERE model in eval mode.
        eval_dataset: RelationDataset loaded from a file where ``predicted_ner``
            has been set to the desired candidate spans.
        args: inference args (device, etc.).
        logger: Python logger.
        source_file_path: path to the *original* data file used for writing
            output (must contain ``ner`` gold annotations).
        output_path: where to write the prediction JSONL file.
        gold_only: if True, filter the output ``predicted_ner`` /
            ``predicted_rel`` to only spans/relations whose endpoints appear
            in the gold ``ner`` of the source file.
    """
    model.eval()
    start_time = timeit.default_timer()

    ner_predictions: dict = {}
    rel_predictions: dict = defaultdict(list)

    rel_label_list = list(eval_dataset.label_list)
    n_rel_label = len(rel_label_list)
    sym_labels = list(eval_dataset.sym_labels)
    n_syms = len(sym_labels)
    n_unsyms = n_rel_label - n_syms

    nil_index = eval_dataset.ner_label_list.index("NIL")

    with torch.no_grad():
        for batch in tqdm(
            eval_dataset.loader,
            desc="Fixed-span inference",
            total=len(eval_dataset.loader),
            disable=disable_progress,
        ):
            sent_indices = batch["indexs"]
            obj_mentions = batch["obj_token_pos"]
            ent_counts = batch["ent_numbers"]

            inputs = {k: v.to(args.device) for k, v in batch.items() if k in _EVAL_KEYS}
            outputs = model(**inputs)

            rel_logits = outputs[0]
            ner_logits = outputs[1]

            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1)

            # Force non-NIL: mask NIL logit to -inf, then argmax.
            ner_logits_masked = ner_logits.clone()
            ner_logits_masked[..., nil_index] = float("-inf")
            ner_preds = torch.argmax(ner_logits_masked, dim=-1)

            # --- NER predictions ---
            for sample_idx, sent_id in enumerate(sent_indices):
                sent_id = tuple(sent_id)
                sample_obj_mentions = obj_mentions[sample_idx]
                n_spans = len(sample_obj_mentions)
                sample_ner_preds = ner_preds[sample_idx][:n_spans]

                sample_logits_full = ner_logits[sample_idx][:n_spans]
                softmax_full = torch.nn.functional.softmax(sample_logits_full, dim=-1)

                # prob_nil: P(NIL) over all classes
                probs_nil = softmax_full[..., nil_index].cpu().numpy()

                # prob_no_nil: P(predicted label) renormalized over non-NIL classes
                softmax_no_nil = torch.nn.functional.softmax(
                    ner_logits_masked[sample_idx][:n_spans], dim=-1
                )
                probs_no_nil = (
                    softmax_no_nil.gather(-1, sample_ner_preds.unsqueeze(-1))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

                sample_labels = [
                    eval_dataset.ner_label_list[idx] for idx in sample_ner_preds
                ]
                ner_predictions[sent_id] = {
                    tuple(span): (label, float(p_nil), float(p_no_nil))
                    for span, label, p_nil, p_no_nil in zip(
                        sample_obj_mentions, sample_labels, probs_nil, probs_no_nil
                    )
                }

            # --- Relation predictions ---
            for sample_idx, sent_id in enumerate(sent_indices):
                n_ent = ent_counts[sample_idx]
                sent_id = tuple(sent_id)
                for subj_idx, obj_idx in combinations(range(n_ent), 2):
                    subj_span = tuple(obj_mentions[sample_idx][subj_idx])
                    obj_span = tuple(obj_mentions[sample_idx][obj_idx])

                    scores = rel_logits[sample_idx, subj_idx, obj_idx]
                    scores_inv = rel_logits[sample_idx, obj_idx, subj_idx]
                    scores_inv = torch.concat(
                        [
                            scores_inv[:n_syms],
                            scores_inv[n_rel_label:],
                            scores_inv[n_syms:n_rel_label],
                        ]
                    )
                    combined = torch.add(scores, scores_inv)
                    probs = torch.nn.functional.softmax(combined, dim=-1)
                    best_idx = torch.argmax(probs)
                    score = probs[best_idx].cpu().item()

                    if best_idx >= n_rel_label:
                        best_idx = best_idx - n_unsyms
                        subj_span, obj_span = obj_span, subj_span

                    label = rel_label_list[best_idx]
                    if label != "NIL":
                        subj_label, *_ = ner_predictions[sent_id][subj_span]
                        obj_label, *_ = ner_predictions[sent_id][obj_span]
                        rel_predictions[sent_id].append(
                            (subj_span, subj_label, obj_span, obj_label, label, score)
                        )

    # --- Collect per-sentence predictions ---
    tot_ner: dict = {}
    tot_ner_proba: dict = {}
    tot_rel: dict = {}
    tot_rel_proba: dict = {}

    for sent_id, sent_ents in ner_predictions.items():
        sent_rels = sorted(rel_predictions[sent_id], key=lambda x: -x[-1])

        pred_ner = sorted(span + (label,) for span, (label, _, __) in sent_ents.items())
        # predicted_ner_proba entries: [start, end, label, prob_total, prob_no_nil]
        # prob_total  = P(label | all classes incl. NIL)
        # prob_no_nil = P(label | non-NIL classes only)
        pred_ner_proba = sorted(
            span + (label, p_total, p_no_nil)
            for span, (label, p_total, p_no_nil) in sent_ents.items()
        )
        pred_rel = sorted(s + o + (lbl,) for s, _, o, __, lbl, ___ in sent_rels)
        pred_rel_proba = sorted(s + o + (lbl, sc) for s, _, o, __, lbl, sc in sent_rels)

        tot_ner[sent_id] = pred_ner
        tot_ner_proba[sent_id] = pred_ner_proba
        tot_rel[sent_id] = pred_rel
        tot_rel_proba[sent_id] = pred_rel_proba

    # --- Write output ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out_f, open(source_file_path) as in_f:
        for l_idx, line in enumerate(in_f):
            data = json.loads(line)
            num_sents = len(data["sentences"])

            pred_ner = [tot_ner.get((l_idx, n), []) for n in range(num_sents)]
            pred_ner_proba = [
                tot_ner_proba.get((l_idx, n), []) for n in range(num_sents)
            ]
            pred_rel = [tot_rel.get((l_idx, n), []) for n in range(num_sents)]
            pred_rel_proba = [
                tot_rel_proba.get((l_idx, n), []) for n in range(num_sents)
            ]

            if gold_only:
                gold_ner = data.get("ner", [[] for _ in range(num_sents)])
                for n in range(num_sents):
                    gold_spans = {
                        (e[0], e[1]) for e in (gold_ner[n] if n < len(gold_ner) else [])
                    }
                    pred_ner[n] = [e for e in pred_ner[n] if (e[0], e[1]) in gold_spans]
                    pred_ner_proba[n] = [
                        e for e in pred_ner_proba[n] if (e[0], e[1]) in gold_spans
                    ]
                    pred_rel[n] = [
                        r
                        for r in pred_rel[n]
                        if (r[0], r[1]) in gold_spans and (r[2], r[3]) in gold_spans
                    ]
                    pred_rel_proba[n] = [
                        r
                        for r in pred_rel_proba[n]
                        if (r[0], r[1]) in gold_spans and (r[2], r[3]) in gold_spans
                    ]

            data["predicted_ner"] = pred_ner
            data["predicted_ner_proba"] = pred_ner_proba
            data["predicted_rel"] = pred_rel
            data["predicted_rel_proba"] = pred_rel_proba
            out_f.write(json.dumps(data) + "\n")

    elapsed = timeit.default_timer() - start_time
    logger.info("Fixed-span inference done in %.1f s", elapsed)


def make_augmented_candidate_file(source_file_path, tmp_file_path):
    """Write a copy of *source_file_path* where ``predicted_ner`` is the union
    of the existing ``predicted_ner`` (pruner/pipeline output) and the gold
    ``ner`` spans.

    Gold spans not already present in ``predicted_ner`` are appended with their
    original label.  This ensures every gold span is scored by HGERE while
    keeping the full pipeline context intact.
    """
    with open(source_file_path) as in_f, open(tmp_file_path, "w") as out_f:
        for line in in_f:
            data = json.loads(line)
            pred_ner = data.get("predicted_ner", [])
            gold_ner = data.get("ner", [])
            num_sents = len(data["sentences"])

            merged = []
            for sent_idx in range(num_sents):
                pred_spans = pred_ner[sent_idx] if sent_idx < len(pred_ner) else []
                gold_spans = gold_ner[sent_idx] if sent_idx < len(gold_ner) else []
                gold_set = {(s[0], s[1]) for s in gold_spans}
                # Gold spans come first so they are never truncated by max_ents.
                augmented = list(gold_spans)
                for span in pred_spans:
                    if (span[0], span[1]) not in gold_set:
                        augmented.append(span)
                merged.append(augmented)

            data["predicted_ner"] = merged
            out_f.write(json.dumps(data) + "\n")


def make_gold_span_file(source_file_path, tmp_file_path):
    """Write a copy of *source_file_path* where ``predicted_ner`` = gold ``ner``.

    This ensures the dataset uses gold entity spans as candidates instead of
    pruner predictions.
    """
    with open(source_file_path) as in_f, open(tmp_file_path, "w") as out_f:
        for line in in_f:
            data = json.loads(line)
            data["predicted_ner"] = data["ner"]
            out_f.write(json.dumps(data) + "\n")
