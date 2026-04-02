"""
HGERE inference over candidate spans.

Candidate source
----------------
Controlled by ``candidates_from`` in :func:`prepare_input_file`:

``"predicted_ner"`` (default)
    Use the ``predicted_ner`` field — pruner output in normal pipeline mode.

``"ner"``
    Replace candidates with gold ``ner`` spans.  Use for gold-span evaluation.

``augment_with_gold=True``
    Merge ``predicted_ner`` with gold ``ner`` spans so every gold span is
    seen by HGERE.  Useful for upper-bound / oracle experiments.

Output format for ``predicted_ner_proba``
-----------------------------------------
Each entry is ``[start, end, score, label]`` where ``score`` is
P(label | non-NIL classes) — the same format expected by the pre-filter
threshold in :func:`gsapere.data.relation_dataset.RelationDataset`.

NER decoding modes
------------------
Controlled by ``force_non_nil`` in :func:`infer_hgere`:

``force_non_nil=False`` (default — pipeline mode)
    Standard softmax argmax over all NER classes, including NIL.
    Only spans whose predicted label is *not* NIL appear in ``predicted_ner``
    and ``predicted_ner_proba``.  Relations are only predicted between pairs
    where *both* endpoints received a non-NIL label.

``force_non_nil=True`` (gold-span / cross-dataset mode)
    NIL is masked out of the NER logits before argmax, so every candidate
    span is forced to receive the highest-probability *non-NIL* label.

Relations are always predicted with the standard HGERE decoding logic
(NIL relations are filtered out).
"""

import json
import shutil
import timeit
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

_EVAL_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ent_numbers",
]


def infer_hgere(
    model,
    eval_dataset,
    args,
    logger,
    source_file_path,
    output_path,
    gold_only: bool = False,
    disable_progress: bool = False,
    force_non_nil: bool = False,
    debug_break_on_first_rel: bool = False,
    debug_log_rel_probs: bool = False,
):
    """Run HGERE inference over candidate spans already loaded in eval_dataset.

    Candidates come from whatever ``predicted_ner`` field was set when the
    dataset was built.  Use :func:`prepare_input_file` beforehand to control
    the candidate source (pruner output, gold spans, or augmented).

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
        force_non_nil: if True, mask NIL from NER logits before argmax so
            every candidate span is forced to receive a non-NIL label.
            Use this only for gold-span / cross-dataset evaluation.
            Default False (pipeline mode): standard argmax including NIL;
            only non-NIL spans appear in output.
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
            desc="ERE inference",
            total=len(eval_dataset.loader),
            disable=disable_progress,
        ):
            sent_indices = batch["indices"]
            obj_mentions = batch["obj_token_pos"]
            ent_counts = batch["ent_numbers"]

            inputs = {k: v.to(args.device) for k, v in batch.items() if k in _EVAL_KEYS}
            outputs = model(**inputs)

            rel_logits = outputs[0]
            ner_logits = outputs[1]

            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1)

            ner_logits_masked = ner_logits.clone()
            ner_logits_masked[..., nil_index] = float("-inf")

            if force_non_nil:
                # Gold-span mode: mask NIL so every span gets a non-NIL label.
                ner_preds = torch.argmax(ner_logits_masked, dim=-1)
            else:
                # Pipeline mode: standard argmax; NIL is a valid prediction.
                ner_preds = torch.argmax(ner_logits, dim=-1)

            # --- NER predictions ---
            for sample_idx, sent_id in enumerate(sent_indices):
                sent_id = tuple(sent_id)
                sample_obj_mentions = obj_mentions[sample_idx]
                n_spans = len(sample_obj_mentions)
                sample_ner_preds = ner_preds[sample_idx][:n_spans]

                # prob_no_nil: P(predicted label) renormalized over non-NIL classes
                softmax_no_nil = torch.nn.functional.softmax(
                    ner_logits_masked[sample_idx][:n_spans], dim=-1
                )
                # @todo Why no_nil here? could we leave that out?
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
                    tuple(span): (label, float(p_no_nil))
                    for span, label, p_no_nil in zip(
                        sample_obj_mentions, sample_labels, probs_no_nil
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

                    if debug_log_rel_probs:
                        probs_cpu = probs.cpu().tolist()
                        all_labels = rel_label_list + rel_label_list[n_syms:]
                        top = sorted(zip(all_labels, probs_cpu), key=lambda x: -x[1])[
                            :5
                        ]
                        logger.info(
                            "[DEBUG REL] sent=%s subj=%s obj=%s  top5=%s",
                            sent_id,
                            subj_span,
                            obj_span,
                            [(lbl, f"{p:.4f}") for lbl, p in top],
                        )

                    if best_idx >= n_rel_label:
                        best_idx = best_idx - n_unsyms
                        subj_span, obj_span = obj_span, subj_span

                    label = rel_label_list[best_idx]
                    if label != "NIL":
                        subj_label = ner_predictions[sent_id][subj_span][0]
                        obj_label = ner_predictions[sent_id][obj_span][0]
                        rel_predictions[sent_id].append(
                            (subj_span, subj_label, obj_span, obj_label, label, score)
                        )
                        if debug_break_on_first_rel:
                            logger.info(
                                "[DEBUG] First relation predicted: sent_id=%s  "
                                "subj=%s(%s) -> obj=%s(%s)  label=%s  score=%.4f",
                                sent_id,
                                subj_span,
                                subj_label,
                                obj_span,
                                obj_label,
                                label,
                                score,
                            )
                            raise RuntimeError(
                                f"[DEBUG] First relation found — "
                                f"sent_id={sent_id} label={label!r} score={score:.4f}"
                            )

    # --- Collect per-sentence predictions ---
    tot_ner: dict = {}
    tot_ner_proba: dict = {}
    tot_rel: dict = {}
    tot_rel_proba: dict = {}

    for sent_id, sent_ents in ner_predictions.items():
        sent_rels = sorted(rel_predictions[sent_id], key=lambda x: -x[-1])

        if force_non_nil:
            # Include every candidate (all are guaranteed non-NIL).
            ents_to_output = sent_ents.items()
        else:
            # Pipeline mode: drop spans the model predicted as NIL.
            ents_to_output = [
                (span, info) for span, info in sent_ents.items() if info[0] != "NIL"
            ]

        pred_ner = sorted(span + (label,) for span, (label, _) in ents_to_output)
        # predicted_ner_proba entries: [start, end, score, label]
        # score = P(label | non-NIL classes) — used by the pre-filter threshold
        pred_ner_proba = sorted(
            span + (p_no_nil, label) for span, (label, p_no_nil) in ents_to_output
        )

        if not force_non_nil:
            # Only keep relations where both endpoints have a non-NIL NER label.
            non_nil_spans = {
                span for span, (label, _) in sent_ents.items() if label != "NIL"
            }
            sent_rels = [
                r for r in sent_rels if r[0] in non_nil_spans and r[2] in non_nil_spans
            ]
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


def prepare_input_file(
    source_file_path: str,
    tmp_file_path: str,
    candidates_from: Literal["predicted_ner", "ner"] = "predicted_ner",
    augment_with_gold: bool = False,
) -> None:
    """Write a copy of *source_file_path* with ``predicted_ner`` set according
    to the chosen candidate source.

    Args:
        source_file_path: original data file.
        tmp_file_path: destination for the prepared copy.
        candidates_from: ``"predicted_ner"`` (default) keeps the pruner output
            as-is.  ``"ner"`` replaces candidates with gold spans — use for
            gold-span / oracle evaluation.
        augment_with_gold: when True, merge pruner ``predicted_ner`` with gold
            ``ner`` spans so every gold span is seen by HGERE.  Gold spans are
            prepended so they are never truncated by ``max_ents``.  Ignored
            when ``candidates_from="ner"``.
    """
    if candidates_from == "predicted_ner" and not augment_with_gold:
        shutil.copy(source_file_path, tmp_file_path)
        return

    with open(source_file_path) as in_f, open(tmp_file_path, "w") as out_f:
        for line in in_f:
            data = json.loads(line)
            gold_ner = data.get("ner", [])
            num_sents = len(data["sentences"])

            if candidates_from == "ner":
                data["predicted_ner"] = [
                    list(gold_ner[i]) if i < len(gold_ner) else []
                    for i in range(num_sents)
                ]
            else:
                # augment_with_gold=True: gold spans first, then pruner extras
                pred_ner = data.get("predicted_ner", [])
                merged = []
                for i in range(num_sents):
                    pred = pred_ner[i] if i < len(pred_ner) else []
                    gold = gold_ner[i] if i < len(gold_ner) else []
                    gold_set = {(s[0], s[1]) for s in gold}
                    merged.append(
                        list(gold) + [s for s in pred if (s[0], s[1]) not in gold_set]
                    )
                data["predicted_ner"] = merged

            out_f.write(json.dumps(data) + "\n")
