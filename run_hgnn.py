import argparse
import glob

# from transformers.modeling_albert import
import json
import logging
import math

# from multiprocessing.sharedctypes import Value
import os
import re
import shutil
import socket
import timeit
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm, trange

# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.utils.rnn import pad_sequence
import wandb

from transformers import (
    # BertForACEBothOneDropoutSub,
    AlbertConfig,
    # AlbertForACEBothOneDropoutSub,
    # AlBertForBaselines,
    #   BertForHyperGNNPlus,
    #   BertForHyperGNNwithUnifyEntity,
    # AlbertForHyperGNN,
    AlbertTokenizer,
    AutoTokenizer,
    # WEIGHTS_NAME,
    BertConfig,
    # BertForACEBothOneDropoutSubNoNer,
    # BertForBaselines,
    #   BertForAttnHyperGNN,
    # BertForHyperGNN,
    BertTokenizer,
    # RobertaConfig,
    # RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from hgere.models.hgere import BertForHyperGNN
from hgere.data.relation_dataset import RelationDataset
from hgere.labels import LABELS
from hgere.utils import get_logger, set_seed

WEIGHTS_NAME = "pytorch_model.bin"
TRAIN_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ner_labels",
    "rel_labels",
    "ent_numbers",
]
EVAL_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ent_numbers",
]

ALL_MODELS = []  # pretrained_config_archive_map removed in transformers 4.x

MODEL_CLASSES = {
    # "baseline": (BertConfig, BertForBaselines, BertTokenizer),
    # "albertbaseline": (AlbertConfig, AlBertForBaselines, AlbertTokenizer),
    "hyper": (BertConfig, BertForHyperGNN, AutoTokenizer),
    # "alberthyper": (AlbertConfig, AlbertForHyperGNN, AlbertTokenizer),
}


def train(model, train_dataset, eval_dataset, args, logger):
    """Train the model"""
    # you can also add name=run_name, config=your_config, etc.
    log_wandb = False
    if args.local_rank in [-1, 0] and args.log_wandb:
        wandb_params = dict(project=args.project_name, config=vars(args))
        if args.run_name is not None:
            wandb_params["name"] = args.run_name
        wandb.init(**wandb_params)
        log_wandb = True
        #ner_prediction_dir_name = Path(args.ner_prediction_dir).name
        #output_dir_name = Path(args.output_dir).name
        #tb_writer = SummaryWriter(
        # ner_prediction_dir_name = Path(args.ner_prediction_dir).name
        # output_dir_name = Path(args.output_dir).name
        # tb_writer = SummaryWriter(
        #    f"logs/{ner_prediction_dir_name}_re_logs/{output_dir_name}"
        # )

    train_dataloader = train_dataset.loader

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    if args.eval_epochs > 0:
        eval_steps = (
            len(train_dataloader) // args.gradient_accumulation_steps * args.eval_epochs
        )
    else:
        eval_steps = args.save_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    # ---------for span encoder---------
    if args.learning_rate_cls > 0:
        opt_grouped_bert = [
            {"params": [], "weight_decay": args.weight_decay, "lr": args.learning_rate},
            {"params": [], "weight_decay": 0.0, "lr": args.learning_rate},
        ]
        opt_grouped_cls = [
            {
                "params": [],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate_cls,
            },
            {"params": [], "weight_decay": 0.0, "lr": args.learning_rate_cls},
        ]
        for n, p in model.named_parameters():
            if "bert" not in n:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_cls[0]["params"].append(p)
                else:
                    opt_grouped_cls[1]["params"].append(p)
            else:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_bert[0]["params"].append(p)
                else:
                    opt_grouped_bert[1]["params"].append(p)
        optimizer = AdamW(opt_grouped_bert + opt_grouped_cls, eps=args.adam_epsilon)
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
    # initilize the scaler to train with float16
    scaler = GradScaler(device=args.device_name, enabled=args.fp16)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)`")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * t_total),
            num_training_steps=t_total,
        )

    if args.continue_training:
        train_states_checkpoint = torch.load(
            os.path.join(args.model_path, "train_states.bin")
        )
        past_epoch = train_states_checkpoint["epoch"]
        optimizer.load_state_dict(train_states_checkpoint["optimizer"])
        scheduler.load_state_dict(train_states_checkpoint["scheduler"])
    else:
        past_epoch = -1
    start_epoch = past_epoch + 1

    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Eval steps = %d", eval_steps)

    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs - start_epoch),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    global_step = args.global_step
    tr_loss, logging_loss = 0.0, 0.0
    tr_ner_loss, logging_ner_loss = 0.0, 0.0
    tr_re_loss, logging_re_loss = 0.0, 0.0

    best_f1 = -1
    best_result = None
    for epoch_num in train_iterator:
        epoch_num += start_epoch
        logger.info(f">>> Epoch {epoch_num} starts.")
        logging_loss_steps, logging_reloss, logging_nerloss = global_step, 0.0, 0.0

        # if args.shuffle and _ > 0:
        #     train_dataset.initialize()
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )

        for step, batch in enumerate(epoch_iterator):
            # for step, batch in enumerate(train_dataloader):

            model.train()

            inputs = {}
            input_keys = TRAIN_KEYS

            for k, v in batch.items():
                if k in input_keys:
                    v = v.to(args.device)
                    inputs[k] = v
            # Compute loss weighting alpha (RE share): static or dynamic sigmoid schedule
            if args.train_time_loss_weighting:
                t = global_step / max(t_total, 1)
                alpha = 1.0 / (
                    1.0
                    + math.exp(
                        -args.train_time_loss_steepness
                        * (t - args.train_time_loss_turn)
                    )
                )
            else:
                alpha = args.loss_re_weight_alpha

            with autocast(
                device_type=args.device_name, dtype=torch.float16, enabled=args.fp16
            ):
                outputs = model(**inputs)

                # model outputs are always tuple in pytorch-transformers (see doc)
                re_loss = outputs[1]
                ner_loss = outputs[2]

                # weight loss (original added loss is in outputs[0])
                loss = alpha * re_loss + (1 - alpha) * ner_loss

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    re_loss = re_loss / args.gradient_accumulation_steps
                    ner_loss = ner_loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                if re_loss > 0:
                    tr_re_loss += re_loss.item()
                if ner_loss > 0:
                    tr_ner_loss += ner_loss.item()

                logging_reloss += re_loss.item()
                logging_nerloss += ner_loss.item()

            scaler.scale(loss).backward()

            # t3 = timeit.default_timer()
            # logger.info(f"time for loss backward: {t3-t2}s")

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(
                train_dataloader
            ):
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                scaler.step(optimizer)

                # Only ubpdate scheduler when optimization is successfull
                old_scale = scaler.get_scale()
                scaler.update()
                new_scale = scaler.get_scale()
                if new_scale == old_scale:  # == => successfull optimization
                    scheduler.step()

                optimizer.zero_grad()
                global_step += 1

                # Log dynamic weight at every step so the full sigmoid curve is visible in wandb
                if (
                    args.train_time_loss_weighting
                    and args.local_rank in [-1, 0]
                    and log_wandb
                ):
                    wandb.log(
                        {
                            "train/loss_weight/alpha_re": alpha,
                            "train/loss_weight/alpha_ner": 1 - alpha,
                        },
                        step=global_step,
                    )

                # Log metrics
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                    and log_wandb
                ):
                    lrates = scheduler.get_last_lr()
                    lr = lrates[0]
                    lr_cls = lrates[2] if len(lrates) > 2 else lrates[0]
                    metrics_to_log = {
                        "train/lr": lr,
                        "train/lr_cls": lr_cls,
                        "train/loss": (tr_loss - logging_loss) / args.logging_steps,
                        "train/loss/re": (tr_re_loss - logging_re_loss)
                        / args.logging_steps,
                        "train/loss/ner": (tr_ner_loss - logging_ner_loss)
                        / args.logging_steps,
                        "train/loss_weight/alpha_re": alpha,
                        "train/loss_weight/alpha_ner": 1 - alpha,
                    }
                    logging_loss = tr_loss
                    logging_re_loss = tr_re_loss
                    logging_ner_loss = tr_ner_loss

                    wandb.log(metrics_to_log, step=global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            # EVALUATE AFTER EACH EPOCH
            # -------------------------
            if args.evaluate_during_training and (
                (epoch_num + 1) % args.eval_epochs == 0
                or epoch_num + 1 == args.num_train_epochs
            ):  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(model, eval_dataset, args, logger)
                f1_re_plus = results["re+_f1"]
                f1_re = results["re_f1"]
                f1_ner = results["ner_f1"]
                if log_wandb:
                    metrics_to_log = {f"eval/{k}": v for k, v in results.items()}
                    # @TODO Delete if not needed anymore:
                    metrics_to_log |= {  # Old format to keep it the same. for new projects this could be deleted.
                        "f1/train/re_strict": f1_re_plus,
                        "f1/train/re": f1_re,
                        "f1/train/ner": f1_ner,
                    }
                    wandb.log(metrics_to_log, step=global_step)

                is_best_result = f1_re_plus > best_f1
                if is_best_result:
                    best_f1 = f1_re_plus
                    best_result = results
                    logger.info(f"New Best F1+: {best_f1}")
                    # @TODO: also save optimizer, scheduler, scaler and best_f1
                    #        Then further training from a checkpoint is possible
                    _save_model(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch_num,
                        args,
                        logger,
                    )

        logger.info(f">>> current global steps: {global_step}")
        logger.info(f">>> lr of epoch {epoch_num}: {scheduler.get_last_lr()[0]:.4e}")
        dstep = global_step - logging_loss_steps
        avg_reloss = logging_reloss / dstep
        avg_nerloss = logging_nerloss / dstep
        logger.info(
            f">>> Average loss of epoch{epoch_num}: ner_{avg_nerloss:.6f}, re_{avg_reloss:.6f}"
        )

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        # tb_writer.close()
        pass

    return global_step, tr_loss / global_step, best_f1, best_result


# def to_list(tensor):
#    return tensor.detach().cpu().tolist()


def get_gold_ner_with_nolabel(ner_golden_labels):
    ner_golden_nolabels = set()
    for ner in ner_golden_labels:
        ner_nolabel = (ner[0], ner[1])
        ner_golden_nolabels.add(ner_nolabel)
    return ner_golden_nolabels


def load_dataset(split, tokenizer, args, logger):
    """
    assert split in {"train", "dev", "test"}
    """
    if split == "train":
        batch_size = args.train_batch_size
        file_path = Path(args.ner_prediction_dir) / args.train_file
    else:
        batch_size = args.eval_batch_size * max(1, args.n_gpu)
        if split == "dev":
            file_path = Path(args.ner_prediction_dir) / args.dev_file
        if split == "test":
            file_path = Path(args.ner_prediction_dir) / args.test_file
    logger.info(f"{split} file: {file_path}")
    assert os.path.isfile(file_path)

    label_set = args.label_set
    logger.info(f"    Evaluation using label set: {label_set}.")
    labels = LABELS[label_set]
    dataset = RelationDataset(
        logger=logger,
        tokenizer=tokenizer,
        labels=labels,
        file_path=file_path,
        args=args,
        max_pair_length=args.max_pair_length,
        preload=args.preload_dataset,
        doc_limit=None,
    )
    dataset.build(
        batch_size=batch_size,
        shuffle=args.shuffle,
        batch_by_size=getattr(args, "batch_by_size", False),
        n_workers=32,
        pin_memory=True,
    )
    logger.info("  Num examples = %d", len(dataset))
    return dataset


def evaluate(model, eval_dataset, args, logger, prefix="", persist_predictions=False):

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        default="hgere",
        help="project name for wandb",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="run name for wandb.",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log the training in wandb",
    )
    parser.add_argument(
        "--label_set",
        type=str,
        default=None,
        help="label set to use (e.g., gsap)",
    )
    parser.add_argument(
        "--loss_re_weight_alpha",
        type=float,
        default=0.5,
        help="Weight the re loss in respect to the ner loss. E.g., 0.7 => 0.7 re loss and 0.3 ner loss",
    )
    parser.add_argument(
        "--train_time_loss_weighting",
        action="store_true",
        help=(
            "Enable dynamic NER→RE loss weighting over training. Alpha shifts from 0.0 (full NER) "
            "to 1.0 (full RE) via a sigmoid schedule. "
            "See documentation/train_time_loss_weighting.md for details."
        ),
    )
    parser.add_argument(
        "--train_time_loss_turn",
        type=float,
        default=0.5,
        help=(
            "Fractional training progress [0, 1] at which the NER→RE weighting is at its midpoint "
            "(alpha=0.5). Default: 0.5 (centre of training)."
        ),
    )
    parser.add_argument(
        "--train_time_loss_steepness",
        type=float,
        default=10.0,
        help=(
            "Steepness of the sigmoid phase transition for dynamic loss weighting. "
            "Higher values produce a sharper switch. Default: 10.0."
        ),
    )

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--ner_prediction_dir",
        default="",
        type=str,
        required=True,
        help="NER prediction dir. Should contain the .json files (or other data files) for the task.",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_train",
        action="store_true",
        help="Whether to run eval on the train set and save the predictions.",
    )
    parser.add_argument(
        "--eval_dev",
        action="store_true",
        help="Whether to run eval on the dev set and save the predictions.",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="want to test and save the predictions.",
    )
    parser.add_argument(
        "--preload_dataset", action="store_true", help="preload dataset"
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--learning_rate_cls",
        default=-1,
        type=float,
        help="The initial learning rate for layers beyond bert.",
    )

    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=5, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=-1,
        help="Save checkpoint every eval_scale*total_steps.",
    )

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through torch.cuda.amp) instead of 32-bit",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--max_pair_length", type=int, default=64, help="")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--eval_logsoftmax", action="store_true")
    parser.add_argument("--eval_softmax", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--batch_by_size",
        action="store_true",
        help=(
            "Sort sentences by entity count before batching, then shuffle batch order. "
            "Produces batches of similar-sized sentences, reducing padding and making "
            "hypergraph iteration more uniform. Mutually exclusive with --shuffle."
        ),
    )
    parser.add_argument("--lminit", action="store_true")
    parser.add_argument("--no_sym", action="store_true")
    parser.add_argument("--att_left", action="store_true")
    parser.add_argument("--att_right", action="store_true")
    parser.add_argument("--use_ner_results", action="store_true")
    parser.add_argument("--use_typemarker", action="store_true")
    parser.add_argument("--eval_unidirect", action="store_true")
    # no cross-sentence
    parser.add_argument("--nocross", action="store_true")
    # fix evaluation

    # change warmup
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Linear warmup over warmup_steps.",
    )
    # # add for continue training
    # parser.add_argument("--load_save", action='store_true', help="load lastest saved checkpoint")

    # parser.add_argument("--use_bucket_data", action='store_true', help="sent with similar ent number in a bucket")
    # exp decoding
    parser.add_argument(
        "--eval_logits", action="store_true", help="decoding with non-normalized logits"
    )

    # add loss scale
    # parser.add_argument('--rel_loss_scale', type=float, default=-1.0, help="for BiaffineRelationCls")

    # encoder
    parser.add_argument(
        "--ent_repr",
        type=str,
        default="mix",
        help="option: sub, obj, mix. choose the source of entity representations",
    )
    parser.add_argument(
        "--uni_ent",
        action="store_true",
        help="if True, sub/obj use the same repr from bert; else bert encode sub/obj respectively",
    )
    parser.add_argument("--ent_enc", type=str, default="cat", help="entity encoder")
    parser.add_argument("--pred_sub", action="store_true", help="")
    parser.add_argument("--ner_cls", type=str, default="cat", help="")

    parser.add_argument("--rel_enc", type=str, default="cat", help="entity encoder")
    parser.add_argument(
        "--ent_dim", type=int, default=200, help="for BiaffineRelationCls"
    )
    parser.add_argument(
        "--rel_dim", type=int, default=200, help="for BiaffineRelationCls"
    )
    parser.add_argument(
        "--rel_rank", type=int, default=200, help="for BiaffineRelationCls"
    )
    parser.add_argument(
        "--rel_factorize", action="store_true", help="use BiaffineRelationCls"
    )
    # parser.add_argument('--rel_cls', type=str, default='biaf', help="")

    parser.add_argument("--baseline", type=str, default="firstorder", help="")

    # HyperGNN
    # parser.add_argument('--unirel', action='store_true', help="if true, treat two relations in hyperedge as unity node")
    # parser.add_argument('--noprojentrel', action='store_true', help="")
    parser.add_argument("--factor_type", type=str, default="ternary", help="")
    parser.add_argument(
        "--mem_dim", type=int, default=200, help="for BiaffineRelationCls"
    )
    parser.add_argument("--iter", type=int, default=3, help="for BiaffineRelationCls")
    parser.add_argument(
        "--re_focal_loss",
        action="store_true",
        help="Use focal loss for relation classification instead of CrossEntropyLoss.",
    )
    parser.add_argument(
        "--re_focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter γ for RE focal loss (only used with --re_focal_loss).",
    )
    parser.add_argument(
        "--ner_focal_loss",
        action="store_true",
        help="Use focal loss for NER classification instead of CrossEntropyLoss.",
    )
    parser.add_argument(
        "--ner_focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter γ for NER focal loss (only used with --ner_focal_loss).",
    )
    parser.add_argument("--layernorm", action="store_true", help="")
    parser.add_argument(
        "--layernorm_1st", action="store_true", help="layernorm for first order"
    )
    # parser.add_argument('--attn_type', type=str, default='attn', help="")
    # parser.add_argument('--use_graph_output_number', type=int, default=1, help="for BiaffineRelationCls")
    # debug
    parser.add_argument("--attn_self", action="store_true", help="")
    # parser.add_argument('--ter_rel_aggregator', type=str, default='cat', help="")
    parser.add_argument(
        "--aggregate_type", type=str, default="attn", help="attn or test"
    )
    parser.add_argument("--aggregate_func", type=str, default="max", help="max or sum")
    parser.add_argument("--agg_with_self", action="store_true", help="")
    parser.add_argument("--fix_obj", action="store_true", help="")

    # HTNN
    # parser.add_argument('--sep_compose', action='store_true', help="")

    parser.add_argument("--edgetype", type=str, default="sib", help="")

    # AttnHTNN composer
    parser.add_argument("--attn_scorer", type=str, default="biaf", help="")
    parser.add_argument("--attn_res", action="store_true", help="")

    # Attn GNN

    parser.add_argument("--n_head", type=int, default=8, help="for BiaffineRelationCls")
    parser.add_argument(
        "--d_head", type=int, default=32, help="for BiaffineRelationCls"
    )

    # Factor graph
    parser.add_argument(
        "--factor_encoder", type=str, default="cat", help="entity encoder"
    )

    # HyperGNN plus
    parser.add_argument("--iter1", type=int, default=1, help="for BiaffineRelationCls")
    # parser.add_argument('--edge2nodefunc', action='store_true', help="")

    args = parser.parse_args()

    # Warn if both dynamic and static weighting are configured
    if args.train_time_loss_weighting and args.loss_re_weight_alpha != 0.5:
        logging.warning(
            "--train_time_loss_weighting is enabled but --loss_re_weight_alpha is also set to %.3f "
            "(non-default value). The static --loss_re_weight_alpha will be IGNORED in favour of "
            "the dynamic sigmoid schedule. Remove --loss_re_weight_alpha to suppress this warning.",
            args.loss_re_weight_alpha,
        )

    # get hostname
    args.hostname = socket.gethostname()

    def save_args(args, path, filename="training_args.txt"):
        if not os.path.exists(path):
            os.makedirs(path)
        args_file = os.path.join(path, filename)
        with open(args_file, "w") as f:
            f.write(str(vars(args)))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return

        if not os.path.exists(path):
            os.makedirs(path)

        print("Experiment dir : {}".format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, "scripts")):
                os.mkdir(os.path.join(path, "scripts"))
            for script in scripts_to_save:
                dst_file = os.path.join(path, "scripts", os.path.basename(script))
                shutil.copyfile(script, dst_file)
        return path

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        exp_path = args.output_dir
        logger = get_logger(args, exp_path, args.eval_test)
        # raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
        logger.warning(
            f"Output directory ({args.output_dir}) already exists and is not empty. It will continue training or use --overwrite_output_dir to overcome."
        )
    elif not args.do_train:
        exp_path = args.output_dir
        assert os.path.exists(
            exp_path
        )  # no training, output_dir need to exist with a model
        logger = get_logger(args, exp_path, args.eval_test)

    else:
        # if args.do_train and args.local_rank in [-1, 0]:
        exp_path = create_exp_dir(
            args.output_dir,
            scripts_to_save=[
                os.path.basename(__file__),
<<<<<<< HEAD
               # "utils/data.py",
               # "transformers/src/transformers/modules.py",
               # "transformers/src/transformers/modeling_bert.py",
               # "transformers/src/transformers/modeling_albert.py",
=======
                "src/hgere/data/relation_dataset.py",
>>>>>>> refs/remotes/origin/standard-transformer
            ],
        )

        logger = get_logger(args, exp_path, args.eval_test)

    if args.do_train:
        save_args(args, os.path.join(exp_path, "args"), "training_args.txt")
    else:
        save_args(args, os.path.join(exp_path, "args"), "test_args.txt")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device_name = (
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        device = torch.device(device_name)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        device_name = "cuda"
        torch.cuda.set_device(args.local_rank)
        device = torch.device(device_name, args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.device_name = device_name
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}",
    )

    # Set seed
    set_seed(args)
    label_set = args.label_set
    logger.info(f"    Evaluation using label set: {label_set}.")
    assert label_set in LABELS  # Please add your labels in utils/labals.py
    labels = LABELS[label_set]
    args.num_ner_labels = labels.num_ner_labels
    args.num_rel_labels = labels.num_rel_labels(args.no_sym)
    """
    if args.label_set == "ace":
        num_ner_labels = 8  # 7 ner labels

        if args.no_sym:  # 6 relation labels
            num_labels = 7 + 7 - 1
        else:
            num_labels = 7 + 7 - 2
    elif args.label_set == "somd":
        num_ner_labels = 15  # 14 ner labels
        if args.no_sym:  #  20 relation types # old 13
            num_labels = 12 + 12 - 1  # 1 is NIL (non symetric relations) # old 14
        else:
            num_labels = 12 + 12 - 1
    # @todo have the dataset as parameter!
    elif args.label_set == "gsap":
        num_ner_labels = 11  # 10 ner labels

        if args.no_sym:  #  20 relation types # old 13
            num_labels = 21 + 21 - 1  # 1 is NIL (non symetric relations) # old 14
        else:
            num_labels = 21 + 21 - 3
    elif args.label_set == "scierc":
        num_ner_labels = 7  # 6 ner types

        if args.no_sym:  # 7 relation types
            num_labels = 8 + 8 - 1
        else:
            num_labels = 8 + 8 - 3
    elif args.label_set == "scier":
        num_ner_labels = 4  # 6 ner types

        if args.no_sym:  # 9 relation types
            num_labels = 10 + 10 - 1
        else:
            num_labels = 10 + 10 - 1
    elif args.label_set == "scinlp":
        num_ner_labels = 5  # 4 ner types

        if args.no_sym:  # 11 relation types
            num_labels = 12 + 12 - 1
        else:
            num_labels = 12 + 12 - 2
    else:
        assert False # please define --label_set (e.g., "gsap")
    """

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # for continue training
    if args.do_train and not args.overwrite_output_dir:
        saved_checkpoint, global_step = get_last_checkpoint(args, "checkpoint")
        args.model_path = (
            saved_checkpoint if saved_checkpoint else args.model_name_or_path
        )
        args.continue_training = True
    else:
        args.model_path = args.model_name_or_path
        global_step = 0
        args.continue_training = False

    args.global_step = global_step

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_path,
        num_labels=args.num_rel_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = args.num_ner_labels

    # print(config)
    # raise Exception()
    _transformers_logger = logging.getLogger("transformers.modeling_utils")
    _prev_level = _transformers_logger.level
    _transformers_logger.setLevel(logging.ERROR)
    model = model_class.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
        args=args,
    )
    _transformers_logger.setLevel(_prev_level)

    adjust_tokenizer(tokenizer, model, args.num_ner_labels, args, logger)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # logger.info(f"Training/evaluation parameters {args}")
    best_f1 = 0
    best_result = {}
    # Training
    if args.do_train:
        logger.info("TRAINING")
        logger.info("+" * 20)
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        train_dataset = load_dataset("train", tokenizer, args, logger)
        dev_dataset = load_dataset("dev", tokenizer, args, logger)
        global_step, tr_loss, best_f1, best_result = train(
            model, train_dataset, dev_dataset, args, logger
        )
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")
    else:
        logger.info("No Training")

    # Evaluation of train and/or dev and/or test
    if args.local_rank in [-1, 0]:
        logger.info("Run the models on selected splits.")
        logger.info("+" * 20)
        checkpoints = get_checkpoints(args)
        # ----------test on all dataset----------------------
        if checkpoints:
            logger.info("==========Evaluate the following checkpoints: %s", checkpoints)
        else:
            logger.info(f"No checkpoints available in {args.output_dir}")
        # Load datasets if needed:
        if args.eval_test:
            test_dataset = load_dataset("test", tokenizer, args, logger)
        if not args.do_train:
            if args.eval_train:
                train_dataset = load_dataset("train", tokenizer, args, logger)
            if args.eval_dev:
                dev_dataset = load_dataset("dev", tokenizer, args, logger)
        for checkpoint in checkpoints:
            report = {}
            if best_result:
                report["best_dev_perfomance"] = dict(
                    fn_dev=args.dev_file, result=best_result
                )
            global_step = checkpoint.split("-")[-1]
            if args.eval_test or args.eval_dev or args.eval_train:
                model = model_class.from_pretrained(
                    checkpoint, config=config, args=args
                )
                model.to(args.device)
            # eval train
            if args.eval_train:
                report[args.dev_file] = evaluate(
                    model,
                    train_dataset,
                    args,
                    logger,
                    prefix=global_step,
                    persist_predictions=True,
                )
            # eval dev
            if args.eval_dev:
                report[args.dev_file] = evaluate(
                    model,
                    dev_dataset,
                    args,
                    logger,
                    prefix=global_step,
                    persist_predictions=True,
                )
            # eval train
            if args.eval_test:
                report[args.test_file] = evaluate(
                    model,
                    test_dataset,
                    args,
                    logger,
                    prefix=global_step,
                    persist_predictions=True,
                )

            output_test_file = os.path.join(
                args.output_dir, f"results_{global_step}.json"
            )
            with open(output_test_file, "w") as f:
                json.dump(report, f, indent=4)


def get_last_checkpoint(args, checkpoint_prefix):
    """search for saved checkpoint fold"""
    checkpoints = os.listdir(args.output_dir)
    global_steps = []
    for checkpoint_fold in checkpoints:
        if checkpoint_fold.startswith(checkpoint_prefix):
            global_step = int(checkpoint_fold.split("-")[-1])
            global_steps.append(global_step)
    if len(global_steps) > 0:
        max_global_step = max(global_steps)
        saved_checkpoint = os.path.join(
            args.output_dir, f"{checkpoint_prefix}-{max_global_step}"
        )
    else:
        saved_checkpoint = None
        max_global_step = 0
    return saved_checkpoint, max_global_step


def _rotate_checkpoints(logger, args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def get_scheme_depricated(path):
    path = str(path)
    if path.find("ace05") != -1 or path.find("ace2005") != -1:
        dataset_name = "ace05"
    elif path.find("ace04") != -1 or path.find("ace2004") != -1:
        dataset_name = "ace04"
    elif path.find("somd") != -1:
        dataset_name = "somd"
    elif path.find("gsap") != -1:
        dataset_name = "gsap"
    elif path.find("scierc") != -1:
        dataset_name = "scierc"
    elif path.find("scier") != -1:
        dataset_name = "scier"
    else:
        raise Exception("Dataset unknown. Can't find labels.")
    return dataset_name


def adjust_tokenizer(tokenizer, model, num_ner_labels, args, logger):
    if args.model_type.startswith("albert"):
        if args.use_typemarker:
            special_tokens_dict = {
                "additional_special_tokens": [
                    "[unused" + str(x) + "]" for x in range(num_ner_labels * 4 + 2)
                ]
            }
        else:
            special_tokens_dict = {
                "additional_special_tokens": [
                    "[unused" + str(x) + "]" for x in range(4)
                ]
            }
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        subject_id = tokenizer.encode("subject", add_special_tokens=False)
        assert len(subject_id) == 1
        subject_id = subject_id[0]
        object_id = tokenizer.encode("object", add_special_tokens=False)
        assert len(object_id) == 1
        object_id = object_id[0]

        mask_id = tokenizer.encode("[MASK]", add_special_tokens=False)
        assert len(mask_id) == 1
        mask_id = mask_id[0]

        logger.info(
            f" subject_id = {subject_id}, object_id = {object_id}, mask_id = {mask_id}"
        )

        if args.lminit:
            if args.model_type.startswith("albert"):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                subs = 1
                sube = 2
                objs = 3
                obje = 4

            word_embeddings[subs].copy_(word_embeddings[mask_id])
            word_embeddings[sube].copy_(word_embeddings[subject_id])

            word_embeddings[objs].copy_(word_embeddings[mask_id])
            word_embeddings[obje].copy_(word_embeddings[object_id])


def _save_model(
    model,
    optimizer,
    scheduler,
    global_step,
    current_epoch,
    args,
    logger,
    checkpoint_prefix="checkpoint",
):
    output_dir = Path(args.output_dir) / f"{checkpoint_prefix}-{global_step}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    model_to_save.save_pretrained(output_dir)
    # for continue training
    train_states_checkpoint = {
        "epoch": current_epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": args,
    }
    train_status_name = os.path.join(output_dir, "train_states.bin")
    torch.save(train_states_checkpoint, train_status_name)

    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", output_dir)
    args_file = os.path.join(output_dir, "training_args.txt")
    with open(args_file, "w") as f:
        f.write(str(vars(args)))
    _rotate_checkpoints(logger, args, checkpoint_prefix)


def get_checkpoints(args):
    checkpoints = [args.output_dir]

    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(
                glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            )
        )
    return checkpoints


if __name__ == "__main__":
    main()
