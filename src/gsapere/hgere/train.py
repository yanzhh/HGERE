"""Training functions for the HGERE model.

Moved verbatim from run_hgnn.py (lines 50-57, 78-433, 1516-1532, 1535-1571, 1678-1711).
"""

from __future__ import annotations

import glob
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup

import wandb
from gsapere.data.data_types import CandidateStats
from gsapere.data.multi_dataset import MultiRelationDataset
from gsapere.hgere.evaluate import evaluate
from gsapere.utils import set_seed

TRAIN_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "sub_positions",
    "ner_labels",
    "rel_labels",
    "ent_numbers",
]

# Keys that are passed to the model but must NOT be moved to a CUDA device
# (they are plain Python scalars / strings).
_NON_TENSOR_KEYS = {"dataset_id"}


def _evaluate_multi_or_single(
    model: Any,
    eval_dataset: Any,
    args: Any,
    logger: Any,
    persist_predictions: bool = False,
    prefix: str = "",
) -> dict[str, float]:
    """Evaluate on a single or multi-dataset eval set.

    When *eval_dataset* is a ``dict[str, RelationDataset]`` or a
    :class:`~gsapere.data.multi_dataset.MultiRelationDataset`, each dataset is
    evaluated independently. Prediction files are written to
    ``<model_dir>/<dataset_name>/`` so results stay separated by dataset.

    Returns per-dataset metrics under ``{name}/{metric}`` keys plus a
    micro-averaged ``re+_f1`` (and ``re+_precision``, ``re+_recall``) computed
    from summed TP/FP/FN counts across all datasets.
    """
    import copy

    # Normalise: both dict and MultiRelationDataset expose a datasets dict
    if isinstance(eval_dataset, MultiRelationDataset):
        datasets_dict = eval_dataset.datasets
    elif isinstance(eval_dataset, dict):
        datasets_dict = eval_dataset
    else:
        datasets_dict = None

    if datasets_dict is not None:
        per_dataset: dict[str, dict[str, float]] = {}
        for name, ds in datasets_dict.items():
            # Write each dataset's predictions to <model_dir>/<dataset_name>/
            ds_args = copy.copy(args)
            ds_args.output_dir = str(Path(args.model_dir) / name)
            per_dataset[name] = evaluate(
                model,
                ds,
                ds_args,
                logger,
                prefix=prefix,
                persist_predictions=persist_predictions,
            )
        merged: dict[str, float] = {}
        for name, results in per_dataset.items():
            for k, v in results.items():
                merged[f"{name}/{k}"] = v
        # Micro re+_f1 across all datasets from summed TP/FP/FN counts.
        total_tp = sum(r["re+_tp"] for r in per_dataset.values())
        total_fp = sum(r["re+_fp"] for r in per_dataset.values())
        total_fn = sum(r["re+_fn"] for r in per_dataset.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        merged["re+_f1"] = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r) > 0
            else 0.0
        )
        merged["re+_precision"] = micro_p
        merged["re+_recall"] = micro_r
        return merged
    return evaluate(
        model,
        eval_dataset,
        args,
        logger,
        prefix=prefix,
        persist_predictions=persist_predictions,
    )


def log_candidate_stats_to_wandb(split: str, stats: CandidateStats) -> None:
    """Log pruner candidate quality stats for one split to W&B.

    No-op when W&B is not initialised (wandb.run is None).
    """
    if wandb.run is None:
        return
    if type(stats) is not dict:
        stats = {"": stats}
    for ds, ds_stats in stats.items():
        if ds:
            ds = f"/{ds}"
        wandb.log(
            {
                f"data{ds}/{split}/n_gold": ds_stats.n_gold,
                f"data{ds}/{split}/n_candidates": ds_stats.n_candidates,
                f"data{ds}/{split}/n_tp": ds_stats.n_tp,
                f"data{ds}/{split}/n_fp": ds_stats.n_fp,
                f"data{ds}/{split}/n_fn": ds_stats.n_fn,
                f"data{ds}/{split}/recall": ds_stats.recall,
                f"data{ds}/{split}/precision": ds_stats.precision,
            },
            step=0,
        )


def train(
    model: Any,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    logger: Any,
) -> tuple:
    """Train the model"""
    # you can also add name=run_name, config=your_config, etc.
    log_wandb = False
    if args.local_rank in [-1, 0] and args.log_wandb:
        wandb_params = dict(project=args.project_name, config=vars(args))
        if args.run_name is not None:
            wandb_params["name"] = args.run_name
        if args.wandb_entity is not None:
            wandb_params["entity"] = args.wandb_entity
        if args.wandb_group is not None:
            wandb_params["group"] = args.wandb_group
        wandb.init(**wandb_params)
        log_wandb = True
        if isinstance(eval_dataset, dict):
            for ds_name, ds in eval_dataset.items():
                log_candidate_stats_to_wandb(f"dev/{ds_name}", ds.candidate_stats)
        else:
            log_candidate_stats_to_wandb("dev", eval_dataset.candidate_stats)
        if hasattr(train_dataset, "candidate_stats"):
            log_candidate_stats_to_wandb("train", train_dataset.candidate_stats)
        # ner_prediction_dir_name = Path(args.ner_prediction_dir).name
        # output_dir_name = Path(args.model_dir).name
        # tb_writer = SummaryWriter(
        # ner_prediction_dir_name = Path(args.ner_prediction_dir).name
        # output_dir_name = Path(args.model_dir).name
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
            if "bert" in n:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_bert[0]["params"].append(p)
                else:
                    opt_grouped_bert[1]["params"].append(p)
            else:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_cls[0]["params"].append(p)
                else:
                    opt_grouped_cls[1]["params"].append(p)
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
        num_warmup_steps = (
            args.warmup_steps
            if args.warmup_steps != -1
            else int(args.warmup_ratio * t_total)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
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
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
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

            # Skip batches where every sentence has zero entity candidates.
            # These contribute no learning signal and the model cannot process them.
            if batch["ent_numbers"].sum() == 0:
                continue

            inputs = {}
            input_keys = TRAIN_KEYS

            for k, v in batch.items():
                if k in input_keys:
                    inputs[k] = v.to(args.device, non_blocking=True)
                elif k in _NON_TENSOR_KEYS:
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

            scaler.scale(loss).backward()

            # Accumulate loss scalars AFTER queuing backward so the CPU does not
            # block the forward→backward transition on the GPU.
            _loss_val = loss.detach().item()
            _re_loss_val = re_loss.detach().item()
            _ner_loss_val = ner_loss.detach().item()
            tr_loss += _loss_val
            if _re_loss_val > 0:
                tr_re_loss += _re_loss_val
            if _ner_loss_val > 0:
                tr_ner_loss += _ner_loss_val
            logging_reloss += _re_loss_val
            logging_nerloss += _ner_loss_val

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
                results = _evaluate_multi_or_single(model, eval_dataset, args, logger)
                f1_re_plus = results["re+_f1"]
                if log_wandb:
                    _EVAL_KEYS = {
                        "ner_precision",
                        "ner_recall",
                        "ner_f1",
                        "re_precision",
                        "re_recall",
                        "re_f1",
                        "re+_precision",
                        "re+_recall",
                        "re+_f1",
                    }
                    dstep = global_step - logging_loss_steps
                    avg_loss_re = logging_reloss / max(dstep, 1)
                    avg_loss_ner = logging_nerloss / max(dstep, 1)
                    metrics_to_log = {
                        f"eval/{k}": v for k, v in results.items() if k in _EVAL_KEYS
                    }
                    metrics_to_log["eval/loss_re"] = avg_loss_re
                    metrics_to_log["eval/loss_ner"] = avg_loss_ner
                    metrics_to_log["eval/loss"] = (
                        alpha * avg_loss_re + (1 - alpha) * avg_loss_ner
                    )
                    # Per-dataset re+_f1 keys (multi-head): log under eval/{name}/re+_f1
                    metrics_to_log |= {
                        f"eval/{k}": v
                        for k, v in results.items()
                        if k.endswith("/re+_f1") and k not in _EVAL_KEYS
                    }
                    metrics_to_log |= {
                        f"eval_detail/{k}": v
                        for k, v in results.items()
                        if k not in _EVAL_KEYS and not k.endswith("/re+_f1")
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


def _rotate_checkpoints(
    logger: Any, args: Any, checkpoint_prefix: str, use_mtime: bool = False
) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(
        os.path.join(args.model_dir, "{}-*".format(checkpoint_prefix))
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


def _save_model(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    current_epoch: int,
    args: Any,
    logger: Any,
    checkpoint_prefix: str = "checkpoint",
) -> None:
    output_dir = Path(args.model_dir) / f"{checkpoint_prefix}-{global_step}"
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

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
    args_yaml_file = os.path.join(output_dir, "training_args.yaml")
    with open(args_yaml_file, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    _rotate_checkpoints(logger, args, checkpoint_prefix)
