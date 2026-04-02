# coding=utf-8
# from enum import global_str
import glob
import json
import logging
import os
import re
import shutil
import warnings
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import wandb
from transformers import (
    AutoTokenizer,
    BertConfig,
    # AlbertConfig,
    # AlbertTokenizer,
    # AlbertForSpanMarkerNerPruner,
    BertTokenizerFast,
    ModernBertConfig,
    get_linear_schedule_with_warmup,
)

from ..data.collators import PrunerCollator
from ..data.pruner_dataset import PrunerDataset
from ..data.tokenizer_utils import adjust_tokenizer
from ..models.span_classifier import (
    BertForSpanMarkerNerPruner,
    ModernBertForSpanMarkerNerPruner,
)
from ..utils import get_logger, set_seed
from .evaluate import evaluate

# eval_pruner import evaluate

ALL_MODELS = []  # pretrained_config_archive_map removed in transformers 4.x

# False positive: PyTorch emits this warning when scheduler is created before the first optimizer step,
# but our code correctly calls optimizer.step() before scheduler.step() during training.
warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)`")

MODEL_CLASSES = {
    "bertspanmarkerpruner": (BertConfig, BertForSpanMarkerNerPruner, BertTokenizerFast),
    "modernbertspanmarkerpruner": (
        ModernBertConfig,
        ModernBertForSpanMarkerNerPruner,
        AutoTokenizer,
    ),
    # "albertspanmarkerpruner": (
    #    AlbertConfig,
    #    AlbertForSpanMarkerNerPruner,
    #    AlbertTokenizer,
    # ),
}

# NEG_INF = -1e30


def train(logger, args, model, tokenizer):
    train_sampler, train_data_loader = load_data(args, tokenizer, logger)
    len_train = len(train_data_loader)
    model, optimizer, scaler, scheduler = setup_training(args, model, len_train, logger)

    # Train!

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_result = -1

    for epoch_num in train_iterator:
        logger.info(f"------------------Epoch {epoch_num} starts.-----------------")
        logging_loss_steps_0, logging_loss_epoch = global_step, 0.0

        # if _ > 0 and (args.shuffle or args.group_edge or args.group_sort):
        #     train_dataset.initialize()
        #     if args.group_edge:
        #         train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        #         train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))

        epoch_iterator = tqdm(
            train_data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(
                t.to(args.device) if isinstance(t, torch.Tensor) else t for t in batch
            )

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "position_ids": batch[2],
                "labels": batch[3],
            }

            if args.model_type.find("span") != -1:
                inputs["mention_pos"] = batch[4]
            if (
                args.model_type.startswith("modernbert")
                and args.model_type.find("span") != -1
            ):
                inputs["sent_subword_length"] = batch[5]
            with autocast(
                device_type=args.device_name, dtype=torch.float16, enabled=args.fp16
            ):
                outputs = model(**inputs)
                loss = outputs[
                    0
                ]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                logging_loss_epoch += loss.item()
                tr_loss += loss.item()

            scaler.scale(loss).backward()
            # loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(
                train_data_loader
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

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    encoder_norm = (
                        sum(
                            p.grad.norm().item() ** 2
                            for n, p in model.named_parameters()
                            if p.grad is not None and n.startswith("bert.")
                        )
                        ** 0.5
                    )
                    head_norm = (
                        sum(
                            p.grad.norm().item() ** 2
                            for n, p in model.named_parameters()
                            if p.grad is not None and not n.startswith("bert.")
                        )
                        ** 0.5
                    )
                    metrics_to_log = {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/loss": (tr_loss - logging_loss) / args.logging_steps,
                        "train/grad_norm_encoder": encoder_norm,
                        "train/grad_norm_head": head_norm,
                    }
                    wandb.log(metrics_to_log, global_step)
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.eval_steps > 0
                    and global_step % args.eval_steps == 0
                ):
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        dev_file = Path(args.data_dir) / args.dev_file
                        results, _ = evaluate(
                            logger, args, model, tokenizer, file_path=dev_file
                        )
                        metrics_to_log = {f"dev/{k}": v for k, v in results.items()}
                        wandb.log(metrics_to_log, global_step)
                        tn_rate = max(
                            results.get("pruner/thresh/tn_rate", 0.0),
                            results.get("pruner/topk/tn_rate", 0.0),
                        )

                        if tn_rate >= best_result:
                            best_result = tn_rate
                            logger.info(
                                f"Best TN/(FP+TN):{best_result:.4f} in step:{global_step}"
                            )
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = "checkpoint"
                        model_dir = (
                            Path(args.model_dir) / f"{checkpoint_prefix}-{global_step}"
                        )
                        if not model_dir.exists():
                            model_dir.mkdir(parents=True, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(model_dir)

                        torch.save(args, os.path.join(model_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", model_dir)

                        _rotate_checkpoints(logger, args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        logger.info(f">>> lr of epoch{epoch_num}: {scheduler.get_last_lr()[0]:.4e}")
        dstep = global_step - logging_loss_steps_0
        avg_loss = logging_loss_epoch / dstep
        logger.info(f">>> Average loss of epoch{epoch_num}: {avg_loss:.4f}")

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_result


def load_data(args, tokenizer, logger):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_file = Path(args.data_dir) / args.train_file
    logger.info(f"Train file: {train_file.name}")
    train_dataset = PrunerDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_pair_length=args.max_pair_length,
        max_mention_ori_length=args.max_mention_ori_length,
        model_type=args.model_type,
        label_set=args.label_set,
        rulebased_pruner_file=getattr(args, "rulebased_pruner_file", None),
        shuffle=getattr(args, "shuffle", False),
        group_sort=getattr(args, "group_sort", False),
        group_edge=getattr(args, "group_edge", False),
        group_axis=getattr(args, "group_axis", -1),
        nocross=getattr(args, "nocross", False),
    )
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_data_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=PrunerCollator(),
        num_workers=1,
    )
    return train_sampler, train_data_loader


def setup_training(args, model, len_train, logger):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        wandb_params = dict(project=args.project_name, config=vars(args))
        if args.run_name is not None:
            wandb_params["name"] = args.run_name
        wandb.init(**wandb_params)
        wandb.watch(model, log="gradients", log_freq=50)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len_train // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len_train // args.gradient_accumulation_steps * args.num_train_epochs

    if args.eval_epochs > 0:
        args.eval_steps = (
            len_train // args.gradient_accumulation_steps * args.eval_epochs
        )
    else:
        args.eval_steps = args.save_steps

    # ---------for span encoder---------
    optimizer = get_span_optimizer(model.named_parameters(), args)
    scaler = GradScaler(device=args.device_name, enabled=args.fp16)
    num_warmup_steps = (
        args.warmup_steps if args.warmup_steps != -1 else int(0.1 * t_total)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)`")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

    # multi-gpu training (should be after fp16 initialization with GradScaler)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after fp16 initialization with GradScaler)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len_train)
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
    logger.info("  Eval steps = %d", args.eval_steps)

    return model, optimizer, scaler, scheduler


def get_span_optimizer(model_named_parameters, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    is_span_model = args.learning_rate_span > 0
    params_span_no_decay = []
    params_span = []
    params_bert_no_decay = []
    params_bert = []
    params_default_no_decay = []
    params_default = []
    for n, p in model_named_parameters:
        is_no_decay = any(nd in n for nd in no_decay)
        is_span_encoder_param = "span_encoder" in n or "ner_classifier" in n
        if is_span_model and is_span_encoder_param and is_no_decay:
            params_span_no_decay.append(p)
        elif is_span_model and is_span_encoder_param:
            params_span.append(p)
        elif is_span_model and is_no_decay:
            params_bert_no_decay.append(p)
        elif is_span_model:
            params_bert.append(p)
        elif is_no_decay:
            params_default_no_decay.append(p)
        else:
            params_default.append(p)
    grouped_params = []
    if is_span_model:
        grouped_params = [
            dict(
                params=params_span,
                weight_decay=args.weight_decay,
                lr=args.learning_rate,
            ),
            dict(params=params_span_no_decay, weight_decay=0.0, lr=args.learning_rate),
            dict(
                params=params_bert,
                weight_decay=args.weight_decay,
                lr=args.learning_rate,
            ),
            dict(params=params_bert_no_decay, weight_decay=0.0, lr=args.learning_rate),
        ]
    else:
        grouped_params = [
            dict(
                params=params_default,
                weight_decay=args.weight_decay,
                lr=args.learning_rate,
            ),
            dict(
                params=params_default_no_decay, weight_decay=0.0, lr=args.learning_rate
            ),
        ]
    optimizer = AdamW(grouped_params, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer


def _setup_device(args) -> None:
    """Configure CUDA device and distributed training on args in-place."""
    if args.local_rank == -1 or args.no_cuda:
        device_name = (
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        device = torch.device(device_name)
        args.n_gpu = torch.cuda.device_count()
    else:
        device_name = "cuda"
        torch.cuda.set_device(args.local_rank)
        device = torch.device(device_name, args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.device_name = device_name


def _get_num_labels(label_set: str) -> int:
    """Return the number of NER labels for a given label set name."""
    label_counts = {
        "ace": 8,
        "scierc": 7,
        "ontonotes": 19,
        "gsap": 11,
        "somd": 14,
        "scinlp": 5,
        "scier": 5,
    }
    if label_set not in label_counts:
        raise ValueError(f"No valid --label_set parameter given: '{label_set}'")
    return label_counts[label_set]


def _load_model_and_tokenizer(args, num_labels: int, logger):
    """Load pretrained config, tokenizer, and model; add special tokens.

    Returns (model, config, tokenizer, model_class).
    """
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.base_model_name_or_path,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.base_model_name_or_path, do_lower_case=args.do_lower_case
    )
    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.onedropout = args.onedropout
    config.use_full_layer = args.use_full_layer

    _transformers_logger = logging.getLogger("transformers.modeling_utils")
    _prev_level = _transformers_logger.level
    _transformers_logger.setLevel(logging.ERROR)
    model = model_class.from_pretrained(
        args.base_model_name_or_path,
        from_tf=bool(".ckpt" in args.base_model_name_or_path),
        config=config,
        args=args,
    )
    _transformers_logger.setLevel(_prev_level)

    if hasattr(model, "span_encoder"):
        model.span_encoder.reset_parameters()

    adjust_tokenizer(
        tokenizer=tokenizer,
        model=model,
        model_type=args.model_type,
        n_special_tokens=2,
        lminit=args.do_train and args.lminit,
        init_tokens=["entity"],
        logger=logger,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    return model, config, tokenizer, model_class


def _save_checkpoint_after_training(
    args, model, tokenizer, global_step: int, best_result: float, logger
) -> None:
    """Evaluate on dev (if requested) and save model checkpoint."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    update = True
    if args.evaluate_during_training:
        dev_file = Path(args.data_dir) / args.dev_file
        results, _ = evaluate(logger, args, model, tokenizer, file_path=dev_file)
        tn_rate = max(
            results.get("pruner/thresh/tn_rate", 0.0),
            results.get("pruner/topk/tn_rate", 0.0),
        )
        if tn_rate >= best_result:
            logger.info(f"Best TN/(FP+TN):{tn_rate:.4f}")
        else:
            update = False

    if update:
        checkpoint_prefix = "checkpoint"
        model_dir_checkpoint = model_dir / f"{checkpoint_prefix}-{global_step}"
        model_dir_checkpoint.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(model_dir_checkpoint)
        torch.save(args, model_dir_checkpoint / "training_args.bin")
        logger.info(f"Saving model checkpoint to {model_dir_checkpoint}")
        _rotate_checkpoints(logger, args, checkpoint_prefix)

    tokenizer.save_pretrained(args.model_dir)
    torch.save(args, model_dir / "training_args.bin")


def _run_test_splits(args, model_class, config, tokenizer, logger) -> None:
    """Load each checkpoint and evaluate on train/dev/test splits."""
    checkpoints = sorted(
        {
            os.path.dirname(c)
            for pattern in ("pytorch_model.bin", "model.safetensors")
            for c in glob.glob(f"{args.model_dir}/**/{pattern}", recursive=True)
        }
    )
    if not args.eval_all_checkpoints:
        checkpoints = checkpoints[-1:]
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    prune_config_override = None
    if args.prune_config:
        prune_config_override = json.loads(Path(args.prune_config).read_text())
        logger.info(
            f"Using prune config from {args.prune_config}: {prune_config_override}"
        )

    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1]
        output_test_file = Path(args.model_dir) / "test_results.txt"

        model = model_class.from_pretrained(checkpoint, config=config, args=args)
        model.to(args.device)
        args.output_results = True

        prune_config = prune_config_override
        if prune_config is None:
            dev_file = Path(args.data_dir) / args.dev_file
            if dev_file.exists():
                logger.info("Running dev inference to determine best pruning config …")
                _, prune_config = evaluate(
                    logger,
                    args,
                    model,
                    tokenizer,
                    file_path=dev_file,
                    prefix=global_step,
                    do_test=True,
                )
                config_path = Path(args.output_dir) / "best_config.json"
                config_path.write_text(json.dumps(prune_config, indent=2) + "\n")
                logger.info(
                    f"Best prune config: {prune_config}  (saved to {config_path})"
                )

        splits = []
        if getattr(args, "eval_train", True):
            splits.append(args.train_file)
        if getattr(args, "eval_dev", True):
            splits.append(args.dev_file)
        if getattr(args, "eval_test", True):
            splits.append(args.test_file)

        results = {}
        for file_name in splits:
            test_file = Path(args.data_dir) / file_name
            if test_file.exists():
                logger.info(f"Evaluate on {test_file.name}")
                result, _ = evaluate(
                    logger,
                    args,
                    model,
                    tokenizer,
                    file_path=test_file,
                    prefix=global_step,
                    do_test=True,
                    prune_config=prune_config,
                )
                results[test_file.name] = result
            else:
                logger.info(f"{test_file} does not exist!")

        with open(output_test_file, "w") as f:
            json.dump(results, f, indent=4)


def run_train_pruner(args=None):
    if args is None:
        from gsapere.commands.train_pruner import parse_arguments

        args = parse_arguments()

    exp_path = Path(args.model_dir)
    if args.do_train:
        model_dir = Path(args.model_dir)
        if (
            not args.overwrite_model_dir
            and model_dir.exists()
            and list(model_dir.iterdir())
        ):
            raise ValueError(
                f"Model directory ({model_dir}) already exists and is not empty. "
                "Use --overwrite_model_dir to overcome."
            )
        exp_path = create_exp_dir(args.output_dir, scripts_to_save=[])
    elif not exp_path.exists():
        raise Exception(
            f"model path given by --model_dir not valid ('{exp_path}'). "
            "Are your sure you trained the model?"
        )

    logger = get_logger(args, exp_path, args.do_test)

    if args.do_train:
        args_file = Path(args.model_dir) / "training_args.txt"
        with args_file.open("w") as json_file:
            json.dump(vars(args), json_file, indent=4)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    _setup_device(args)
    logger.info("Experiment dir: %s", exp_path)
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    set_seed(args)

    num_labels = _get_num_labels(args.label_set)
    model, config, tokenizer, model_class = _load_model_and_tokenizer(
        args, num_labels, logger
    )
    model.to(args.device)

    if getattr(args, "debug_overflow", False):
        from transformers.debug_utils import DebugUnderflowOverflow

        DebugUnderflowOverflow(model)
        logger.info(
            "DebugUnderflowOverflow enabled — will print first NaN/Inf location"
        )

    best_result = 0
    if args.do_train:
        global_step, tr_loss, best_result = train(logger, args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        _save_checkpoint_after_training(
            args, model, tokenizer, global_step, best_result, logger
        )

    if args.do_test and args.local_rank in [-1, 0]:
        _run_test_splits(args, model_class, config, tokenizer, logger)


def _rotate_checkpoints(logger, args, checkpoint_prefix, use_mtime=False):
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


def create_exp_dir(path, scripts_to_save=None):
    """
    Create path and copy scripts
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if scripts_to_save is not None:
        script_path = path / "scripts"
        if not script_path.exists():
            script_path.mkdir(parents=True, exist_ok=True)
        for script in scripts_to_save:
            dst_file = script_path / Path(script).name
            shutil.copyfile(script, dst_file)
    return path
