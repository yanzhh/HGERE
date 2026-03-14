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
    BertConfig,
    # AlbertConfig,
    # AlbertTokenizer,
    # AlbertForSpanMarkerNerPruner,
    BertTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)

from ...data.pruner_data_utils import ACEDatasetNER
from ...models.span_classifier import BertForSpanMarkerNerPruner
from ...utils import get_logger, set_seed
from .evaluate import evaluate

# eval_pruner import evaluate

ALL_MODELS = []  # pretrained_config_archive_map removed in transformers 4.x

# False positive: PyTorch emits this warning when scheduler is created before the first optimizer step,
# but our code correctly calls optimizer.step() before scheduler.step() during training.
warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)`")

MODEL_CLASSES = {
    "bertspanmarkerpruner": (BertConfig, BertForSpanMarkerNerPruner, BertTokenizer),
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
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "position_ids": batch[2],
                "labels": batch[3],
            }

            if args.model_type.find("span") != -1:
                inputs["mention_pos"] = batch[4]
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
                    metrics_to_log = {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/loss": (tr_loss - logging_loss) / args.logging_steps,
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
    train_dataset = ACEDatasetNER(
        logger=logger, tokenizer=tokenizer, file_path=train_file, args=args
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


def run_train_span_classifier(args=None):
    if args is None:
        from hgere.commands.train_span_classifier import parse_arguments

        args = parse_arguments()
    exp_path = Path(args.model_dir)
    if args.do_train:
        model_dir = Path(args.model_dir)
        if (
            not args.overwrite_model_dir
            and model_dir.exists()
            and args.do_train
            and list(model_dir.iterdir())
        ):
            raise ValueError(
                f"Model directory ({model_dir}) already exists and is not empty. Use --overwrite_model_dir to overcome."
            )

        # if args.do_train and args.local_rank in [-1, 0]:
        exp_path = create_exp_dir(
            args.output_dir,
            scripts_to_save=[],
        )
    elif not exp_path.exists():
        raise Exception(
            f"model path given by --model_dir not valid ('{exp_path}'). Are your sure you trained the model?"
        )

    logger = get_logger(args, exp_path, args.do_test)

    if args.do_train:
        args_file = model_dir / "training_args.txt"
        with args_file.open("w") as json_file:
            json.dump(vars(args), json_file, indent=4)

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

    logger.info("Experiment dir: %s", exp_path)
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.label_set == "ace":
        num_labels = 8
    elif args.label_set == "scierc":
        num_labels = 7
    elif args.label_set == "ontonotes":
        num_labels = 19
    elif args.label_set == "gsap":
        num_labels = 11
    elif args.label_set == "somd":
        num_labels = 14
    elif args.label_set == "scinlp":
        num_labels = 5
    elif args.label_set == "scier":
        num_labels = 5
    else:
        assert False  # No valid --label_set parameter given (e.g., gsap)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

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

    # from_pretrained zeros nn.Parameter objects not in the BERT checkpoint.
    # Re-initialize BiSpanRepr's biaffine weight and bias after loading.
    if hasattr(model, "span_encoder"):
        model.span_encoder.reset_parameters()

    add_special_tokens(model, args, tokenizer, logger)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    if getattr(args, "debug_overflow", False):
        from transformers.debug_utils import DebugUnderflowOverflow
        DebugUnderflowOverflow(model)
        logger.info("DebugUnderflowOverflow enabled — will print first NaN/Inf location")

    best_result = 0
    # Training
    if args.do_train:
        global_step, tr_loss, best_result = train(logger, args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        output_dir = Path(args.output_dir)
        if not output_dir.exists() and args.local_rank in [-1, 0]:
            output_dir.mkdir(parents=True)
        model_dir = Path(args.model_dir)
        if not model_dir.exists() and args.local_rank in [-1, 0]:
            model_dir.mkdir(parents=True)

        update = True
        if args.evaluate_during_training:
            dev_file = Path(args.data_dir) / args.dev_file
            results, _ = evaluate(logger, args, model, tokenizer, file_path=dev_file)
            # logger.info(f"Epoch: {epoch_num}, F1: {results['f1']}, recall: {results['recall']}")
            # logger.info(f"Epoch: {epoch_num}, ent_recall: {results['recall_score']}, R_overlap: {results['r_overlap']}")
            tn_rate = max(
                results.get("pruner/thresh/tn_rate", 0.0),
                results.get("pruner/topk/tn_rate", 0.0),
            )
            if tn_rate >= best_result:
                best_result = tn_rate
                logger.info(f"Best TN/(FP+TN):{best_result:.4f}")
            else:
                update = False

        if update:
            checkpoint_prefix = "checkpoint"
            model_dir_checkpoint = model_dir / f"{checkpoint_prefix}-{global_step}"
            model_dir_checkpoint.mkdir(parents=True, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            model_to_save.save_pretrained(model_dir_checkpoint)

            torch.save(args, model_dir_checkpoint / "training_args.bin")
            logger.info(f"Saving model checkpoint to {model_dir_checkpoint}")
            _rotate_checkpoints(logger, args, checkpoint_prefix)

        tokenizer.save_pretrained(args.model_dir)

        torch.save(args, model_dir / "training_args.bin")

    # -------------------------------------------------------------
    # test all files
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = list(
            {
                os.path.dirname(c)
                for pattern in ("pytorch_model.bin", "model.safetensors")
                for c in glob.glob(f"{args.model_dir}/**/{pattern}", recursive=True)
            }
        )
        checkpoints = sorted(checkpoints)
        if not args.eval_all_checkpoints:
            print(checkpoints)
            checkpoints = checkpoints[-1:]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        # Determine pruning config:
        #   --prune-config PATH  → load from JSON file (skip dev tuning)
        #   (default)            → estimate best config from dev set
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

            # Always write output files during do_test
            args.output_results = True

            prune_config = prune_config_override
            if prune_config is None:
                # Run dev to find the best pruning config
                dev_file = Path(args.data_dir) / args.dev_file
                if dev_file.exists():
                    logger.info(
                        "Running dev inference to determine best pruning config …"
                    )
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

            results = {}
            for file_name in (
                args.train_file,
                args.dev_file,
                args.test_file,
            ):
                test_file = Path(args.data_dir) / file_name
                split_name = test_file.name
                if test_file.exists():
                    logger.info(f"Evaluate on {split_name}")
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
                    results[split_name] = result
                else:
                    logger.info(f"{test_file} does not exist!")
            with open(output_test_file, "w") as f:
                json.dump(results, f, indent=4)


def add_special_tokens(model, args, tokenizer, logger):
    # Add special tokens to tokenizer
    if args.model_type.startswith("albert"):
        special_tokens_dict = {
            "additional_special_tokens": [f"[unused{x}]" for x in range(4)]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.lminit:
        # not roberta: BERT or ALBERTA (or SciBERT)
        if args.model_type.find("roberta") == -1:
            entity_id = tokenizer.encode("entity", add_special_tokens=False)
            assert len(entity_id) == 1
            entity_id = entity_id[0]
            mask_id = tokenizer.encode("[MASK]", add_special_tokens=False)
            assert len(mask_id) == 1
            mask_id = mask_id[0]
        else:  # Roberta: Hard Coded
            entity_id = 10014
            mask_id = 50264

        logger.info("entity_id: %d", entity_id)
        logger.info("mask_id: %d", mask_id)

        if args.model_type.startswith("albert"):
            word_embeddings = model.albert.embeddings.word_embeddings.weight.data
            word_embeddings[30000].copy_(word_embeddings[mask_id])
            word_embeddings[30001].copy_(word_embeddings[entity_id])
        elif args.model_type.startswith("roberta"):
            word_embeddings = model.roberta.embeddings.word_embeddings.weight.data
            word_embeddings[50261].copy_(word_embeddings[mask_id])  # entity
            word_embeddings[50262].data.copy_(word_embeddings[entity_id])
        else:
            word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            word_embeddings[1].copy_(word_embeddings[mask_id])
            word_embeddings[2].copy_(word_embeddings[entity_id])  # entity


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
