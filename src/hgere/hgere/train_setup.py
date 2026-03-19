import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from transformers import (
    AutoTokenizer,
    BertConfig,
    ModernBertConfig,
)

from ..data.relation_dataset import RelationDataset
from ..hgere.evaluate import evaluate, get_checkpoints
from ..hgere.train import train
from ..labels import LABELS
from ..models.hgere import BertForHyperGNN, ModernBertForHyperGNN
from ..utils import set_seed

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


MODEL_CLASSES: dict[str, tuple[Any, Any, Any]] = {
    "hyper": (BertConfig, BertForHyperGNN, AutoTokenizer),
    "modernberthyper": (ModernBertConfig, ModernBertForHyperGNN, AutoTokenizer),
}


def setup_training(args, logger):
    # Set seed
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_path,
        num_labels=args.num_rel_labels,
    )
    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = args.num_ner_labels
    _transformers_logger = logging.getLogger("transformers.modeling_utils")
    _prev_level = _transformers_logger.level
    _transformers_logger.setLevel(logging.ERROR)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )
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

    best_f1 = 0
    best_result: dict[str, Any] = {}
    # Training
    if args.do_train:
        logger.info("TRAINING")
        logger.info("+" * 20)
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
            report: dict[str, Any] = {}
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
            # eval test
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    split: str, tokenizer: Any, args: Any, logger: logging.Logger
) -> RelationDataset:
    """Load a train/dev/test split as a :class:`RelationDataset`."""
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


# ---------------------------------------------------------------------------
# Tokenizer adjustment
# ---------------------------------------------------------------------------


def adjust_tokenizer(
    tokenizer: Any,
    model: Any,
    num_ner_labels: int,
    args: Any,
    logger: logging.Logger,
) -> None:
    """Resize the tokenizer and (optionally) initialise marker embeddings."""
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
        model.albert.resize_token_embeddings(len(tokenizer))
    elif args.model_type.startswith("modernbert"):
        # ModernBERT (BPE tokenizer) has no [unused*] slots — add them explicitly.
        n_markers = num_ner_labels * 4 + 2 if args.use_typemarker else 4
        special_tokens_dict = {
            "additional_special_tokens": [f"[unused{x}]" for x in range(n_markers)]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.bert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        mask_id = tokenizer.encode("[MASK]", add_special_tokens=False)
        assert len(mask_id) == 1
        mask_id = mask_id[0]

        if args.model_type.startswith("modernbert"):
            # BPE tokenizer: "subject"/"object" may be multi-token; fall back to [MASK].
            subject_ids = tokenizer.encode("subject", add_special_tokens=False)
            subject_id = subject_ids[0] if len(subject_ids) == 1 else mask_id
            object_ids = tokenizer.encode("object", add_special_tokens=False)
            object_id = object_ids[0] if len(object_ids) == 1 else mask_id
            logger.info(
                f" subject_id = {subject_id}, object_id = {object_id}, mask_id = {mask_id}"
            )
            if args.lminit:
                word_embeddings = model.bert.embeddings.tok_embeddings.weight.data
                subs = tokenizer.convert_tokens_to_ids("[unused0]")
                sube = tokenizer.convert_tokens_to_ids("[unused1]")
                objs = tokenizer.convert_tokens_to_ids("[unused2]")
                obje = tokenizer.convert_tokens_to_ids("[unused3]")
                word_embeddings[subs].copy_(word_embeddings[mask_id])
                word_embeddings[sube].copy_(word_embeddings[subject_id])
                word_embeddings[objs].copy_(word_embeddings[mask_id])
                word_embeddings[obje].copy_(word_embeddings[object_id])
            return

        subject_id = tokenizer.encode("subject", add_special_tokens=False)
        assert len(subject_id) == 1
        subject_id = subject_id[0]
        object_id = tokenizer.encode("object", add_special_tokens=False)
        assert len(object_id) == 1
        object_id = object_id[0]

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


def get_last_checkpoint(args: Any, checkpoint_prefix: str) -> tuple:
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
