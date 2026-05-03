from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import wandb

from ..data.config import RelationDatasetParams
from ..data.multi_dataset import MultiRelationDataset
from ..data.relation_dataset import RelationDataset
from ..data.tokenizer_utils import (
    add_dataset_cls_tokens,
    adjust_tokenizer as _adjust_tokenizer,
)
from ..hgere.evaluate import get_checkpoints
from ..hgere.train import _evaluate_multi_or_single, log_candidate_stats_to_wandb, train
from ..labels import LABELS
from ..models.hgere import MODEL_CLASSES
from ..utils import set_seed


def _log_entity_stats(
    split: str, dataset: RelationDataset, logger: logging.Logger
) -> None:
    """Log min / mean / max entity count per sentence for one dataset split.

    Also reports how many sentences have zero entity candidates, which
    indicates pruner misses that cannot be recovered by HGERE.
    """
    sizes = dataset.sizes
    n = len(sizes)
    if n == 0:
        logger.info("[%s] entity stats: 0 sentences", split)
        return
    min_ent = min(sizes)
    max_ent = max(sizes)
    mean_ent = sum(sizes) / n
    zero_ent = sum(1 for s in sizes if s == 0)
    logger.info(
        "[%s] entity count per sentence — n=%d  min=%d  mean=%.1f  max=%d"
        "  zero-entity sentences=%d (%.1f%%)",
        split,
        n,
        min_ent,
        mean_ent,
        max_ent,
        zero_ent,
        100.0 * zero_ent / n,
    )
    if wandb.run is not None:
        t = dataset.truncation_stats
        wandb.log(
            {
                f"data/{split}/ent_min": min_ent,
                f"data/{split}/ent_mean": mean_ent,
                f"data/{split}/ent_max": max_ent,
                f"data/{split}/zero_ent_pct": 100.0 * zero_ent / n,
                f"data/{split}/truncation/maxents_sents": t.n_maxents_sents,
                f"data/{split}/truncation/maxents_dropped": t.n_maxents_dropped,
                f"data/{split}/truncation/seqlen_subjects": t.n_seqlen_subjects,
                f"data/{split}/truncation/seqlen_objects": t.n_seqlen_objects,
            },
            commit=False,
        )


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

    # Multi-dataset: attach per-dataset head config to HF config so that the
    # model __init__ builds nn.ModuleDicts instead of single heads.
    datasets_cfg = getattr(args, "datasets", None)
    if datasets_cfg is not None:
        config.dataset_heads = {
            ds_entry.label_set: {
                "num_ner_labels": LABELS[ds_entry.label_set].num_ner_labels,
                "num_rel_labels": LABELS[ds_entry.label_set].num_rel_labels(
                    args.no_sym
                ),
            }
            for ds_entry in datasets_cfg
        }
    else:
        config.dataset_heads = None

    _transformers_logger = logging.getLogger("transformers.modeling_utils")
    _prev_level = _transformers_logger.level
    _transformers_logger.setLevel(logging.ERROR)
    tokenizer = tokenizer_class.from_pretrained(
        args.base_model_name_or_path, do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
        args=args,
    )
    _transformers_logger.setLevel(_prev_level)

    n_special_tokens = args.num_ner_labels * 4 + 2 if args.use_typemarker else 4
    _adjust_tokenizer(
        tokenizer=tokenizer,
        model=model,
        model_type=args.model_type,
        n_special_tokens=n_special_tokens,
        lminit=args.do_train and args.lminit,
        init_tokens=["subject", "object"],
        logger=logger,
    )

    if args.do_train and getattr(args, "use_dataset_id_token_as_cls", False):
        if datasets_cfg is None:
            raise ValueError(
                "use_dataset_id_token_as_cls requires multi-dataset mode (datasets list)"
            )
        dataset_cls_mapping = add_dataset_cls_tokens(
            tokenizer=tokenizer,
            model=model,
            model_type=args.model_type,
            dataset_names=[ds.label_set for ds in datasets_cfg],
            start_unused_idx=n_special_tokens,
            logger=logger,
        )
        # Persist the mapping in the HF model config (→ config.json) and on
        # args (→ training_args.bin) so inference can recover token IDs without
        # needing a separately-saved tokenizer.
        config.dataset_cls_token_ids = dataset_cls_mapping
        args.dataset_cls_token_ids = dataset_cls_mapping
    else:
        args.dataset_cls_token_ids = {}

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    best_f1 = 0
    best_result: dict[str, Any] = {}
    # Training
    if args.do_train:
        logger.info("TRAINING")
        logger.info("+" * 20)
        if datasets_cfg is not None:
            train_dataset, dev_dataset = _load_multi_datasets(
                datasets_cfg, tokenizer, args, logger
            )
        else:
            train_dataset = load_dataset("train", tokenizer, args, logger)
            _log_entity_stats("train", train_dataset, logger)
            dev_dataset = load_dataset("dev", tokenizer, args, logger)
            _log_entity_stats("dev", dev_dataset, logger)
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
            logger.info(f"No checkpoints available in {args.model_dir}")
        # Load datasets if needed:
        if args.eval_test:
            if datasets_cfg is not None:
                _, test_dataset = _load_multi_datasets(
                    datasets_cfg, tokenizer, args, logger, splits=("test", "test")
                )
            else:
                test_dataset = load_dataset("test", tokenizer, args, logger)
                _log_entity_stats("test", test_dataset, logger)
                log_candidate_stats_to_wandb("test", test_dataset.candidate_stats)
        if not args.do_train:
            if args.eval_train:
                if datasets_cfg is not None:
                    # Return the train dict directly (not wrapped in MultiRelationDataset)
                    # so _evaluate_multi_or_single can route per-dataset.
                    _, train_dataset = _load_multi_datasets(
                        datasets_cfg,
                        tokenizer,
                        args,
                        logger,
                        splits=("train", "train"),
                    )
                else:
                    train_dataset = load_dataset("train", tokenizer, args, logger)
            if args.eval_dev:
                if datasets_cfg is not None:
                    _, dev_dataset = _load_multi_datasets(
                        datasets_cfg, tokenizer, args, logger
                    )
                else:
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
            # Top-level aggregate keys (micro for multi-head, direct for single-head)
            _SPLIT_KEYS = {
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

            def _wandb_split_metrics(prefix: str, results: dict) -> dict:
                """Flat aggregate keys + per-dataset re+_f1 under wandb prefix."""
                out = {
                    f"{prefix}/{k}": v for k, v in results.items() if k in _SPLIT_KEYS
                }
                # Per-dataset breakdown: keys of the form "{name}/re+_f1"
                for k, v in results.items():
                    if k.endswith("/re+_f1") and k not in _SPLIT_KEYS:
                        out[f"{prefix}/{k}"] = v
                return out

            if args.eval_train:
                train_results = _evaluate_multi_or_single(
                    model,
                    train_dataset,
                    args,
                    logger,
                    persist_predictions=True,
                    prefix=global_step,
                )
                report["train"] = train_results
                if wandb.run is not None:
                    wandb.log(
                        _wandb_split_metrics("train_eval", train_results),
                        commit=False,
                    )
            # eval dev
            if args.eval_dev:
                report["dev"] = _evaluate_multi_or_single(
                    model,
                    dev_dataset,
                    args,
                    logger,
                    persist_predictions=True,
                    prefix=global_step,
                )
            # eval test
            if args.eval_test:
                test_results = _evaluate_multi_or_single(
                    model,
                    test_dataset,
                    args,
                    logger,
                    persist_predictions=True,
                    prefix=global_step,
                )
                report["test"] = test_results
                if wandb.run is not None:
                    wandb.log(
                        _wandb_split_metrics("test", test_results),
                        commit=False,
                    )

            output_test_file = os.path.join(
                getattr(args, "output_dir", None) or args.model_dir,
                f"results_{global_step}.json",
            )
            with open(output_test_file, "w") as f:
                json.dump(report, f, indent=4)

    if wandb.run is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset_for_entry(
    split: str,
    ds_entry: Any,
    tokenizer: Any,
    args: Any,
    logger: logging.Logger,
) -> RelationDataset:
    """Load one dataset split for a :class:`~gsapere.hgere.config.DatasetEntry`."""
    file_map = {
        "train": ds_entry.train_file,
        "dev": ds_entry.dev_file,
        "test": ds_entry.test_file,
    }
    file_path = Path(ds_entry.ner_prediction_dir) / file_map[split]
    logger.info(f"  [{ds_entry.label_set}] {split} file: {file_path}")
    assert os.path.isfile(file_path), (
        f"Missing {split} file for {ds_entry.label_set}: {file_path}"
    )

    labels = LABELS[ds_entry.label_set]
    batch_size = (
        args.train_batch_size
        if split == "train"
        else args.eval_batch_size * max(1, args.n_gpu)
    )
    params = RelationDatasetParams(
        max_seq_length=args.max_seq_length,
        model_type=args.model_type,
        use_typemarker=args.use_typemarker,
        no_sym=args.no_sym,
        nocross=args.nocross,
        local_rank=args.local_rank,
        split=split,
        preload=args.preload_dataset,
        pre_filter_params=getattr(args, "pre_filter_params", None),
        max_ents=args.max_ents,
        use_gold_ner=getattr(args, "use_gold_ner", False),
        dataset_id=ds_entry.label_set,
        dataset_cls_token_id=(getattr(args, "dataset_cls_token_ids", None) or {}).get(
            ds_entry.label_set
        ),
    )
    dataset = RelationDataset(
        logger=logger,
        tokenizer=tokenizer,
        labels=labels,
        file_path=file_path,
        params=params,
    )
    dataset.build(
        batch_size=batch_size,
        shuffle=args.shuffle and split == "train",
        batch_by_size=getattr(args, "batch_by_size", False),
        n_workers=getattr(args, "n_workers", 32),
        pin_memory=True,
        persistent_workers=getattr(args, "persistent_workers", False),
        seed=getattr(args, "seed", 42),
    )
    logger.info("    [%s] %s examples = %d", ds_entry.label_set, split, len(dataset))
    if dataset.sizes and all(s == 0 for s in dataset.sizes):
        logger.warning(
            "[%s] %s: all %d sentences have zero NER candidates — dataset will not "
            "contribute to training. Check that the pruner has been run on this split.",
            ds_entry.label_set,
            split,
            len(dataset.sizes),
        )
    return dataset


def _load_multi_datasets(
    datasets_cfg: Any,
    tokenizer: Any,
    args: Any,
    logger: logging.Logger,
    splits: tuple[str, str] = ("train", "dev"),
) -> tuple[Any, dict[str, RelationDataset]]:
    """Load all datasets for a list of :class:`~gsapere.hgere.config.DatasetEntry`.

    Returns:
        A 2-tuple ``(train_wrapper, eval_dict)`` where *train_wrapper* is a
        :class:`MultiRelationDataset` (or ``None`` when ``splits[0] != "train"``)
        and *eval_dict* maps dataset name → dataset for the eval split.
    """
    train_split, eval_split = splits
    train_datasets: dict[str, RelationDataset] = {}
    eval_datasets: dict[str, RelationDataset] = {}
    for ds_entry in datasets_cfg:
        if train_split == eval_split:
            # Both sides want the same split (e.g. test/test)
            ds = _load_dataset_for_entry(train_split, ds_entry, tokenizer, args, logger)
            train_datasets[ds_entry.label_set] = ds
            eval_datasets[ds_entry.label_set] = ds
        else:
            train_datasets[ds_entry.label_set] = _load_dataset_for_entry(
                train_split, ds_entry, tokenizer, args, logger
            )
            eval_datasets[ds_entry.label_set] = _load_dataset_for_entry(
                eval_split, ds_entry, tokenizer, args, logger
            )

    sampling_weights = {ds.label_set: ds.sampling_weight for ds in datasets_cfg}
    train_wrapper = MultiRelationDataset(
        datasets=train_datasets,
        sampling_temperature=getattr(args, "sampling_temperature", 0.5),
        sampling_weights=sampling_weights,
    )
    return train_wrapper, eval_datasets


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
    params = RelationDatasetParams(
        max_seq_length=args.max_seq_length,
        model_type=args.model_type,
        use_typemarker=args.use_typemarker,
        no_sym=args.no_sym,
        nocross=args.nocross,
        local_rank=args.local_rank,
        split=split,
        preload=args.preload_dataset,
        pre_filter_params=getattr(args, "pre_filter_params", None),
        max_ents=args.max_ents,
        use_gold_ner=getattr(args, "use_gold_ner", False),
    )
    dataset = RelationDataset(
        logger=logger,
        tokenizer=tokenizer,
        labels=labels,
        file_path=file_path,
        params=params,
    )
    dataset.build(
        batch_size=batch_size,
        shuffle=args.shuffle,
        batch_by_size=getattr(args, "batch_by_size", False),
        n_workers=getattr(args, "n_workers", 32),
        pin_memory=True,
        persistent_workers=getattr(args, "persistent_workers", False),
        seed=getattr(args, "seed", 42),
    )
    logger.info("  Num examples = %d", len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Tokenizer adjustment
# ---------------------------------------------------------------------------


def resolve_checkpoint(model_dir: Path, prefix: str = "checkpoint") -> Path | None:
    """Return the latest ``<prefix>-N`` subdirectory in *model_dir*, or ``None``.

    Scans *model_dir* for subdirectories whose names start with *prefix*,
    picks the one with the highest integer suffix, and returns its full path.
    Returns ``None`` when no matching subdirectory exists.
    """
    candidates = [
        p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith(prefix + "-")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("-")[-1]))


def get_last_checkpoint(args: Any, checkpoint_prefix: str) -> tuple[str | None, int]:
    """Return (checkpoint_path_str, global_step) for the latest saved checkpoint.

    Thin wrapper around :func:`resolve_checkpoint` kept for backwards
    compatibility with the training entry point.
    """
    result = resolve_checkpoint(Path(args.model_dir), prefix=checkpoint_prefix)
    if result is None:
        return None, 0
    return str(result), int(result.name.split("-")[-1])
