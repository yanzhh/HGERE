"""CLI entry point: train-hgere

``train-hgere`` accepts two modes:

1. **YAML config** — single positional argument::

       uv run train-hgere configs/train/gsap/train_gsap.yaml

2. **CLI parameters** — all fields passed directly as flags::

       uv run train-hgere \\
           --model_dir saves/hgere/my_run \\
           --label_set gsap \\
           --base_model_name_or_path pretrained_models/scibert \\
           --ner_prediction_dir saves/pruner/output \\
           --train_params__learning_rate 2e-5 \\
           --train_params__num_train_epochs 10

The argument parser is generated dynamically from
:class:`~hgere.hgere.config.HGERETrainConfig` (the Pydantic model),
so there is no duplicate parameter definition.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from ..hgere.config import HGERETrainConfig
from ..hgere.train_setup import get_last_checkpoint, setup_training
from ..labels import LABELS
from ..utils import get_logger
from ._cli_utils import load_config_from_argv, save_args

# ---------------------------------------------------------------------------
# Config → flat namespace
# ---------------------------------------------------------------------------


def _resolve_seeds(config: HGERETrainConfig) -> list[int]:
    """Return ``config.seeds`` as a list of ints."""
    seeds = config.seeds
    return [seeds] if isinstance(seeds, int) else list(seeds)


def _apply_seed_suffix(
    config: HGERETrainConfig, seed: int, suffix: str
) -> HGERETrainConfig:
    """Return a copy of *config* with *suffix* appended to model_dir, run_name, output_dir.

    Also sets wandb_group to the base run_name (before suffix) when not already configured,
    so all seed runs appear under the same W&B group.
    """
    run_name = config.train_params.run_name
    output_dir = config.train_params.output_dir
    base_name = run_name if run_name is not None else Path(config.model_dir).name
    wandb_group = config.train_params.wandb_group or base_name
    return config.model_copy(
        update={
            "seeds": seed,
            "model_dir": f"{config.model_dir}{suffix}",
            "train_params": config.train_params.model_copy(
                update={
                    "run_name": f"{run_name}{suffix}" if run_name is not None else None,
                    "output_dir": f"{output_dir}{suffix}"
                    if output_dir is not None
                    else None,
                    "wandb_group": wandb_group,
                }
            ),
        }
    )


def _config_to_namespace(config: HGERETrainConfig) -> argparse.Namespace:
    """Convert a validated :class:`HGERETrainConfig` to a flat :class:`argparse.Namespace`.

    1. Top-level fields (excluding ``schema_version``, ``train_params``,
       ``datasets``, and ``seeds``) are included as plain dicts via ``model_dump``.
    2. ``train_params`` fields are flattened into the top level.
    3. ``seeds`` is resolved to a single ``seed`` int (first/only element).
    4. ``datasets`` is preserved as its Pydantic list so callers can
       access ``ds_entry.name`` etc. as attributes.
    5. ``config_name`` (a legacy HuggingFace arg used by ``setup_training`` but
       absent from the Pydantic config) is defaulted to ``""``.
    """
    flat: dict[str, Any] = config.model_dump(
        exclude={"schema_version", "train_params", "datasets", "seeds"}
    )
    flat.update(config.train_params.model_dump())
    flat.setdefault("config_name", "")
    if not flat.get("output_dir"):
        flat["output_dir"] = flat["model_dir"]
    if flat.get("run_name") is None:
        flat["run_name"] = Path(flat["model_dir"]).name
    if not flat.get("wandb_group"):
        flat["wandb_group"] = flat["run_name"]
    # Preserve datasets as the original Pydantic list so downstream code can
    # access DatasetEntry attributes directly.
    flat["datasets"] = config.datasets
    # Resolve seeds to the single runtime seed value.
    flat["seed"] = _resolve_seeds(config)[0]
    return argparse.Namespace(**flat)


# ---------------------------------------------------------------------------
# Main training / evaluation entry point
# ---------------------------------------------------------------------------


def _run_single_seed(config: HGERETrainConfig) -> None:
    """Run training/inference for a single resolved *config* (one seed, paths finalised)."""
    args = _config_to_namespace(config)

    # Warn if both dynamic and static loss weighting are configured
    if args.train_time_loss_weighting and args.loss_re_weight_alpha != 0.5:
        logging.warning(
            "train_time_loss_weighting is enabled but loss_re_weight_alpha is also set "
            "to %.3f (non-default). The static alpha will be IGNORED in favour of the "
            "dynamic sigmoid schedule. Remove loss_re_weight_alpha to suppress this warning.",
            args.loss_re_weight_alpha,
        )

    # Get hostname
    args.hostname = socket.gethostname()

    def create_exp_dir(
        path: str, scripts_to_save: Optional[list[str]] = None
    ) -> Optional[str]:
        if args.model_dir.endswith("test"):
            return None

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
        os.path.exists(args.model_dir)
        and os.listdir(args.model_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        exp_path = args.model_dir
        logger = get_logger(args, exp_path, args.eval_test)
        logger.warning(
            f"Output directory ({args.model_dir}) already exists and is not empty. "
            "It will continue training or use overwrite_output_dir to overcome."
        )
    elif not args.do_train:
        exp_path = args.model_dir
        assert os.path.exists(exp_path)  # no training — output_dir must contain a model
        logger = get_logger(args, exp_path, args.eval_test)

    else:
        # Fresh training run: remove any stale checkpoint-* subdirectories so
        # get_checkpoints doesn't find leftovers from a previous run.
        if os.path.exists(args.model_dir) and args.overwrite_output_dir:
            for entry in Path(args.model_dir).iterdir():
                if entry.is_dir() and entry.name.startswith("checkpoint-"):
                    shutil.rmtree(entry)
        exp_path = create_exp_dir(args.model_dir, scripts_to_save=[]) or args.model_dir
        logger = get_logger(args, exp_path, args.eval_test)

    if args.do_train:
        save_args(args, os.path.join(exp_path, "args"), "training_args.txt")
    else:
        save_args(args, os.path.join(exp_path, "args"), "test_args.txt")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
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
    else:  # Initialises the distributed backend for multi-GPU/node training
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
        f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}",
    )

    if args.datasets is not None:
        # Multi-dataset mode: derive num_labels from the first dataset entry.
        # These are only used as baseline values for the HF config object;
        # per-dataset head sizes are stored in config.dataset_heads.
        _first_entry = args.datasets[0]
        labels = LABELS[_first_entry.label_set]
        args.label_set = _first_entry.label_set  # fallback for log messages
        logger.info(
            f"    Multi-dataset mode. Primary label set (for HF config): {args.label_set}."
        )
    else:
        label_set = args.label_set
        logger.info(f"    Evaluation using label set: {label_set}.")
        assert label_set in LABELS  # Please add your labels in utils/labels.py
        labels = LABELS[label_set]
    args.num_ner_labels = labels.num_ner_labels
    args.num_rel_labels = labels.num_rel_labels(args.no_sym)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    # Resume from last checkpoint if continuing training
    if args.do_train and not args.overwrite_output_dir:
        saved_checkpoint, global_step = get_last_checkpoint(args, "checkpoint")
        args.model_path = (
            saved_checkpoint if saved_checkpoint else args.base_model_name_or_path
        )
        args.continue_training = True
    else:
        args.model_path = args.base_model_name_or_path
        global_step = 0
        args.continue_training = False

    args.global_step = global_step

    setup_training(args, logger)


def main(argv: Optional[list[str]] = None) -> None:
    """Run the full HGERE training + evaluation pipeline.

    Parameters
    ----------
    argv:
        Argument list.  Pass ``None`` to read from ``sys.argv[1:]``.
        See module docstring for the two accepted modes (YAML file or CLI flags).
        ``seeds`` may be a single integer or a list; when a list is given the
        pipeline runs once per seed and ``_seed<n>`` is appended to
        ``model_dir``, ``run_name``, and ``output_dir``.
    """
    if argv is None:
        argv = sys.argv[1:]

    config = load_config_from_argv(
        argv,
        HGERETrainConfig,
        description=(
            "Train the HGERE model. Pass a YAML config as a positional argument "
            "(train-hgere config.yaml) or supply all parameters directly as CLI "
            "flags (e.g. --model_dir saves/hgere --train_params__learning_rate 2e-5)."
        ),
    )

    seeds = _resolve_seeds(config)
    multiple = len(seeds) > 1
    for seed in seeds:
        suffix = f"_seed{seed}" if multiple else ""
        seed_config = _apply_seed_suffix(config, seed, suffix) if multiple else config
        _run_single_seed(seed_config)


cli = main
