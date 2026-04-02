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


def _config_to_namespace(config: HGERETrainConfig) -> argparse.Namespace:
    """Convert a validated :class:`HGERETrainConfig` to a flat :class:`argparse.Namespace`.

    1. Top-level fields (excluding ``schema_version`` and ``train_params``) are
       included as-is.
    2. ``train_params`` fields are flattened into the top level.
    3. ``config_name`` (a legacy HuggingFace arg used by ``setup_training`` but
       absent from the Pydantic config) is defaulted to ``""``.
    """
    flat: dict[str, Any] = config.model_dump(exclude={"schema_version", "train_params"})
    flat.update(config.train_params.model_dump())
    flat.setdefault("config_name", "")
    if not flat.get("output_dir"):
        flat["output_dir"] = flat["model_dir"]
    return argparse.Namespace(**flat)


# ---------------------------------------------------------------------------
# Main training / evaluation entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Run the full HGERE training + evaluation pipeline.

    Parameters
    ----------
    argv:
        Argument list.  Pass ``None`` to read from ``sys.argv[1:]``.
        See module docstring for the two accepted modes (YAML file or CLI flags).
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
