"""CLI entry point: train-hgere / train-hgere-by-config

``train-hgere`` (``main()``) — config-driven entry point.  Accepts either:

1. A single positional argument (config file path)::

       uv run train-hgere configs/train/gsap/train_gsap.yaml

2. A ``--config`` flag with optional per-field CLI overrides::

       uv run train-hgere --config configs/train/gsap/train_gsap.yaml
       uv run train-hgere --config configs/train/gsap/train_gsap.yaml \\
           --train_params__learning_rate 2e-5 \\
           --train_params__fp16

``train-hgere-by-config`` (``cli()``) — alias for ``main()``.

The argument parser is generated dynamically from :class:`~hgere.hgere.config.HGERETrainConfig`
(the Pydantic model), so there is no duplicate parameter definition.

Priority: CLI flags > config file > Pydantic defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import sys
from typing import Any, Optional

import torch

from ..config.cli_gen import apply_config_and_cli, build_argparser
from ..hgere.config import HGERETrainConfig
from ..hgere.train_setup import get_last_checkpoint, setup_training
from ..labels import LABELS
from ..utils import get_logger

# ---------------------------------------------------------------------------
# Field name remaps: Pydantic config name → argparse / setup_training name
# ---------------------------------------------------------------------------

#: Maps HGERETrainConfig field names to the flat attribute names expected by
#: setup_training (which was originally written against the old argparse).
_KEY_REMAP: dict[str, str] = {
    "model_dir": "output_dir",
    "base_model_name_or_path": "model_name_or_path",
    "n_iter": "iter",
}


# ---------------------------------------------------------------------------
# Config → flat namespace
# ---------------------------------------------------------------------------


def _config_to_namespace(config: HGERETrainConfig) -> argparse.Namespace:
    """Convert a validated :class:`HGERETrainConfig` to a flat :class:`argparse.Namespace`.

    1. Top-level fields (excluding ``schema_version`` and ``train_params``) are
       included as-is.
    2. ``train_params`` fields are flattened into the top level.
    3. Field names that differ between the config and ``setup_training`` are
       remapped via :data:`_KEY_REMAP`.
    4. ``config_name`` (a legacy HuggingFace arg used by ``setup_training`` but
       absent from the Pydantic config) is defaulted to ``""``.
    """
    flat: dict[str, Any] = config.model_dump(exclude={"schema_version", "train_params"})
    flat.update(config.train_params.model_dump())

    remapped: dict[str, Any] = {_KEY_REMAP.get(k, k): v for k, v in flat.items()}
    remapped.setdefault("config_name", "")
    return argparse.Namespace(**remapped)


# ---------------------------------------------------------------------------
# Main training / evaluation entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Run the full HGERE training + evaluation pipeline.

    Parameters
    ----------
    argv:
        Argument list.  Pass ``None`` to read from ``sys.argv[1:]``.
        A single element that does not start with ``-`` is treated as a config
        file path (shortcut form).  Otherwise a full argparse parser derived
        from :class:`~hgere.hgere.config.HGERETrainConfig` is used.
    """
    if argv is None:
        argv = sys.argv[1:]

    # ------------------------------------------------------------------
    # Shortcut: train-hgere config.yaml  (single positional arg)
    # ------------------------------------------------------------------
    if len(argv) == 1 and not argv[0].startswith("-"):
        try:
            config = HGERETrainConfig.from_yaml(argv[0])
        except Exception as exc:
            print(f"Error loading config file '{argv[0]}': {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # ------------------------------------------------------------------
        # Full argparse generated from the Pydantic model
        # ------------------------------------------------------------------
        parser = build_argparser(
            HGERETrainConfig,
            description=(
                "Train the HGERE model. Pass a config YAML as a positional argument "
                "(train-hgere config.yaml) or use --config PATH with optional "
                "per-field CLI overrides (e.g. --train_params__learning_rate 2e-5)."
            ),
        )
        namespace = parser.parse_args(argv)

        if namespace.config is None:
            parser.error(
                "Provide a config file as a positional argument "
                "(train-hgere config.yaml) or with --config PATH."
            )

        try:
            config = apply_config_and_cli(
                namespace,
                HGERETrainConfig,
                config_loader=lambda p: HGERETrainConfig.from_yaml(p).model_dump(),
            )
        except Exception as exc:
            logging.error("Invalid config: %s", exc)
            sys.exit(1)

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

    def save_args(args: Any, path: str, filename: str = "training_args.json") -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        args_file = os.path.join(path, filename)
        with open(args_file, "w") as f:
            json.dump(
                {k: v for k, v in vars(args).items() if _is_json_serializable(v)},
                f,
                indent=4,
            )

    def create_exp_dir(
        path: str, scripts_to_save: Optional[list[str]] = None
    ) -> Optional[str]:
        if args.output_dir.endswith("test"):
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
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        exp_path = args.output_dir
        logger = get_logger(args, exp_path, args.eval_test)
        logger.warning(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "It will continue training or use overwrite_output_dir to overcome."
        )
    elif not args.do_train:
        exp_path = args.output_dir
        assert os.path.exists(exp_path)  # no training — output_dir must contain a model
        logger = get_logger(args, exp_path, args.eval_test)

    else:
        exp_path = create_exp_dir(args.output_dir, scripts_to_save=[])
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
            saved_checkpoint if saved_checkpoint else args.model_name_or_path
        )
        args.continue_training = True
    else:
        args.model_path = args.model_name_or_path
        global_step = 0
        args.continue_training = False

    args.global_step = global_step

    setup_training(args, logger)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_json_serializable(value: Any) -> bool:
    """Return True if *value* can be serialised to JSON without error."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
