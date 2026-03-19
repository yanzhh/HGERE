"""CLI commands: train-hgere / train-hgere-by-config

``train-hgere`` (``main()``) — flat argparse entry point; accepts all flags directly.

``train-hgere-by-config`` (``cli()``) — config-driven wrapper; loads a YAML
:class:`~hgere.hgere.config.HGERETrainConfig` and delegates to ``main()``.
``main()`` accepts an optional ``argv`` list so it can also be called from
``run_hgnn.py`` for backwards compatibility.

Usage — config file only
------------------------
    uv run train-hgere-by-config --config configs/train/gsap/train_gsap.yaml

Usage — override individual values via CLI
------------------------------------------
    uv run train-hgere-by-config \\
        --config configs/train/gsap/train_gsap.yaml \\
        --train_params__learning_rate 2e-5 \\
        --train_params__fp16

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
from ..hgere.train_setup import MODEL_CLASSES, get_last_checkpoint, setup_training
from ..labels import LABELS
from ..utils import get_logger

# ---------------------------------------------------------------------------
# Main training / evaluation entry point (argparse-based)
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Run the full HGERE training + evaluation pipeline.

    Parameters
    ----------
    argv:
        Argument list.  Pass ``None`` to read from ``sys.argv`` (default).
        Pass an explicit list to call programmatically (e.g. from ``cli()``).
    """
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
        help="Path to pre-trained model or shortcut name.",
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
    parser.add_argument("--nocross", action="store_true")

    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--eval_logits", action="store_true", help="decoding with non-normalized logits"
    )

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

    parser.add_argument("--baseline", type=str, default="firstorder", help="")

    # HyperGNN
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
    parser.add_argument("--attn_self", action="store_true", help="")
    parser.add_argument(
        "--aggregate_type", type=str, default="attn", help="attn or test"
    )
    parser.add_argument("--aggregate_func", type=str, default="max", help="max or sum")
    parser.add_argument("--agg_with_self", action="store_true", help="")
    parser.add_argument("--fix_obj", action="store_true", help="")

    parser.add_argument("--edgetype", type=str, default="sib", help="")

    # AttnHTNN composer
    parser.add_argument("--attn_scorer", type=str, default="biaf", help="")
    parser.add_argument("--attn_res", action="store_true", help="")

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

    args = parser.parse_args(argv)

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

    def save_args(args: Any, path: str, filename: str = "training_args.json") -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        args_file = os.path.join(path, filename)
        with open(args_file, "w") as f:
            json.dump(vars(args), f, indent=4)

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
            f"Output directory ({args.output_dir}) already exists and is not empty. It will continue training or use --overwrite_output_dir to overcome."
        )
    elif not args.do_train:
        exp_path = args.output_dir
        assert os.path.exists(
            exp_path
        )  # no training, output_dir need to exist with a model
        logger = get_logger(args, exp_path, args.eval_test)

    else:
        exp_path = create_exp_dir(
            args.output_dir,
            scripts_to_save=[
                # os.path.basename(__file__),
            ],
        )

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

    label_set = args.label_set
    logger.info(f"    Evaluation using label set: {label_set}.")
    assert label_set in LABELS  # Please add your labels in utils/labals.py
    labels = LABELS[label_set]
    args.num_ner_labels = labels.num_ner_labels
    args.num_rel_labels = labels.num_rel_labels(args.no_sym)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

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

    setup_training(args, logger)


# ---------------------------------------------------------------------------
# Pydantic CLI entry point (train-hgere-by-config)
# ---------------------------------------------------------------------------


def cli() -> None:
    parser = build_argparser(
        HGERETrainConfig,
        description="Train the HGERE model. Parameters are loaded from a YAML config "
        "and can be overridden with individual CLI flags.",
    )
    namespace = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    if namespace.config is None:
        parser.error("--config is required.")

    try:
        config = apply_config_and_cli(
            namespace,
            HGERETrainConfig,
            config_loader=lambda p: HGERETrainConfig.from_yaml(p).model_dump(),
        )
    except Exception as exc:
        logger.error("Invalid config: %s", exc)
        sys.exit(1)

    main(model_to_argv(config))


# ---------------------------------------------------------------------------
# Boolean store_true flags (used by model_to_argv)
# ---------------------------------------------------------------------------

_BOOL_FLAGS: frozenset[str] = frozenset(
    {
        "do_train",
        "eval_train",
        "eval_dev",
        "eval_test",
        "do_lower_case",
        "log_wandb",
        "fp16",
        "evaluate_during_training",
        "eval_all_checkpoints",
        "overwrite_output_dir",
        "no_cuda",
        "overwrite_cache",
        "attn_self",
        "ner_focal_loss",
        "re_focal_loss",
        "layernorm",
        "layernorm_1st",
        "no_sym",
        "lminit",
        "nocross",
        "shuffle",
        "batch_by_size",
        "save_results",
        "no_test",
        "att_left",
        "att_right",
        "use_ner_results",
        "use_typemarker",
        "eval_unidirect",
        "rel_factorize",
        "uni_ent",
        "pred_sub",
        "agg_with_self",
        "fix_obj",
        "attn_res",
        "eval_logits",
        "eval_logsoftmax",
        "eval_softmax",
        "preload_dataset",
        "train_time_loss_weighting",
    }
)

# Pydantic field name → argparse flag name (only where they differ).
_KEY_REMAP: dict[str, str] = {
    "model_dir": "output_dir",
    "base_model_name_or_path": "model_name_or_path",
    "n_iter": "iter",
}


# ---------------------------------------------------------------------------
# Pydantic config → argparse argv
# ---------------------------------------------------------------------------


def model_to_argv(config: HGERETrainConfig) -> list[str]:
    """Translate a validated :class:`HGERETrainConfig` to a flat argv list
    accepted by :func:`main`.

    1. Flattens ``train_params`` into the top-level dict.
    2. Remaps field names that differ between the config and the argparser.
    3. Converts booleans to ``store_true`` style flags.
    """
    flat: dict[str, Any] = config.model_dump(exclude={"schema_version", "train_params"})
    flat.update(config.train_params.model_dump())

    argv: list[str] = []
    for key, value in flat.items():
        if value is None:
            continue
        flag = _KEY_REMAP.get(key, key)
        if key in _BOOL_FLAGS:
            if value:
                argv.append(f"--{flag}")
        else:
            argv.extend([f"--{flag}", str(value)])
    return argv
