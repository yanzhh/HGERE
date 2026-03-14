import argparse
import socket

from hgere.span_classifier.neural.train import MODEL_CLASSES as _MODEL_CLASSES
from hgere.span_classifier.neural.train import run_train_span_classifier

MODEL_CLASSES = list(_MODEL_CLASSES.keys())


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--project_name",
        type=str,
        default="hgere-pruner",
        help="project name for wandb",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="run name for wandb.",
    )
    parser.add_argument(
        "--label_set",
        type=str,
        default=None,
        help="label set to use (e.g., gsap)",
    )
    parser.add_argument(
        "--data_dir",
        default="ace_data",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
    )
    parser.add_argument(
        "--base_model_name_or_path",
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
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where the checkpoints will be written.",
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
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run test on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
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
        "--learning_rate_span",
        default=-1,
        type=float,
        help="The initial learning rate for span encoder.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
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
        "--eval_epochs", type=int, default=-1, help="Save checkpoint every eval_epochs."
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
        "--overwrite_model_dir",
        action="store_true",
        help="Overwrite the content of the model directory",
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
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
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
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents loaded (useful for quick testing).",
    )
    parser.add_argument("--debug_overflow", action="store_true", help="Enable DebugUnderflowOverflow to locate first NaN/Inf.")
    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument("--alpha", type=float, default=1, help="")
    parser.add_argument(
        "--pruner_loss",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Loss function for the pruner binary classifier. 'bce' = standard BCEWithLogitsLoss; 'focal' = focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter γ for focal loss (only used when --pruner_loss focal).",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Class-balance factor α for focal loss (only used when --pruner_loss focal).",
    )
    parser.add_argument("--max_pair_length", type=int, default=256, help="")
    parser.add_argument("--max_mention_ori_length", type=int, default=8, help="")
    parser.add_argument("--lminit", action="store_true")
    parser.add_argument("--norm_emb", action="store_true")
    parser.add_argument("--output_results", action="store_true")
    parser.add_argument("--onedropout", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--use_full_layer", type=int, default=-1, help="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--group_edge", action="store_true")
    parser.add_argument("--group_axis", type=int, default=-1, help="")
    parser.add_argument("--group_sort", action="store_true")
    parser.add_argument("--nocross", action="store_true")

    # for pruner
    parser.add_argument(
        "--target_recall_diff",
        type=float,
        default=0.01,
        help=(
            "Gap below pool upper-bound recall used as analysis target during eval "
            "(e.g. 0.01 = 1%%). Controls threshold/top-K search logged to wandb."
        ),
    )
    parser.add_argument(
        "--topk_ratio",
        default=0.5,
        type=float,
        help="Topk ratio, candidate entity number divide sentence length.",
    )
    parser.add_argument(
        "--min_mentions_num",
        type=int,
        default=3,
        help="min mentions number feed in pruner",
    )
    parser.add_argument(
        "--prune-config",
        dest="prune_config",
        default=None,
        type=str,
        help=(
            "Path to a best_config.json produced by evaluation/threshold_analysis.py. "
            "When set, the pruning parameters are loaded from this file instead of being "
            "estimated from the dev set at the end of training."
        ),
    )
    parser.add_argument(
        "--max_mentions_num",
        type=int,
        default=18,
        help="max mentions number feed in pruner",
    )
    parser.add_argument(
        "--extra_repr", type=str, default=None, help="use extra span repr"
    )
    parser.add_argument(
        "--rulebased-pruner-file",
        dest="rulebased_pruner_file",
        default=None,
        type=str,
        help=(
            "Path to a trained rule-based pruner JSON file. When set, spans matching "
            "its patterns are filtered out before the neural pruner sees them."
        ),
    )

    # for biaf span repr
    parser.add_argument("--biaf_span", action="store_true", help="use BiaffineSpanRepr")
    parser.add_argument(
        "--biaf_factorize", action="store_true", help="use BiaffineSpanRepr"
    )
    parser.add_argument("--biaf_mode", type=int, default=3, help="for BiaffineSpanRepr")
    parser.add_argument("--rank", type=int, default=768, help="for BiaffineSpanRepr")
    parser.add_argument(
        "--span_hidden_size", type=int, default=768, help="for BiaffineSpanRepr"
    )
    parser.add_argument(
        "--span_size", type=int, default=256, help="for BiaffineSpanRepr"
    )

    args = parser.parse_args()
    args.hostname = socket.gethostname()
    args.neg_inf = -1e4 if args.fp16 else -1e30
    return args


def cli():
    args = parse_arguments()
    run_train_span_classifier(args)


if __name__ == "__main__":
    cli()
