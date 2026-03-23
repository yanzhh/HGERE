"""Cross-check HGERE inference via the training evaluation loop.

Runs ``evaluate()`` from ``src/gsapere/hgere/evaluate.py`` directly on a
loaded dataset — bypassing ``inference.py`` / ``infer_hgere()`` entirely.
The persisted JSONL output and markdown report can then be compared with
the output of ``infer_fixed_spans`` / ``infer_pruner_augmented`` to detect
any index or logic discrepancies.

Usage
-----
    # Gold-span candidates (mirrors infer-fixed-spans)
    uv run python scripts/analysis/eval_via_train_eval.py \\
        --model_name_or_path saves/my_checkpoint \\
        --label_set gsap \\
        --input_file data/test.jsonl \\
        --candidates_from ner \\
        --output_dir reports/train_eval_check/

    # Pruner candidates (mirrors infer-pruner-augmented with augment_with_gold)
    uv run python scripts/analysis/eval_via_train_eval.py \\
        --model_name_or_path saves/my_checkpoint \\
        --label_set gsap \\
        --input_file data/ent_pred_test.jsonl \\
        --candidates_from predicted_ner \\
        --augment_with_gold \\
        --output_dir reports/train_eval_check/

Outputs
-------
- <output_dir>/<stem>_train_eval_<timestamp>.json   — prediction JSONL
- <output_dir>/<stem>_train_eval_<timestamp>.md     — markdown report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gsapere.data.relation_dataset import RelationDataset
from gsapere.hgere.evaluate import evaluate
from gsapere.hgere.inference import prepare_input_file
from gsapere.labels import LABELS
from gsapere.utils import set_seed
from transformers import AutoTokenizer, BertConfig

from gsapere.models.hgere import BertForHyperGNN

MODEL_CLASSES = {
    "hyper": (BertConfig, BertForHyperGNN, AutoTokenizer),
}


# ---------------------------------------------------------------------------
# Report formatting (mirrors eval_hgere_predictions.py)
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{v * 100:.1f}"


def _row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _header(cols: list) -> str:
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    return _row(cols) + "\n" + sep


def build_report(results: dict, pred_path: Path, input_path: Path) -> str:
    lines: list[str] = []
    lines.append("# HGERE Evaluation Report (via train eval loop)")
    lines.append("")
    lines.append(f"**Predictions:** `{pred_path}`  ")
    lines.append(f"**Input:** `{input_path}`  ")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## NER")
    lines.append("")
    ner_rows = {
        "NER": {
            "precision": results["ner_precision"],
            "recall": results["ner_recall"],
            "f1": results["ner_f1"],
        }
    }
    header_cols = ["", "precision", "recall", "f1"]
    lines.append(_header(header_cols))
    for name, m in ner_rows.items():
        lines.append(_row([name] + [_pct(m[c]) for c in ["precision", "recall", "f1"]]))
    lines.append("")

    lines.append("## Relations")
    lines.append("")
    rel_rows = {
        "re  (span+label)": {
            "precision": results["re_precision"],
            "recall": results["re_recall"],
            "f1": results["re_f1"],
        },
        "re+ (span+label+type)": {
            "precision": results["re+_precision"],
            "recall": results["re+_recall"],
            "f1": results["re+_f1"],
        },
    }
    lines.append(_header(["", "precision", "recall", "f1"]))
    for name, m in rel_rows.items():
        lines.append(_row([name] + [_pct(m[c]) for c in ["precision", "recall", "f1"]]))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate HGERE using the training evaluation loop directly "
            "(bypasses inference.py for cross-checking)."
        )
    )

    # --- required ---
    p.add_argument(
        "--model_type", type=str, default="hyper", choices=list(MODEL_CLASSES.keys())
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--label_set", type=str, required=True)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/train_eval_check"),
        help="Directory for the prediction JSONL and markdown report.",
    )

    # --- candidate source (mirrors prepare_input_file) ---
    p.add_argument(
        "--candidates_from",
        choices=["predicted_ner", "ner"],
        default="predicted_ner",
        help=(
            "Candidate span source: 'predicted_ner' (pruner output, default) "
            "or 'ner' (gold spans, for gold-span evaluation)."
        ),
    )
    p.add_argument(
        "--augment_with_gold",
        action="store_true",
        help="Prepend gold spans to predicted_ner before inference.",
    )

    # --- tokenizer / sequences ---
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--max_pair_length", type=int, default=64)

    # --- dataset behaviour ---
    p.add_argument("--use_typemarker", action="store_true")
    p.add_argument("--no_sym", action="store_true")
    p.add_argument("--nocross", action="store_true")
    p.add_argument("--per_gpu_eval_batch_size", type=int, default=8)

    # --- hardware ---
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # --- model architecture (must match checkpoint) ---
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ent_repr", type=str, default="mix")
    p.add_argument("--ent_enc", type=str, default="cat")
    p.add_argument("--uni_ent", action="store_true")
    p.add_argument("--pred_sub", action="store_true")
    p.add_argument("--ner_cls", type=str, default="cat")
    p.add_argument("--rel_enc", type=str, default="cat")
    p.add_argument("--ent_dim", type=int, default=200)
    p.add_argument("--rel_dim", type=int, default=200)
    p.add_argument("--rel_rank", type=int, default=200)
    p.add_argument("--rel_factorize", action="store_true")
    p.add_argument("--baseline", type=str, default="firstorder")
    p.add_argument("--factor_type", type=str, default="ternary")
    p.add_argument("--mem_dim", type=int, default=200)
    p.add_argument("--iter", type=int, default=3)
    p.add_argument("--iter1", type=int, default=1)
    p.add_argument("--layernorm", action="store_true")
    p.add_argument("--layernorm_1st", action="store_true")
    p.add_argument("--attn_self", action="store_true")
    p.add_argument("--aggregate_type", type=str, default="attn")
    p.add_argument("--aggregate_func", type=str, default="max")
    p.add_argument("--agg_with_self", action="store_true")
    p.add_argument("--fix_obj", action="store_true")
    p.add_argument("--edgetype", type=str, default="sib")
    p.add_argument("--attn_scorer", type=str, default="biaf")
    p.add_argument("--attn_res", action="store_true")
    p.add_argument("--att_left", action="store_true")
    p.add_argument("--att_right", action="store_true")
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--factor_encoder", type=str, default="cat")
    p.add_argument("--re_focal_loss", action="store_true")
    p.add_argument("--re_focal_gamma", type=float, default=2.0)
    p.add_argument("--ner_focal_loss", action="store_true")
    p.add_argument("--ner_focal_gamma", type=float, default=2.0)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.n_iter = args.iter  # model uses args.n_iter (Pydantic config name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # --- device ---
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    set_seed(args)

    # --- labels ---
    assert args.label_set in LABELS, f"Unknown label_set '{args.label_set}'."
    labels = LABELS[args.label_set]
    args.num_ner_labels = labels.num_ner_labels
    args.num_rel_labels = labels.num_rel_labels(args.no_sym)

    # --- model / tokenizer ---
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    model_path = Path(args.model_name_or_path).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not (model_path / "config.json").exists():
        checkpoints = sorted(
            [
                p
                for p in model_path.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint-")
            ],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No config.json and no checkpoint-* subdirs in {model_path}"
            )
        model_path = checkpoints[-1]
        logger.info("Using checkpoint: %s", model_path)
    args.model_name_or_path = str(model_path)

    config = config_class.from_pretrained(
        args.model_name_or_path, num_labels=args.num_rel_labels
    )
    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = args.num_ner_labels

    tokenizer_path = Path(
        args.tokenizer_path
        or getattr(config, "_name_or_path", None)
        or args.model_name_or_path
    )
    if (
        not (tokenizer_path / "vocab.txt").exists()
        and not (tokenizer_path / "tokenizer.json").exists()
    ):
        raise FileNotFoundError(
            f"No vocab.txt or tokenizer.json in '{tokenizer_path}'. Pass --tokenizer_path."
        )
    tokenizer = tokenizer_class.from_pretrained(
        str(tokenizer_path), do_lower_case=args.do_lower_case
    )

    args.do_train = False
    args.lminit = False

    _tf_logger = logging.getLogger("transformers.modeling_utils")
    _prev = _tf_logger.level
    _tf_logger.setLevel(logging.ERROR)
    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, args=args
    )
    _tf_logger.setLevel(_prev)

    model.to(device)
    model.eval()

    # --- prepare candidate file ---
    input_path = Path(args.input_file).resolve()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_f:
        tmp_path = tmp_f.name

    try:
        prepare_input_file(
            str(input_path),
            tmp_path,
            candidates_from=args.candidates_from,
            augment_with_gold=args.augment_with_gold,
        )

        args.train_batch_size = args.per_gpu_eval_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.shuffle = False
        args.batch_by_size = False
        args.preload_dataset = False
        args.model_type = args.model_type.lower()

        # evaluate() uses args.model_dir to persist predictions
        output_dir: Path = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir = str(output_dir)
        dataset = RelationDataset(
            logger=logger,
            tokenizer=tokenizer,
            labels=labels,
            file_path=tmp_path,
            args=args,
            max_pair_length=args.max_pair_length,
            preload=False,
        )
        dataset.build(
            batch_size=args.eval_batch_size,
            shuffle=False,
            batch_by_size=False,
            n_workers=4,
            pin_memory=True,
        )
        logger.info("Loaded %d examples from %s", len(dataset), input_path)

        # Run the train-eval loop directly, persisting predictions.
        results = evaluate(
            model,
            dataset,
            args,
            logger,
            persist_predictions=True,
        )

        logger.info("Results: %s", json.dumps(results, indent=2))

    finally:
        os.unlink(tmp_path)

    # evaluate() writes to args.model_dir/<basename of dataset.file_path>
    # dataset.file_path is tmp_path (a temp file), so the output name is unpredictable.
    # Find the most recently written .json file in output_dir.
    candidates = sorted(
        output_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    pred_jsonl_path: Path | None = None
    if candidates:
        pred_jsonl_path = candidates[0]
        logger.info("Predictions written to: %s", pred_jsonl_path)
    else:
        logger.warning("No prediction JSONL found in %s", output_dir)

    # --- Save report ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_path.stem
    report_path = output_dir / f"{stem}_train_eval_{timestamp}.md"
    metrics_path = output_dir / f"{stem}_train_eval_{timestamp}.json"

    report = build_report(results, pred_jsonl_path or Path("(not found)"), input_path)
    report_path.write_text(report)
    metrics_path.write_text(json.dumps(results, indent=2))

    print(f"Report  → {report_path}")
    print(f"Metrics → {metrics_path}")
    if pred_jsonl_path:
        print(f"Preds   → {pred_jsonl_path}")


if __name__ == "__main__":
    main()
