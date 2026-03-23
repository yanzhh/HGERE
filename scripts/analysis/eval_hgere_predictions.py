"""Evaluate HGERE prediction files for NER and relation extraction.

Usage
-----
    uv run python scripts/analysis/eval_hgere_predictions.py \\
        datasets/gsap-ere/2025-05-15/test_predictions.jsonl

    # With pruner candidate section (reads final_pruning threshold from config)
    uv run python scripts/analysis/eval_hgere_predictions.py \\
        datasets/gsap-ere/2025-05-15/test_predictions.jsonl \\
        --pipeline-config configs/pipeline_gsap_test.yaml

    # Custom output directory
    uv run python scripts/analysis/eval_hgere_predictions.py \\
        datasets/gsap-ere/2025-05-15/test_predictions.jsonl --output-dir reports/

Outputs
-------
A Markdown report saved to <output_dir>/<stem>_eval_<timestamp>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gsapere.evaluation.hgere import (
    compute_ner_metrics,
    compute_ner_metrics_per_type,
    compute_pruner_metrics,
    compute_rel_metrics,
    compute_rel_metrics_per_type,
    compute_rel_metrics_with_ner,
    load_predictions,
)


def _load_pruner_threshold(pipeline_config_path: Path) -> float:
    """Read pruner.final_pruning.threshold from a pipeline YAML config."""
    with open(pipeline_config_path) as f:
        cfg = yaml.safe_load(f)
    method = cfg["pruner"]["final_pruning"]["method"]
    if method != "threshold":
        raise ValueError(
            f"Pipeline config uses final_pruning method '{method}', not 'threshold'. "
            "Only threshold-based pruning is supported for candidate evaluation."
        )
    return float(cfg["pruner"]["final_pruning"]["threshold"])


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{v * 100:.1f}"


def _row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _header(cols: list) -> str:
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    return _row(cols) + "\n" + sep


def _metrics_table(metrics_by_name: dict[str, dict], cols: list[str]) -> str:
    """Render a markdown table from a dict of {row_name: metrics_dict}."""
    header_cols = [""] + cols
    lines = [_header(header_cols)]
    for name, m in metrics_by_name.items():
        row = [name] + [_pct(m.get(c, 0.0)) for c in cols]
        lines.append(_row(row))
    return "\n".join(lines)


def _counts_table(metrics_by_name: dict[str, dict], count_cols: list[str]) -> str:
    header_cols = [""] + count_cols
    lines = [_header(header_cols)]
    for name, m in metrics_by_name.items():
        row = [name] + [m.get(c, 0) for c in count_cols]
        lines.append(_row(row))
    return "\n".join(lines)


def build_report(
    pred_path: Path,
    docs: list[dict],
    pruner_threshold: float | None = None,
) -> str:
    ner_overall = compute_ner_metrics(docs)
    ner_per_type = compute_ner_metrics_per_type(docs)
    rel_overall = compute_rel_metrics(docs)
    rel_with_ner = compute_rel_metrics_with_ner(docs)
    rel_per_type = compute_rel_metrics_per_type(docs)
    pruner_overall = (
        compute_pruner_metrics(docs, pruner_threshold)
        if pruner_threshold is not None
        else None
    )

    n_docs = len(docs)
    n_sents = sum(len(doc.get("sentences", [])) for doc in docs)

    lines: list[str] = []
    lines.append("# HGERE Evaluation Report")
    lines.append("")
    lines.append(f"**File:** `{pred_path}`  ")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Documents:** {n_docs}  **Sentences:** {n_sents}")
    lines.append("")

    # --- NER ---
    lines.append("## NER")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    ner_avg_rows = {
        "micro": {
            "precision": ner_overall["ner_precision"],
            "recall": ner_overall["ner_recall"],
            "f1": ner_overall["ner_f1"],
        },
        "macro": {
            "precision": ner_overall["ner_macro_precision"],
            "recall": ner_overall["ner_macro_recall"],
            "f1": ner_overall["ner_macro_f1"],
        },
        "w-macro": {
            "precision": ner_overall["ner_wmacro_precision"],
            "recall": ner_overall["ner_wmacro_recall"],
            "f1": ner_overall["ner_wmacro_f1"],
        },
    }
    lines.append(_metrics_table(ner_avg_rows, ["precision", "recall", "f1"]))
    lines.append("")
    ner_cnt_cols = ["ner_n_gold", "ner_n_pred", "ner_tp", "ner_fp", "ner_fn"]
    lines.append(_counts_table({"NER": ner_overall}, ner_cnt_cols))
    lines.append("")
    lines.append("### Per Entity Type")
    lines.append("")
    prf_cols = ["precision", "recall", "f1"]
    cnt_cols = ["n_gold", "n_pred", "tp", "fp", "fn"]
    lines.append(_metrics_table(ner_per_type, prf_cols))
    lines.append("")
    lines.append(_counts_table(ner_per_type, cnt_cols))
    lines.append("")

    # --- Relations ---
    lines.append("## Relations")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    # Combined overview table
    # P/R/F1 by averaging strategy for re and re+
    rel_avg_rows = {
        "re  micro": {
            "precision": rel_overall["re_precision"],
            "recall": rel_overall["re_recall"],
            "f1": rel_overall["re_f1"],
        },
        "re  macro": {
            "precision": rel_overall["re_macro_precision"],
            "recall": rel_overall["re_macro_recall"],
            "f1": rel_overall["re_macro_f1"],
        },
        "re  w-macro": {
            "precision": rel_overall["re_wmacro_precision"],
            "recall": rel_overall["re_wmacro_recall"],
            "f1": rel_overall["re_wmacro_f1"],
        },
        "re+ micro": {
            "precision": rel_with_ner["re+_precision"],
            "recall": rel_with_ner["re+_recall"],
            "f1": rel_with_ner["re+_f1"],
        },
        "re+ macro": {
            "precision": rel_with_ner["re+_macro_precision"],
            "recall": rel_with_ner["re+_macro_recall"],
            "f1": rel_with_ner["re+_macro_f1"],
        },
        "re+ w-macro": {
            "precision": rel_with_ner["re+_wmacro_precision"],
            "recall": rel_with_ner["re+_wmacro_recall"],
            "f1": rel_with_ner["re+_wmacro_f1"],
        },
    }
    lines.append(_metrics_table(rel_avg_rows, ["precision", "recall", "f1"]))
    lines.append("")
    # Counts (micro only)
    overview_lines = [_header(["", "n_gold", "n_pred", "TP", "FP", "FN"])]
    for label, m, prefix in [("re", rel_overall, "re"), ("re+", rel_with_ner, "re+")]:
        overview_lines.append(
            _row(
                [
                    label,
                    m[f"{prefix}_n_gold"],
                    m[f"{prefix}_n_pred"],
                    m[f"{prefix}_tp"],
                    m[f"{prefix}_fp"],
                    m[f"{prefix}_fn"],
                ]
            )
        )
    lines.append("\n".join(overview_lines))
    lines.append("")

    if rel_per_type:
        lines.append("### Per Relation Type")
        lines.append("")
        lines.append(_metrics_table(rel_per_type, prf_cols))
        lines.append("")
        lines.append(_counts_table(rel_per_type, cnt_cols))
        lines.append("")

    # --- Pruner candidates ---
    if pruner_overall is not None:
        lines.append("## Pruner Candidates")
        lines.append("")
        lines.append(f"**Threshold:** {pruner_threshold}")
        lines.append("")
        pruner_prf_cols = ["pruner_precision", "pruner_recall", "pruner_f1"]
        pruner_cnt_cols = [
            "pruner_n_gold",
            "pruner_n_pred",
            "pruner_tp",
            "pruner_fp",
            "pruner_fn",
        ]
        lines.append(_metrics_table({"Pruner": pruner_overall}, pruner_prf_cols))
        lines.append("")
        lines.append(_counts_table({"Pruner": pruner_overall}, pruner_cnt_cols))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HGERE prediction files for NER and relation extraction."
    )
    parser.add_argument(
        "predictions", type=Path, help="Path to predictions JSONL file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write the report to (default: reports/).",
    )
    parser.add_argument(
        "--print",
        dest="print_report",
        action="store_true",
        help="Also print the report to stdout.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help=(
            "Path to a pipeline YAML config.  When provided, reads "
            "pruner.final_pruning.threshold and adds a Pruner Candidates section "
            "based on the ner_candidates_proba field."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_path: Path = args.predictions.resolve()
    if not pred_path.exists():
        print(f"Error: file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    pruner_threshold: float | None = None
    if args.pipeline_config is not None:
        pruner_threshold = _load_pruner_threshold(args.pipeline_config)

    docs = load_predictions(str(pred_path))

    report = build_report(pred_path, docs, pruner_threshold=pruner_threshold)

    # Also dump JSON metrics alongside the report
    ner_overall = compute_ner_metrics(docs)
    ner_per_type = compute_ner_metrics_per_type(docs)
    rel_overall = compute_rel_metrics(docs)
    rel_with_ner = compute_rel_metrics_with_ner(docs)
    rel_per_type = compute_rel_metrics_per_type(docs)
    all_metrics: dict = {
        "ner": ner_overall,
        "ner_per_type": ner_per_type,
        "re": rel_overall,
        "re+": rel_with_ner,
        "re_per_type": rel_per_type,
    }
    if pruner_threshold is not None:
        all_metrics["pruner"] = compute_pruner_metrics(docs, pruner_threshold)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = pred_path.stem
    report_path = output_dir / f"{stem}_eval_{timestamp}.md"
    metrics_path = output_dir / f"{stem}_eval_{timestamp}.json"

    report_path.write_text(report)
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    print(f"Report  → {report_path}")
    print(f"Metrics → {metrics_path}")

    if args.print_report:
        print()
        print(report)


if __name__ == "__main__":
    main()
