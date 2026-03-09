"""
Pruner threshold analysis for the SciER dataset.

Usage
-----
    python evaluation/threshold_analysis.py \\
        --pred  saves/scier/pruner/biafencoder*/ent_pred_dev.json \\
        --dataset scier

The script:
  1. Sweeps thresholds from 0 to 1 and collects precision / recall curves.
  2. Binary-searches for the highest threshold that still achieves >= TARGET_RECALL.
  3. Grid-searches for the best top-K parameters (topk_ratio, min_mentions_num,
     max_mentions_num) achieving the same TARGET_RECALL with minimum FP share.
  4. Prints overall and per-entity-type metric tables comparing both methods.
  5. Saves a precision/recall vs threshold figure.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Allow running from any working directory
ROOT = Path(__file__).resolve().parents[1]   # …/HGERE
sys.path.insert(0, str(ROOT))

from evaluation.pruner import (
    load_predictions,
    threshold_sweep,
    find_threshold_for_recall,
    compute_metrics_at_threshold,
    compute_metrics_topk,
    optimize_topk_params,
)
from utils.labels import LABELS


def _count_total_candidates(docs: list) -> int:
    """Total number of spans scored by the pruner (before any threshold)."""
    return sum(
        len(sent)
        for doc in docs
        for sent in doc["predicted_ner_proba"]
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_RECALL = 0.99   # fallback if upper-bound recall cannot be computed
SWEEP_STEPS   = 200   # number of evenly-spaced threshold values for the curve

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pruner threshold analysis")
    p.add_argument(
        "--pred",
        required=True,
        help="Path to entity prediction file (jsonl), e.g. saves/scier/pruner/.../ent_pred_dev.json",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(LABELS.keys()),
        help="Dataset name (must be a key in utils/labels.py LABELS dict)",
    )
    p.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help=(
            "Target recall for the binary search. "
            "Defaults to (upper-bound recall on dev) - 0.01, "
            "where upper-bound recall = n_gold_in_pool / n_gold."
        ),
    )
    p.add_argument(
        "--train-pred",
        default=None,
        help=(
            "Optional path to a train-split prediction file. "
            "When provided, parameters tuned on dev are also evaluated on the train set."
        ),
    )
    p.add_argument(
        "--test-pred",
        default=None,
        help=(
            "Optional path to a test-split prediction file. "
            "When provided, parameters are tuned on --pred (dev) and evaluated on --test-pred (test). "
            "The report tables show test-set metrics."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save outputs. Defaults to the same directory as --pred.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _overall_table(
    thresh_metrics: dict,
    best_threshold: float,
    topk_metrics: dict,
    n_total_candidates: int,
    upper_bound_recall: float,
    target_recall: float,
    show_target: bool = True,
) -> str:
    """Side-by-side comparison of the threshold method and the top-K method."""
    # TN = unselected non-gold candidates within the pool.
    # FN includes gold spans outside the pool (exceeding max span length); we use
    # fn_within_pool so those out-of-pool spans don't make TN negative.
    def _tn(m: dict) -> int:
        fn_within_pool = m["n_gold_in_pool"] - m["tp"]
        return (n_total_candidates - m["n_pred"]) - fn_within_pool

    tn_thresh = _tn(thresh_metrics)
    tn_topk   = _tn(topk_metrics)

    def _tn_rate(m: dict, tn: int) -> float:
        """TN / (FP + TN) — fraction of non-gold pool spans correctly rejected."""
        denom = m["fp"] + tn
        return tn / denom if denom > 0 else 1.0

    n_gold            = thresh_metrics["n_gold"]
    n_gold_in_pool    = thresh_metrics["n_gold_in_pool"]   # same for both methods
    out_of_pool_gold  = n_gold - n_gold_in_pool

    topk_params = (
        f"λ={topk_metrics['topk_ratio']}  "
        f"lmin={topk_metrics['min_mentions_num']}  "
        f"lmax={topk_metrics['max_mentions_num']}"
    )

    rows = [
        ["selection",            f"threshold={best_threshold:.6f}", topk_params],
        ["**recall**",           _fmt(thresh_metrics["recall"]),     _fmt(topk_metrics["recall"])],
        ["upper_bound_recall",   _fmt(upper_bound_recall),           _fmt(upper_bound_recall)],
        *([["target_recall",     _fmt(target_recall),                _fmt(target_recall)]] if show_target else []),
        ["precision",            _fmt(thresh_metrics["precision"]),  _fmt(topk_metrics["precision"])],
        ["f1",                   _fmt(thresh_metrics["f1"]),         _fmt(topk_metrics["f1"])],
        ["false_positive_share", _fmt(thresh_metrics["false_positive_share"]), _fmt(topk_metrics["false_positive_share"])],
        ["TP",                   thresh_metrics["tp"],               topk_metrics["tp"]],
        ["FP",                   thresh_metrics["fp"],               topk_metrics["fp"]],
        ["FN (total)",           thresh_metrics["fn"],               topk_metrics["fn"]],
        ["TN",                   tn_thresh,                          tn_topk],
        ["TN/(FP+TN)",           _fmt(_tn_rate(thresh_metrics, tn_thresh)), _fmt(_tn_rate(topk_metrics, tn_topk))],
        ["n_gold",               n_gold,                             n_gold],
        ["n_gold_in_pool",       n_gold_in_pool,                     n_gold_in_pool],
        ["out_of_pool_gold",     out_of_pool_gold,                   out_of_pool_gold],
        ["n_pred",               thresh_metrics["n_pred"],           topk_metrics["n_pred"]],
        ["n_total_candidates",   n_total_candidates,                 n_total_candidates],
    ]
    return tabulate(rows, headers=["metric", "score threshold", "top-K"], tablefmt="github")


def _flatten_overall(
    metrics: dict,
    n_total_candidates: int,
    split: str,
    pred_file: str,
    method: str,
    threshold: Optional[float],
    topk_ratio: Optional[float],
    min_mentions_num: Optional[int],
    max_mentions_num: Optional[int],
    upper_bound_recall: float,
) -> dict:
    """Flat dict for one (split, method) row of the combined overall JSON table."""
    fn_within_pool = metrics["n_gold_in_pool"] - metrics["tp"]
    tn = (n_total_candidates - metrics["n_pred"]) - fn_within_pool
    denom = metrics["fp"] + tn
    tn_rate = tn / denom if denom > 0 else 1.0
    out_of_pool = metrics["n_gold"] - metrics["n_gold_in_pool"]
    return {
        "pred_file":           pred_file,
        "split":               split,
        "method":              method,
        "threshold":           threshold,
        "topk_ratio":          topk_ratio,
        "min_mentions_num":    min_mentions_num,
        "max_mentions_num":    max_mentions_num,
        "upper_bound_recall":  upper_bound_recall,
        "recall":              metrics["recall"],
        "precision":           metrics["precision"],
        "f1":                  metrics["f1"],
        "false_positive_share": metrics["false_positive_share"],
        "tp":                  metrics["tp"],
        "fp":                  metrics["fp"],
        "fn":                  metrics["fn"],
        "tn":                  tn,
        "tn_rate":             tn_rate,
        "n_gold":              metrics["n_gold"],
        "n_gold_in_pool":      metrics["n_gold_in_pool"],
        "out_of_pool_gold":    out_of_pool,
        "n_pred":              metrics["n_pred"],
        "n_total_candidates":  n_total_candidates,
    }


def _flatten_per_type(
    per_type: dict,
    entity_types: list,
    split: str,
    pred_file: str,
    method: str,
) -> "list[dict]":
    """List of flat dicts for one (split, method) block of the per-type JSON table."""
    rows = []
    for etype in entity_types:
        m = per_type.get(etype, {})
        rows.append({
            "pred_file":   pred_file,
            "split":       split,
            "method":      method,
            "entity_type": etype,
            "recall":      m.get("recall", 0.0),
            "tp":          m.get("tp",     0),
            "fn":          m.get("fn",     0),
            "n_gold":      m.get("n_gold", 0),
        })
    return rows


def _per_type_table(
    thresh_per_type: dict,
    topk_per_type: dict,
    entity_types: list,
) -> str:
    """Per-type recall comparison. FP spans carry no type, so only recall is shown."""
    headers = ["entity_type", "recall (thresh)", "recall (top-K)", "TP (thresh)", "TP (top-K)", "FN (thresh)", "FN (top-K)", "n_gold"]
    rows = []
    for etype in entity_types:
        t = thresh_per_type.get(etype, {})
        k = topk_per_type.get(etype, {})
        rows.append([
            etype,
            _fmt(t.get("recall", 0.0)),
            _fmt(k.get("recall", 0.0)),
            t.get("tp",     0),
            k.get("tp",     0),
            t.get("fn",     0),
            k.get("fn",     0),
            t.get("n_gold", 0),
        ])
    return tabulate(rows, headers=headers, tablefmt="github")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_threshold_curve(
    sweep_results: list,
    best_threshold: float,
    metrics_at_best: dict,
    n_total_candidates: int,
    out_path: Path,
    target_recall: float = TARGET_RECALL,
    upper_bound_recall: float = 1.0,
) -> None:
    thresholds  = [r["threshold"]  for r in sweep_results]
    precisions  = [r["precision"]  for r in sweep_results]
    recalls     = [r["recall"]     for r in sweep_results]
    f1s         = [r["f1"]         for r in sweep_results]
    fp_shares   = [r["false_positive_share"] for r in sweep_results]
    n_preds     = [r["n_pred"]     for r in sweep_results]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1]})

    # --- top panel: metric curves ---
    ax.plot(thresholds, precisions, label="Precision",            color="steelblue",  linewidth=2)
    ax.plot(thresholds, recalls,    label="Recall",               color="darkorange", linewidth=2)
    ax.plot(thresholds, f1s,        label="F1",                   color="seagreen",   linewidth=2, linestyle="--")
    ax.plot(thresholds, fp_shares,  label="False Positive Share", color="firebrick",  linewidth=1.5, linestyle=":")

    ax.axvline(best_threshold, color="grey", linestyle="--", linewidth=1.2,
               label=f"Best threshold ({best_threshold:.4f})")
    ax.axhline(upper_bound_recall, color="steelblue", linestyle=":", linewidth=1,
               label=f"Upper-bound recall ({upper_bound_recall:.4f})")
    ax.axhline(target_recall, color="darkorange", linestyle=":", linewidth=1,
               label=f"Target recall ({target_recall:.4f})")

    # Annotation box at best threshold
    n_pred_at_best = metrics_at_best["n_pred"]
    annotation = (
        f"threshold = {best_threshold:.4f}\n"
        f"n_pred    = {n_pred_at_best:,}  /  {n_total_candidates:,} total\n"
        f"recall    = {metrics_at_best['recall']:.4f}\n"
        f"precision = {metrics_at_best['precision']:.4f}"
    )
    ax.annotate(
        annotation,
        xy=(best_threshold, target_recall),
        xytext=(best_threshold + 0.04, 0.45),
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="grey"),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Pruner: Metrics vs. Threshold", fontsize=13)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- bottom panel: n_pred vs threshold ---
    ax2.plot(thresholds, n_preds, color="slategrey", linewidth=1.5)
    ax2.axvline(best_threshold, color="grey", linestyle="--", linewidth=1.2)
    ax2.axhline(n_total_candidates, color="slategrey", linestyle=":", linewidth=1,
                label=f"Total candidates ({n_total_candidates:,})")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Threshold", fontsize=11)
    ax2.set_ylabel("# predictions", fontsize=11)
    ax2.set_title("Number of Predicted Spans vs. Threshold", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dev_path = Path(args.pred)
    if not dev_path.exists():
        sys.exit(f"ERROR: prediction file not found: {dev_path}")

    train_path = Path(args.train_pred) if args.train_pred else None
    if train_path and not train_path.exists():
        sys.exit(f"ERROR: train prediction file not found: {train_path}")

    test_path = Path(args.test_pred) if args.test_pred else None
    if test_path and not test_path.exists():
        sys.exit(f"ERROR: test prediction file not found: {test_path}")

    out_dir = Path(args.out_dir) if args.out_dir else dev_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    label_scheme = LABELS[args.dataset]
    entity_types = [t for t in label_scheme.ner if t != "NIL"]

    # ------------------------------------------------------------------
    # Load dev (tuning) split
    # ------------------------------------------------------------------
    print(f"Loading dev predictions : {dev_path}")
    dev_docs = load_predictions(str(dev_path))
    print(f"  {len(dev_docs)} documents loaded")
    n_dev_candidates = _count_total_candidates(dev_docs)
    print(f"  {n_dev_candidates:,} total candidate spans scored by pruner")

    # ------------------------------------------------------------------
    # Compute upper-bound recall on dev and set target recall
    # ------------------------------------------------------------------
    # Upper-bound recall = fraction of gold spans that the pruner can ever recall
    # (those within its candidate pool, i.e. not exceeding max span length).
    dev_metrics_at_zero = compute_metrics_at_threshold(dev_docs, threshold=0.0)
    dev_n_gold_in_pool  = dev_metrics_at_zero["n_gold_in_pool"]
    dev_n_gold          = dev_metrics_at_zero["n_gold"]
    dev_upper_bound     = dev_n_gold_in_pool / dev_n_gold if dev_n_gold > 0 else 1.0
    dev_out_of_pool     = dev_n_gold - dev_n_gold_in_pool

    if args.target_recall is not None:
        target_recall = args.target_recall
        print(f"\n  Upper-bound recall (dev): {dev_upper_bound:.4f}  ({dev_out_of_pool} gold spans outside pool)")
        print(f"  Target recall (user-supplied): {target_recall:.4f}")
    else:
        target_recall = dev_upper_bound - 0.01
        print(
            f"\n  Upper-bound recall (dev): {dev_upper_bound:.4f}  ({dev_out_of_pool} gold spans outside pool)\n"
            f"  Target recall set to upper-bound - 1%: {target_recall:.4f}"
        )

    # ------------------------------------------------------------------
    # 1. Threshold sweep on dev for the figure
    # ------------------------------------------------------------------
    thresholds = list(np.linspace(0.0, 1.0, SWEEP_STEPS))
    print(f"\nSweeping {len(thresholds)} thresholds on dev …")
    sweep_results = threshold_sweep(dev_docs, thresholds)

    # ------------------------------------------------------------------
    # 2. Binary search for best threshold on dev
    # ------------------------------------------------------------------
    print(f"\nBinary search for threshold achieving recall >= {target_recall} on dev …")
    best_threshold, thresh_dev_metrics = find_threshold_for_recall(dev_docs, target_recall=target_recall)
    print(f"  Best threshold: {best_threshold:.6f}  (recall={thresh_dev_metrics['recall']:.4f})")
    if thresh_dev_metrics["recall"] < target_recall:
        print(
            f"  WARNING: target recall {target_recall} is not achievable — "
            f"max recall on dev is {thresh_dev_metrics['recall']:.4f} "
            f"(some gold spans exceed the pruner's max span length)."
        )

    # ------------------------------------------------------------------
    # 3. Top-K parameter optimisation on dev
    # ------------------------------------------------------------------
    print(f"\nGrid-searching top-K parameters for recall >= {target_recall} on dev …")
    best_topk_params, all_topk = optimize_topk_params(dev_docs, target_recall=target_recall)
    if best_topk_params is None:
        # Fall back to the configuration with the highest recall (first in all_topk after sorting)
        best_topk_params = all_topk[0]
        print(
            f"  WARNING: target recall {target_recall} not achievable with top-K — "
            f"using best available (recall={best_topk_params['recall']:.4f})."
        )
    lam  = best_topk_params["topk_ratio"]
    lmin = best_topk_params["min_mentions_num"]
    lmax = best_topk_params["max_mentions_num"]
    print(f"  Best top-K params: λ={lam}  lmin={lmin}  lmax={lmax}")

    # ------------------------------------------------------------------
    # 4. Dev evaluation (always reported, with target_recall)
    # ------------------------------------------------------------------
    thresh_metrics_dev = compute_metrics_at_threshold(dev_docs, best_threshold, entity_types=entity_types)
    topk_metrics_dev   = compute_metrics_topk(dev_docs, lam, lmin, lmax, entity_types=entity_types)

    overall_tbl_dev  = _overall_table(thresh_metrics_dev, best_threshold, topk_metrics_dev,
                                       n_dev_candidates, upper_bound_recall=dev_upper_bound,
                                       target_recall=target_recall, show_target=True)
    per_type_tbl_dev = _per_type_table(thresh_metrics_dev["per_type"], topk_metrics_dev["per_type"], entity_types)

    print("\n" + "=" * 70)
    print("Overall metrics — score threshold vs. top-K selection  (dev)")
    print("=" * 70)
    print(overall_tbl_dev)
    print("\n" + "=" * 70)
    print("Per-entity-type recall  (dev)")
    print("=" * 70)
    print(per_type_tbl_dev)

    # ------------------------------------------------------------------
    # Train evaluation (only if train split provided, target_recall omitted)
    # ------------------------------------------------------------------
    overall_tbl_train  = None
    per_type_tbl_train = None
    train_upper_bound  = dev_upper_bound

    if train_path:
        print(f"\nLoading train predictions: {train_path}")
        train_docs = load_predictions(str(train_path))
        print(f"  {len(train_docs)} documents loaded")
        n_train_candidates = _count_total_candidates(train_docs)

        thresh_metrics_train = compute_metrics_at_threshold(train_docs, best_threshold, entity_types=entity_types)
        topk_metrics_train   = compute_metrics_topk(train_docs, lam, lmin, lmax, entity_types=entity_types)

        train_metrics_at_zero = compute_metrics_at_threshold(train_docs, threshold=0.0)
        train_n_gold_in_pool  = train_metrics_at_zero["n_gold_in_pool"]
        train_n_gold          = train_metrics_at_zero["n_gold"]
        train_upper_bound     = train_n_gold_in_pool / train_n_gold if train_n_gold > 0 else 1.0

        overall_tbl_train  = _overall_table(thresh_metrics_train, best_threshold, topk_metrics_train,
                                             n_train_candidates, upper_bound_recall=train_upper_bound,
                                             target_recall=target_recall, show_target=False)
        per_type_tbl_train = _per_type_table(thresh_metrics_train["per_type"], topk_metrics_train["per_type"], entity_types)

        print("\n" + "=" * 70)
        print("Overall metrics — score threshold vs. top-K selection  (train)")
        print("=" * 70)
        print(overall_tbl_train)
        print("\n" + "=" * 70)
        print("Per-entity-type recall  (train)")
        print("=" * 70)
        print(per_type_tbl_train)

    # ------------------------------------------------------------------
    # Test evaluation (only if test split provided, target_recall omitted)
    # ------------------------------------------------------------------
    overall_tbl_test  = None
    per_type_tbl_test = None
    eval_upper_bound  = dev_upper_bound
    eval_n_gold_in_pool = dev_n_gold_in_pool
    eval_n_gold         = dev_n_gold

    if test_path:
        print(f"\nLoading test predictions: {test_path}")
        eval_docs = load_predictions(str(test_path))
        print(f"  {len(eval_docs)} documents loaded")
        n_eval_candidates = _count_total_candidates(eval_docs)
        thresh_metrics_test = compute_metrics_at_threshold(eval_docs, best_threshold, entity_types=entity_types)
        topk_metrics_test   = compute_metrics_topk(eval_docs, lam, lmin, lmax, entity_types=entity_types)

        test_metrics_at_zero = compute_metrics_at_threshold(eval_docs, threshold=0.0)
        eval_n_gold_in_pool  = test_metrics_at_zero["n_gold_in_pool"]
        eval_n_gold          = test_metrics_at_zero["n_gold"]
        eval_upper_bound     = eval_n_gold_in_pool / eval_n_gold if eval_n_gold > 0 else 1.0

        overall_tbl_test  = _overall_table(thresh_metrics_test, best_threshold, topk_metrics_test,
                                            n_eval_candidates, upper_bound_recall=eval_upper_bound,
                                            target_recall=target_recall, show_target=False)
        per_type_tbl_test = _per_type_table(thresh_metrics_test["per_type"], topk_metrics_test["per_type"], entity_types)

        print("\n" + "=" * 70)
        print("Overall metrics — score threshold vs. top-K selection  (test)")
        print("=" * 70)
        print(overall_tbl_test)
        print("\n" + "=" * 70)
        print("Per-entity-type recall  (test)")
        print("=" * 70)
        print(per_type_tbl_test)

    # ------------------------------------------------------------------
    # 6. Save report as markdown
    # ------------------------------------------------------------------
    report_path = out_dir / "threshold_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("# Pruner Threshold Analysis\n\n")
        f.write("| | |\n|---|---|\n")
        f.write(f"| **Dev file** | `{dev_path}` |\n")
        if train_path:
            f.write(f"| **Train file** | `{train_path}` |\n")
        if test_path:
            f.write(f"| **Test file** | `{test_path}` |\n")
        f.write(f"| **Dataset** | {args.dataset} |\n")
        f.write(f"| **Upper-bound recall (dev)** | {dev_upper_bound:.4f} "
                f"({dev_out_of_pool} gold spans outside pruner pool) |\n")
        f.write(f"| **Target recall** | {target_recall:.4f} (dev upper-bound − 1%) |\n")
        if train_path:
            f.write(f"| **Upper-bound recall (train)** | {train_upper_bound:.4f} |\n")
        if test_path:
            f.write(f"| **Upper-bound recall (test)** | {eval_upper_bound:.4f} "
                    f"({eval_n_gold - eval_n_gold_in_pool} gold spans outside pruner pool) |\n")
        f.write("\n---\n\n")
        f.write("## Dev — score threshold vs. top-K selection\n\n")
        f.write("_tuned and evaluated on dev_\n\n")
        f.write(overall_tbl_dev + "\n\n")
        f.write("## Dev — per-entity-type recall\n\n")
        f.write("_tuned and evaluated on dev_\n\n")
        f.write(per_type_tbl_dev + "\n")
        if overall_tbl_train is not None:
            f.write("\n---\n\n")
            f.write("## Train — score threshold vs. top-K selection\n\n")
            f.write("_parameters tuned on dev, evaluated on train_\n\n")
            f.write(overall_tbl_train + "\n\n")
            f.write("## Train — per-entity-type recall\n\n")
            f.write("_parameters tuned on dev, evaluated on train_\n\n")
            f.write(per_type_tbl_train + "\n")
        if overall_tbl_test is not None:
            f.write("\n---\n\n")
            f.write("## Test — score threshold vs. top-K selection\n\n")
            f.write("_parameters tuned on dev, evaluated on test_\n\n")
            f.write(overall_tbl_test + "\n\n")
            f.write("## Test — per-entity-type recall\n\n")
            f.write("_parameters tuned on dev, evaluated on test_\n\n")
            f.write(per_type_tbl_test + "\n")
    print(f"\nReport saved to: {report_path}")

    # ------------------------------------------------------------------
    # 7. Save best_config.json
    #    Best = method with lower FP share on dev (both tuned to target recall).
    # ------------------------------------------------------------------
    thresh_meets = thresh_metrics_dev["recall"] >= target_recall
    topk_meets   = topk_metrics_dev["recall"]   >= target_recall
    if thresh_meets and topk_meets:
        best_method = (
            "threshold"
            if thresh_metrics_dev["false_positive_share"] <= topk_metrics_dev["false_positive_share"]
            else "topk"
        )
    elif thresh_meets:
        best_method = "threshold"
    elif topk_meets:
        best_method = "topk"
    else:
        # Neither meets target — pick higher recall
        best_method = (
            "threshold"
            if thresh_metrics_dev["recall"] >= topk_metrics_dev["recall"]
            else "topk"
        )

    if best_method == "threshold":
        best_config = {"best_method": "threshold", "parameters": best_threshold}
    else:
        best_config = {
            "best_method": "topk",
            "parameters": {"topk_ratio": lam, "min_mentions_num": lmin, "max_mentions_num": lmax},
        }

    config_path = out_dir / "best_config.json"
    config_path.write_text(json.dumps(best_config, indent=2) + "\n")
    print(f"Best config saved to: {config_path}")

    # ------------------------------------------------------------------
    # 8. Save combined JSON tables
    # ------------------------------------------------------------------
    overall_rows   = []
    per_type_rows  = []

    def _add_split(split, pred_file, thresh_m, topk_m, n_cands, ub_recall):
        overall_rows.append(_flatten_overall(
            thresh_m, n_cands, split, pred_file, "threshold",
            threshold=best_threshold, topk_ratio=None,
            min_mentions_num=None, max_mentions_num=None,
            upper_bound_recall=ub_recall,
        ))
        overall_rows.append(_flatten_overall(
            topk_m, n_cands, split, pred_file, "topk",
            threshold=None, topk_ratio=lam,
            min_mentions_num=lmin, max_mentions_num=lmax,
            upper_bound_recall=ub_recall,
        ))
        per_type_rows.extend(_flatten_per_type(thresh_m["per_type"], entity_types, split, pred_file, "threshold"))
        per_type_rows.extend(_flatten_per_type(topk_m["per_type"],   entity_types, split, pred_file, "topk"))

    _add_split("dev",   str(dev_path),   thresh_metrics_dev, topk_metrics_dev, n_dev_candidates, dev_upper_bound)
    if train_path:
        _add_split("train", str(train_path), thresh_metrics_train, topk_metrics_train, n_train_candidates, train_upper_bound)
    if test_path:
        _add_split("test",  str(test_path),  thresh_metrics_test,  topk_metrics_test,  n_eval_candidates,  eval_upper_bound)

    overall_json_path  = out_dir / "metrics_table.json"
    per_type_json_path = out_dir / "per_type_table.json"
    overall_json_path.write_text(json.dumps(overall_rows,  indent=2) + "\n")
    per_type_json_path.write_text(json.dumps(per_type_rows, indent=2) + "\n")
    print(f"Overall metrics JSON saved to : {overall_json_path}")
    print(f"Per-type metrics JSON saved to: {per_type_json_path}")

    # ------------------------------------------------------------------
    # 9. Figure  (always based on dev sweep)
    # ------------------------------------------------------------------
    fig_path = out_dir / "threshold_analysis.png"
    plot_threshold_curve(sweep_results, best_threshold, thresh_metrics_dev, n_dev_candidates, fig_path,
                         target_recall=target_recall, upper_bound_recall=dev_upper_bound)


if __name__ == "__main__":
    main()
