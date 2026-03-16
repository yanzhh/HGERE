"""
Plot NER candidate counts per sentence grouped by sentence-length bin.
Generates a side-by-side violin plot: threshold vs. top-k filtering.

Usage:
    uv run python scripts/plot_ner_candidates_by_sent_len.py \
        --threshold 0.1 \
        --topk_factor 0.5 --topk_min 1 --topk_max 18 \
        --pred_file pruner_predictions/scierc/focal-prefilter-2e-5/ent_pred_de.json \
        --out documentation/ner_candidates_by_sent_len_t0.1.png \
        --title "SciERC · focal-prefilter-2e-5 · dev set"
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── sentence-length bins ──────────────────────────────────────────────────────
BINS = [(6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 60)]


def bin_label(lo, hi):
    return f"{lo}-{hi}"


def sent_bin(n_tokens):
    for lo, hi in BINS:
        if lo <= n_tokens <= hi:
            return (lo, hi)
    return None   # outside range — excluded


# ── filtering strategies ──────────────────────────────────────────────────────

def candidates_threshold(sent_preds, threshold):
    """Return spans with prob >= threshold."""
    return [s for s in sent_preds if s[2] >= threshold]


def candidates_topk(sent_preds, sent_len, factor, kmin, kmax):
    """Return top-k spans by prob."""
    k = max(kmin, min(kmax, round(factor * sent_len)))
    ranked = sorted(sent_preds, key=lambda x: x[2], reverse=True)
    return ranked[:k]


# ── per-sentence data extraction ──────────────────────────────────────────────

def extract(docs, threshold, topk_factor, topk_min, topk_max):
    """
    Returns two lists of (sent_len, n_candidates) tuples — one per strategy —
    plus aggregate counts for the metrics table:
        (tp, fp, fn, n_pred)
    """
    thresh_rows, topk_rows = [], []
    # For metric table we need span-level TP/FP/FN
    gold_set = set()
    pred_thresh_set = set()
    pred_topk_set = set()

    for doc_idx, doc in enumerate(docs):
        for sent_idx, (sentence, sent_preds, sent_gold) in enumerate(
            zip(doc["sentences"], doc["predicted_ner_proba"], doc["ner"])
        ):
            n_tokens = len(sentence)
            b = sent_bin(n_tokens)
            uid = (doc_idx, sent_idx)

            for start, end, etype in sent_gold:
                gold_set.add((uid, start, end))

            # threshold
            cands_t = candidates_threshold(sent_preds, threshold)
            if b is not None:
                thresh_rows.append((n_tokens, len(cands_t)))
            for start, end, *_ in cands_t:
                pred_thresh_set.add((uid, start, end))

            # top-k
            cands_k = candidates_topk(sent_preds, n_tokens, topk_factor, topk_min, topk_max)
            if b is not None:
                topk_rows.append((n_tokens, len(cands_k)))
            for start, end, *_ in cands_k:
                pred_topk_set.add((uid, start, end))

    def counts(pred_set):
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        return tp, fp, fn, len(pred_set)

    return thresh_rows, topk_rows, counts(pred_thresh_set), counts(pred_topk_set)


# ── group by bin ──────────────────────────────────────────────────────────────

def group_by_bin(rows):
    groups = {b: [] for b in BINS}
    for n_tokens, n_cands in rows:
        b = sent_bin(n_tokens)
        if b is not None:
            groups[b].append(n_cands)
    return groups


# ── metrics helper ────────────────────────────────────────────────────────────

def metrics(tp, fp, fn, n_pred):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── plotting ──────────────────────────────────────────────────────────────────
COLOR_THRESH = "#5b8ec4"   # blue
COLOR_TOPK   = "#e8a87c"   # orange

def violin_panel(ax, groups, color, title, ylim=(0, 20)):
    labels = [bin_label(*b) for b in BINS]
    data   = [groups[b] for b in BINS]
    ns     = [len(d) for d in data]

    # Only plot bins with data
    positions = [i + 1 for i, d in enumerate(data) if d]
    active_data   = [d for d in data if d]
    active_labels = [l for l, d in zip(labels, data) if d]
    active_ns     = [n for n, d in zip(ns, data) if d]

    parts = ax.violinplot(active_data, positions=positions, showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")

    # Overlay mean ± std whiskers
    for pos, d in zip(positions, active_data):
        arr = np.array(d)
        m = arr.mean()
        p25, p75 = np.percentile(arr, 25), np.percentile(arr, 75)
        ax.vlines(pos, p25, p75, color="black", linewidth=1.5, zorder=3)
        ax.plot(pos, m, "o", color="white", markeredgecolor="black",
                markersize=6, zorder=4)

    # n= annotations
    for pos, n in zip(positions, active_ns):
        ax.text(pos, ylim[1] - 0.4, f"n={n}", ha="center", va="top",
                fontsize=7.5, color="gray")

    ax.set_title(title, fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(active_labels, fontsize=8)
    ax.set_xlabel("Sentence length (tokens)", fontsize=8)
    ax.set_ylabel("# NER candidates", fontsize=8)
    ax.set_ylim(*ylim)

    mean_patch = mpatches.Patch(facecolor="none", edgecolor="none", label="")
    ax.legend(
        handles=[plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="white", markeredgecolor="black",
                            markersize=7, label="mean")],
        loc="lower right", fontsize=8
    )


def make_table(fig, rows, y_pos=0.01):
    """Draw a summary table at the bottom of the figure."""
    col_labels = ["", "Precision", "Recall", "F1", "Predictions", "TP", "FP", "FN"]
    table_data = []
    for label, prec, rec, f1, n_pred, tp, fp, fn, color in rows:
        table_data.append([
            label,
            f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
            str(n_pred), str(tp), str(fp), str(fn)
        ])

    ax_table = fig.add_axes([0.05, 0.0, 0.92, 0.18])
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#b8cfe8")
        tbl[0, j].set_text_props(weight="bold")

    # row background colours
    bg_colors = [rows[i][8] for i in range(len(rows))]
    for i, bg in enumerate(bg_colors):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="pruner_predictions/scierc/focal-prefilter-2e-5/ent_pred_de.json")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--topk_factor", type=float, default=0.5)
    parser.add_argument("--topk_min", type=int, default=1)
    parser.add_argument("--topk_max", type=int, default=18)
    parser.add_argument("--out", default="documentation/ner_candidates_by_sent_len.png")
    parser.add_argument("--title", default="SciERC · focal-prefilter-2e-5 · dev set")
    args = parser.parse_args()

    print(f"Loading {args.pred_file} ...")
    with open(args.pred_file) as f:
        docs = [json.loads(l) for l in f if l.strip()]

    thresh_rows, topk_rows, (tp_t, fp_t, fn_t, npred_t), (tp_k, fp_k, fn_k, npred_k) = extract(
        docs, args.threshold, args.topk_factor, args.topk_min, args.topk_max
    )

    prec_t, rec_t, f1_t = metrics(tp_t, fp_t, fn_t, npred_t)
    prec_k, rec_k, f1_k = metrics(tp_k, fp_k, fn_k, npred_k)

    print(f"Threshold {args.threshold:.4f}: P={prec_t:.4f}  R={rec_t:.4f}  F1={f1_t:.4f}  "
          f"pred={npred_t}  TP={tp_t}  FP={fp_t}  FN={fn_t}")
    print(f"Top-k (factor={args.topk_factor}, min={args.topk_min}, max={args.topk_max}): "
          f"P={prec_k:.4f}  R={rec_k:.4f}  F1={f1_k:.4f}  "
          f"pred={npred_k}  TP={tp_k}  FP={fp_k}  FN={fn_k}")

    groups_t = group_by_bin(thresh_rows)
    groups_k = group_by_bin(topk_rows)

    fig = plt.figure(figsize=(13, 7))
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.07, right=0.97, wspace=0.3)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    thresh_title = (f"Threshold {args.threshold:.4f}"
                    f"  (min=1, max={args.topk_max})")
    topk_title   = (f"Top-k (factor={args.topk_factor},"
                    f" min={args.topk_min}, max={args.topk_max})")

    violin_panel(ax1, groups_t, COLOR_THRESH, thresh_title)
    violin_panel(ax2, groups_k, COLOR_TOPK,   topk_title)

    fig.suptitle(f"NER candidates per sentence  ·  {args.title}", fontsize=11, y=0.97)

    table_rows = [
        (f"Threshold {args.threshold:.4f}", prec_t, rec_t, f1_t, npred_t, tp_t, fp_t, fn_t, "#dce9f5"),
        (f"Top-k (factor={args.topk_factor}, min={args.topk_min}, m",
         prec_k, rec_k, f1_k, npred_k, tp_k, fp_k, fn_k, "#f5e8dc"),
    ]
    make_table(fig, table_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
