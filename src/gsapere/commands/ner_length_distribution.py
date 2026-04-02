"""Plot the entity length (n_tokens) distribution for a dataset directory.

All .json and .jsonl files found in --dataset_path get one bar group each.
Labels are derived from the file stem (e.g. train.json → train).

Usage
-----
    uv run gsapere-analysis-ner-length-distribution \\
        --dataset_path /path/to/dataset/

    # Custom output path
    uv run gsapere-analysis-ner-length-distribution \\
        --dataset_path /path/to/dataset/ \\
        --output_path reports/ner_length_dist.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_entity_lengths(docs: list[dict]) -> list[int]:
    """Return entity token lengths from all sentences in all docs."""
    lengths: list[int] = []
    for doc in docs:
        for sent_ner in doc.get("ner", []):
            for span in sent_ner:
                start, end = int(span[0]), int(span[1])
                lengths.append(end - start + 1)
    return lengths


def plot_length_distribution(
    files: list[Path],
    output_path: Path,
    max_length: int,
) -> None:
    all_counts: list[tuple[str, Counter[int]]] = []
    for path in files:
        docs = load_jsonl(path)
        lengths = collect_entity_lengths(docs)
        counts: Counter[int] = Counter(lengths)
        all_counts.append((path.stem, counts))

    x = np.arange(1, max_length + 1)
    n_splits = len(all_counts)
    width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

    fig, ax = plt.subplots(figsize=(max(8, max_length), 5))

    for (label, counts), offset in zip(all_counts, offsets):
        heights = [counts.get(length, 0) for length in x]
        ax.bar(x + offset, heights, width=width, label=label, alpha=0.85)

    ax.set_xlabel("Entity length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Entity length distribution")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NER entity length distribution for all splits in a dataset directory."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Directory containing .json / .jsonl dataset files (one bar group per file).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("reports/ner_length_distribution.png"),
        help="Output image path (default: reports/ner_length_distribution.png).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=15,
        help="Maximum entity length to include in the plot (default: 15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path: Path = args.dataset_path
    if not dataset_path.is_dir():
        print(f"Error: not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    files = sorted(p for p in dataset_path.iterdir() if p.suffix in {".json", ".jsonl"})
    if not files:
        print(
            f"Error: no .json / .jsonl files found in {dataset_path}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Found {len(files)} file(s): {[f.name for f in files]}")

    plot_length_distribution(
        files=files,
        output_path=args.output_path,
        max_length=args.max_length,
    )


def cli() -> None:
    main()
