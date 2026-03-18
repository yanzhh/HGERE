"""CLI command: benchmark-pipeline

Measures end-to-end latency and per-stage throughput (pruner vs. HGERE)
for both single-document and batch processing.

Requires real model weights — not run in the normal test suite.

Usage
-----
    uv run benchmark-pipeline \\
        --config pipeline/config.yaml \\
        --input path/to/docs.jsonl \\
        --n_docs 20 \\
        --n_runs 10 \\
        --warmup 2

Output
------
Per-stage and end-to-end timing table (mean ± std, docs/sec).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate
from tqdm import tqdm


def _load_docs(input_path: Path, n_docs: int | None) -> list[dict[str, Any]]:
    docs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
                if n_docs is not None and len(docs) >= n_docs:
                    break
    return docs


def _time_call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call fn(*args, **kwargs), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _stats(times: list[float]) -> dict[str, float]:
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(times), "max": max(times)}


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _build_report(
    pipeline: Any,
    docs: list[dict[str, Any]],
    n_runs: int,
    warmup: int,
    config_path: str,
    input_path: str,
) -> str:
    """Run all timing passes and return the full report as a string."""
    n_docs = len(docs)
    single_doc = docs[0]
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Config    : {config_path}")
    emit(f"Input     : {input_path}")
    emit(f"Settings  : {n_docs} doc(s), {n_runs} run(s), {warmup} warmup(s)")
    emit(
        f"Device    : pruner={pipeline._pruner._device}"
        f"  hgere={pipeline._hgere._device}"
    )
    emit()

    # ---- warmup -------------------------------------------------------
    print("Warming up...", end=" ", flush=True)
    for _ in range(warmup):
        pipeline.process_document(single_doc)
    print("done.")
    emit()

    # ---- single-document timing ---------------------------------------
    single_times: list[float] = []
    for i in tqdm(range(n_runs), desc="single-doc", unit="run"):
        _, t = _time_call(pipeline.process_document, single_doc)
        single_times.append(t)

    # ---- batch timing: full pipeline ----------------------------------
    batch_total_times: list[float] = []
    for i in tqdm(range(n_runs), desc=f"pipeline  ({n_docs} docs)", unit="run"):
        _, t = _time_call(pipeline.process_documents, docs)
        batch_total_times.append(t)

    # ---- per-stage timing on the batch --------------------------------
    docs_with_candidates = pipeline._pruner.run(docs)

    pruner_times: list[float] = []
    for i in tqdm(range(n_runs), desc=f"pruner    ({n_docs} docs)", unit="run"):
        _, t = _time_call(pipeline._pruner.run, docs)
        pruner_times.append(t)

    hgere_times: list[float] = []
    for i in tqdm(range(n_runs), desc=f"hgere     ({n_docs} docs)", unit="run"):
        _, t = _time_call(pipeline._hgere.run, docs_with_candidates)
        hgere_times.append(t)

    # ---- table --------------------------------------------------------
    def _row(label: str, times: list[float], n: int) -> list[Any]:
        s = _stats(times)
        return [
            label,
            n,
            _format_ms(s["mean"]),
            f"± {_format_ms(s['std'])}",
            _format_ms(s["min"]),
            _format_ms(s["max"]),
            _format_ms(s["mean"] / n),
            f"{n / s['mean']:.1f}",
        ]

    headers = [
        "Stage",
        "Docs/call",
        "Mean total",
        "Std",
        "Min",
        "Max",
        "Mean/doc",
        "Docs/sec",
    ]
    rows = [
        _row("process_document (single)", single_times, 1),
        _row("pruner (batch)", pruner_times, n_docs),
        _row("hgere  (batch)", hgere_times, n_docs),
        _row("pipeline (batch, end-to-end)", batch_total_times, n_docs),
    ]
    emit(tabulate(rows, headers=headers, tablefmt="simple"))

    # Stage share
    pruner_mean = _stats(pruner_times)["mean"]
    hgere_mean = _stats(hgere_times)["mean"]
    total = pruner_mean + hgere_mean
    if total > 0:
        emit(
            f"\nStage share (batch): pruner {pruner_mean / total * 100:.1f}%"
            f"  |  hgere {hgere_mean / total * 100:.1f}%"
        )

    return "\n".join(lines)


def _run_benchmark(
    pipeline: Any,
    docs: list[dict[str, Any]],
    n_runs: int,
    warmup: int,
    config_path: str,
    input_path: str,
    report_dir: Path,
) -> None:
    report = _build_report(pipeline, docs, n_runs, warmup, config_path, input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"benchmark_{timestamp}.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed of the HGERE pipeline."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to pipeline YAML config."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="JSONL file with input documents."
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=None,
        help="Number of documents to load (default: all).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of timed repetitions (default: 10).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup calls before timing (default: 2).",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="reports",
        help="Directory to save the report (default: reports/).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING,  # suppress model-loading noise during benchmark
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    docs = _load_docs(input_path, args.n_docs)
    if not docs:
        print("Error: no documents found in input file.", file=sys.stderr)
        sys.exit(1)

    # Import here so model-loading happens after arg validation
    from hgere.pipeline.pipeline import Pipeline

    print(f"Loading pipeline from {args.config} ...")
    pipeline = Pipeline.from_yaml(args.config)
    print("Pipeline ready.")

    _run_benchmark(
        pipeline,
        docs,
        n_runs=args.n_runs,
        warmup=args.warmup,
        config_path=args.config,
        input_path=args.input,
        report_dir=Path(args.report_dir),
    )


if __name__ == "__main__":
    cli()
