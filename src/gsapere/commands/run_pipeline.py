"""CLI command: run-pipeline

Processes JSON/JSONL documents through the full two-stage pipeline
(pruner → HGERE) using a YAML config and writes enriched predictions.

Single-file mode
----------------
    uv run run-pipeline \\
        --config pipeline/config.yaml \\
        --input docs.jsonl \\
        --output predictions.jsonl

Directory mode
--------------
    uv run run-pipeline \\
        --config pipeline/config.yaml \\
        --input docs/

    Searches docs/ for *.json / *.jsonl files and writes results to
    docs_predictions/ (same parent directory) with the same filenames.
    Pass --output <dir> to override the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from gsapere.pipeline.pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the two-stage HGERE pipeline on JSON/JSONL files."
    )
    p.add_argument(
        "--config", type=str, required=True, help="Path to pipeline YAML config."
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (*.json / *.jsonl) or directory containing such files.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file (single-file mode) or directory (directory mode). "
            "Must match the type of --input. "
            "Default (directory mode): <input_parent>/<input_name>_predictions/. "
            "Default (single-file mode): <input_parent>/<input_stem>_predictions<input_suffix>."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Number of documents per batch (0 = all at once).",
    )
    p.add_argument(
        "--debug_break_on_first_rel",
        action="store_true",
        help="Debug: log and raise after the first predicted relation.",
    )
    p.add_argument(
        "--debug_log_rel_probs",
        action="store_true",
        help="Debug: log top-5 relation probabilities for every span pair.",
    )
    return p


def _collect_input_files(input_path: Path) -> list[Path]:
    """Return all *.json and *.jsonl files under input_path (or just input_path itself)."""
    if input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir() if f.suffix in {".json", ".jsonl"}
        )
        return files
    return [input_path]


def _resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg is not None:
        return Path(output_arg)
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_predictions"
    return input_path.parent / f"{input_path.stem}_predictions{input_path.suffix}"


def _validate_input_output_types(
    input_path: Path, output_path: Path, logger: logging.Logger
) -> None:
    """Exit if input and output are not the same kind (file vs directory)."""
    input_is_dir = input_path.is_dir()
    # For output: if it already exists, check its type; otherwise infer from trailing separator
    # or match with input type.
    if output_path.exists():
        output_is_dir = output_path.is_dir()
    else:
        # Treat as directory when input is a directory (will be created later)
        output_is_dir = input_is_dir

    if input_is_dir != output_is_dir:
        logger.error(
            "--input and --output must be the same type "
            "(both a file or both a directory). "
            "Got input=%s (dir=%s) and output=%s (dir=%s).",
            input_path,
            input_is_dir,
            output_path,
            output_is_dir,
        )
        sys.exit(1)


def _load_docs(file_path: Path, logger: logging.Logger) -> list[dict]:
    docs = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    logger.info("Loaded %d documents from %s", len(docs), file_path)
    return docs


def _process_file(
    pipeline: Pipeline,
    input_file: Path,
    output_file: Path,
    batch_size: int,
    logger: logging.Logger,
    debug_break_on_first_rel: bool = False,
    debug_log_rel_probs: bool = False,
) -> None:
    docs = _load_docs(input_file, logger)
    if not docs:
        logger.warning("No documents in %s — skipping.", input_file)
        return

    effective_batch = batch_size if batch_size > 0 else len(docs)
    all_results: list[dict] = []

    show_doc_bar = len(docs) > 1
    doc_bar = (
        tqdm(total=len(docs), desc=input_file.name, unit="doc", leave=False)
        if show_doc_bar
        else None
    )
    try:
        for start in range(0, len(docs), effective_batch):
            batch = docs[start : start + effective_batch]
            total_sents = sum(len(d.get("sentences", [])) for d in batch)
            show_progress = total_sents > 100
            all_results.extend(
                pipeline.process_documents(
                    batch,
                    show_progress=show_progress,
                    debug_break_on_first_rel=debug_break_on_first_rel,
                    debug_log_rel_probs=debug_log_rel_probs,
                )
            )
            if doc_bar is not None:
                doc_bar.update(len(batch))
    finally:
        if doc_bar is not None:
            doc_bar.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for doc in all_results:
            f.write(json.dumps(doc) + "\n")
    logger.info("Wrote %d predictions to %s", len(all_results), output_file)


def cli() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        sys.exit(1)

    output_path = _resolve_output_path(input_path, args.output)
    _validate_input_output_types(input_path, output_path, logger)

    input_files = _collect_input_files(input_path)
    if not input_files:
        logger.error("No *.json or *.jsonl files found in %s", input_path)
        sys.exit(1)

    logger.info("Loading pipeline from %s", args.config)
    pipeline = Pipeline.from_yaml(args.config)

    file_iter: tqdm | list[Path]
    if input_path.is_dir():
        file_iter = tqdm(input_files, desc="Files", unit="file")
    else:
        file_iter = input_files

    for input_file in file_iter:
        if input_path.is_dir():
            output_file = output_path / input_file.name
        else:
            output_file = output_path

        _process_file(
            pipeline,
            input_file,
            output_file,
            args.batch_size,
            logger,
            debug_break_on_first_rel=args.debug_break_on_first_rel,
            debug_log_rel_probs=args.debug_log_rel_probs,
        )
