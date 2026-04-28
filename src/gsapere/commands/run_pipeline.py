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
import os
import sys
from pathlib import Path

# Prevent wandb from initialising during pipeline inference.
os.environ.setdefault("WANDB_MODE", "disabled")

from tqdm import tqdm

from gsapere.labels import LABELS
from gsapere.pipeline.config import PipelineConfig
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


def _output_path_for_label_set(output_file: Path, label_set: str) -> Path:
    """Write into a subfolder named after the label set, keeping the original filename."""
    return output_file.parent / label_set / output_file.name


def _effective_output_file(
    base_output_file: Path,
    label_set: str | None,
    seed: int | None,
) -> Path:
    """Build the final output path, inserting label_set and seed subfolders when set."""
    path = base_output_file.parent
    if label_set is not None:
        path = path / label_set
    if seed is not None:
        path = path / str(seed)
    return path / base_output_file.name


def _write_results(
    results: list[dict], output_file: Path, logger: logging.Logger
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for doc in results:
            f.write(json.dumps(doc) + "\n")
    logger.info("Wrote %d predictions to %s", len(results), output_file)


def _process_file(
    pipeline: Pipeline,
    input_file: Path,
    output_file: Path,
    batch_size: int,
    logger: logging.Logger,
    seed: int | None = None,
    debug_break_on_first_rel: bool = False,
    debug_log_rel_probs: bool = False,
) -> None:
    docs = _load_docs(input_file, logger)
    if not docs:
        logger.warning("No documents in %s — skipping.", input_file)
        return

    label_sets = pipeline._config.label_sets
    multi = len(label_sets) > 1

    effective_batch = batch_size if batch_size > 0 else len(docs)

    if multi:
        all_results: dict[str, list[dict]] = {ls: [] for ls in label_sets}
    else:
        all_results_single: list[dict] = []

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
            if multi:
                batch_results = pipeline.process_documents_multi(
                    batch,
                    show_progress=show_progress,
                    debug_break_on_first_rel=debug_break_on_first_rel,
                    debug_log_rel_probs=debug_log_rel_probs,
                )
                for ls, docs_out in batch_results.items():
                    all_results[ls].extend(docs_out)
            else:
                all_results_single.extend(
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

    if multi:
        for ls, results in all_results.items():
            _write_results(
                results, _effective_output_file(output_file, ls, seed), logger
            )
    else:
        _write_results(
            all_results_single, _effective_output_file(output_file, None, seed), logger
        )


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

    logger.info("Loading pipeline config from %s", args.config)
    base_config = PipelineConfig.from_yaml(args.config)

    unknown = [ls for ls in base_config.label_sets if ls not in LABELS]
    if unknown:
        logger.error(
            "Unknown label set(s): %s. Known label sets: %s",
            unknown,
            sorted(LABELS.keys()),
        )
        sys.exit(1)

    # Expand seeds: a plain int means "single run, no seed subfolder";
    # a list means "one run per seed with /{seed}/ output subfolder".
    raw_seeds = base_config.hgere.seeds
    if isinstance(raw_seeds, list):
        seeds_iter = raw_seeds
        use_seed_subdir = True
    else:
        seeds_iter = [raw_seeds]
        use_seed_subdir = False

    file_iter: tqdm | list[Path]
    if input_path.is_dir():
        file_iter = tqdm(input_files, desc="Files", unit="file")
    else:
        file_iter = input_files

    for seed in seeds_iter:
        if use_seed_subdir:
            seed_hgere = base_config.hgere.model_copy(
                update={"model_dir": f"{base_config.hgere.model_dir}_seed{seed}"}
            )
            seed_config = base_config.model_copy(update={"hgere": seed_hgere})
            logger.info(
                "Running seed %d — loading HGERE from %s", seed, seed_hgere.model_dir
            )
        else:
            seed_config = base_config

        pipeline = Pipeline(seed_config)
        effective_seed = seed if use_seed_subdir else None

        for input_file in file_iter:
            output_file = (
                output_path / input_file.name if input_path.is_dir() else output_path
            )

            _process_file(
                pipeline,
                input_file,
                output_file,
                args.batch_size,
                logger,
                seed=effective_seed,
                debug_break_on_first_rel=args.debug_break_on_first_rel,
                debug_log_rel_probs=args.debug_log_rel_probs,
            )
