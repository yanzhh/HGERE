# CLAUDE.md

## Project Overview

This is a fork of the HGERE repo, adapted for **GSAP-ERE** (Entity and Relation Extraction on scientific text).

## Pipeline

Two-stage pipeline:

1. **Entity Pruning** (`pruner` path) — binary classifier that scores all possible n-grams (up to a
   configured max span length) in each sentence. Its job is to reduce the large set of candidate spans
   to a manageable subset passed to HGERE. It does **not** predict entity types — only a scalar
   probability that a span is an entity. See `documentation/pruner.md` for details.
2. **Entity and Relation Extraction** (`hgere` path) — runs HGERE on pruned candidates

## Entry Points

- `run_hgnn.py` — train/evaluate the HGERE model
- `run_pruner.py` — train/evaluate the pruner model
- `eval_pruner.py` — evaluate pruner standalone

## Scripts

Scripts are organized under `scripts/` by stage and dataset:

```
scripts/
  hgere/     # HGERE training/inference scripts per dataset
  pruner/    # Pruner training/inference scripts per dataset
```

Datasets covered: `gsap-rel`, `gsap-ere`, `scier`, `scinlp`, `somd2`, `ace04`, `ace05`, `scierc`

**Preferred approach for inference:** use the bash scripts under `scripts/` (e.g. `infer_scier_scier_focal.sh`), which call `run_pruner.py` or `run_hgnn.py` directly. The `pipeline/` directory contains experimental utilities and should not be used as the primary inference path.

## Pretrained Models

Stored in `pretrained_models/`:

- `scibert_scivocab_uncased` — SciBERT transformer base (used by both pruner and HGERE)
- `modernbert_base`

## Datasets

Training and evaluation datasets:

- **GSAP** — primary target dataset
- **SciNLP**
- **SciER**

Data lives in `/home/ottowg/projects/gsap/related_datasets/` (external to this repo).

## Package Management

- Uses **uv** with `pyproject.toml`.
- **Do not run `uv sync` or any `uv pip install`** — the user runs these manually.
- When adding dependencies, only edit `pyproject.toml` and tell the user to run `uv sync`.
- Always use `uv run` to execute scripts and tools (e.g. `uv run pytest`, `uv run python script.py`).

## Development Workflow

- **Test-driven development (TDD)**: always write tests first, then implement or restructure code.
  - Tests go in `tests/` mirroring the `src/` structure (e.g. `tests/gsapere/data/test_pruner_dataset.py`).
  - Run tests with `uv run pytest`.
- **Typed signatures**: all functions and methods must have complete type annotations (parameters + return type).
- **Code formatting**: run `uv run ruff format <file>` after writing or editing any Python file.
- **Linting**: run `uv run ruff check <file>` and fix all issues before considering work done.

## CLI and Parameter Design

All CLI entry points must follow the **Pydantic-first** pattern — no duplicate argparse definitions:

1. **Pydantic model is the single source of truth** for all parameters. Define fields in
   `src/gsapere/<model>/config.py` with `Field(description=...)`. Never hardcode the same parameter in
   both a config model and an argparse `add_argument` call.

2. **Use `load_config_from_argv`** from `gsapere.commands._cli_utils` to handle the two accepted forms:
   - Positional shortcut: `train-<cmd> config.yaml` (single arg without `--`)
   - Pure CLI params: `train-<cmd> --model_dir saves/... --train_params__learning_rate 2e-5`

3. **Bridge to legacy training code** via a `_config_to_namespace(config) -> argparse.Namespace`
   function that flattens the Pydantic model into a flat namespace. Apply any field remaps here
   (e.g. `model_dir → output_dir`). Compute derived fields like `neg_inf` here too.

4. **`cli()` is just an alias for `main()`** — needed for `pyproject.toml` entry points.

5. **Add to `pyproject.toml`** only the primary command name (e.g. `train-span-classifier`).
   Do not add separate `*-by-config` entry points.

6. **Regenerate parameter docs** after changing any config model:
   ```
   uv run generate-pruner-docs
   ```
   Docs are written to `documentation/api/`.

Reference implementations: `src/gsapere/commands/train_hgere.py`,
`src/gsapere/commands/train_span_classifier.py`.

## Other

- `pipeline/` — preprocessing utilities
- `notebooks/` — exploratory notebooks
- `saves/` — model checkpoints
- `logs/` — training logs
- `start_tensorboard.sh` — launch TensorBoard
