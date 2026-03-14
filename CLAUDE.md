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

## Other

- `pipeline/` — preprocessing utilities
- `notebooks/` — exploratory notebooks
- `saves/` — model checkpoints
- `logs/` — training logs
- `start_tensorboard.sh` — launch TensorBoard
