# Documentation

## Pipeline

- [Pruner](pruner.md) — role, input/output format, threshold tuning, and evaluation of the span pruner stage
- [Rule-based pruner pre-filtering](rulebased_pruner_prefiltering.md) — lightweight pre-filter to remove trivial non-entity spans before neural pruner training
- [Dataset download](download-dataset.md) — downloading datasets from their original sources, split details, and manual fallbacks

## Training

- [Multi-head multi-dataset training](multi_head_training.md) — joint training across multiple datasets with per-dataset classification heads
- [Train-time dynamic loss weighting](train_time_loss_weighting.md) — sigmoid-scheduled NER/RE loss balancing over the course of training

## API Reference

Auto-generated parameter references — run `uv run gsapere-generate-pruner-docs` to regenerate.

- [HGERE parameter reference](api/hgere.md)
- [Pruner parameter reference](api/pruner.md)
