# Multi-Head Multi-Dataset Training

## Motivation

HGERE was originally designed to train on a single dataset at a time — SciER,
SciNLP, or GSAP-ERE — each with its own entity and relation label scheme.
Training on a single dataset provides good performance on that domain but wastes
the signal available in the other corpora, which share a common underlying
representation (scientific text, entity spans, relation tuples).

Multi-head training lets a single HGERE model learn from all three datasets
simultaneously:

- The **shared encoder** (SciBERT backbone, HyperGNN layers) benefits from
  the full combined training signal.
- Each dataset gets its own **NER head** and **relation head**, so label
  schemes never interfere with each other.
- Each mini-batch comes from exactly one dataset and activates only that
  dataset's heads. Inactive heads receive no gradients.

---

## Architecture

```
                    ┌─────────────────────────────┐
  batch (scier)  ──▶│  Shared encoder             │
  batch (scinlp) ──▶│  (SciBERT + HyperGNN)       │
  batch (gsap)   ──▶│                             │
                    └──────────────┬──────────────┘
                                   │ representations
                     ┌─────────────┼────────────────┐
                     ▼             ▼                ▼
               NER head       NER head         NER head
               RE head        RE head          RE head
               (scier)        (scinlp)         (gsap)
```

Head selection is driven by the `dataset_id` string that flows through the
entire data pipeline into `model.forward()`.

The per-dataset heads are stored as `nn.ModuleDict` members on the model
(`model.ner_heads["scier"]`, `model.rel_heads["scinlp"]`, etc.) and are saved
and loaded automatically by HuggingFace's `save_pretrained` / `from_pretrained`.

---

## Single-Dataset Backward Compatibility

Multi-head mode is **opt-in**. Any existing single-dataset config works
unchanged:

- If `multi_dataset` is absent from the config, a single NER head and single
  relation head are created exactly as before.
- `dataset_id` is `None` throughout the pipeline; the model's `forward()`
  falls back to the single-head path.
- All existing checkpoints load and evaluate without modification.

---

## Configuration

Instead of `label_set` and `ner_prediction_dir` at the top level, set
`multi_dataset`:

```yaml
# configs/train/multi_dataset_example.yaml

model_dir: saves/multi_head/scier_scinlp_run1
base_model_name_or_path: allenai/scibert_scivocab_uncased
model_type: hyper
no_sym: true
max_seq_length: 512
# ... other shared model hyperparams ...

multi_dataset:
  sampling_temperature: 0.5    # see Sampling below
  datasets:
    - name: scier
      label_set: scier
      ner_prediction_dir: saves/scier/pruner/output
      train_file: ent_pred_train.json
      dev_file:   ent_pred_dev.json
      test_file:  ent_pred_test.json
      sampling_weight: 1.0
    - name: scinlp
      label_set: scinlp
      ner_prediction_dir: saves/scinlp/pruner/output
      sampling_weight: 1.0

train_params:
  do_train: true
  evaluate_during_training: true
  # ... rest of train_params unchanged ...
```

The two top-level fields `label_set` and `ner_prediction_dir` must be **absent**
when `multi_dataset` is set (the config validator raises an error if both are
present).

### `multi_dataset` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `datasets` | list | — | One entry per dataset (required) |
| `sampling_temperature` | float | `0.5` | Controls batch interleaving order (see below) |

### `datasets[*]` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | — | Dataset identifier, e.g. `"scier"` |
| `label_set` | str | — | Key into the `LABELS` registry |
| `ner_prediction_dir` | str | — | Path to the pruner output directory |
| `train_file` | str | `"train.json"` | Training split filename |
| `dev_file` | str | `"dev.json"` | Dev split filename |
| `test_file` | str | `"test.json"` | Test split filename |
| `sampling_weight` | float | `1.0` | Relative weight for batch sampling |

---

## Batch Sampling

Each training epoch contains exactly `sum(n_batches_per_dataset)` steps — no
dataset is oversampled or undersampled. The **order** in which batches from
different datasets are interleaved is controlled by a temperature-weighted
stochastic schedule:

```
p(d) ∝  n_d^T  ×  sampling_weight[d]
```

where `n_d` is the remaining batch quota for dataset `d` and `T` is
`sampling_temperature`.

| Temperature | Effect |
|---|---|
| `T = 1.0` | Probability proportional to remaining batch count (larger datasets appear more often early in epoch) |
| `T ≈ 0` | Near-uniform mixing; datasets are sampled roughly round-robin regardless of size |
| `T > 1` | Even stronger concentration on the larger dataset |

`sampling_weight` provides an additional per-dataset scalar multiplier. Setting
`sampling_weight: 2.0` for one dataset doubles its relative probability of being
selected at each step (while still exhausting all its batches over the epoch).

Each mini-batch is **homogeneous**: it contains sentences from exactly one
dataset, and `batch["dataset_id"]` is set to that dataset's name.

---

## Evaluation

During training, each dataset's dev set is evaluated independently after every
`eval_epochs` epochs. The metric used for best-model selection is the
**macro-average `re+_f1`** across all datasets.

W&B metrics are logged with per-dataset prefixes:

| Key | Description |
|---|---|
| `eval/scier/re+_f1` | RE+F1 on SciER dev |
| `eval/scinlp/re+_f1` | RE+F1 on SciNLP dev |
| `eval/re+_f1` | Macro-average (used for checkpoint selection) |

After training, final evaluation on train/dev/test splits runs per dataset and
writes separate result JSON files.

---

## Usage

```bash
# Train with multi-dataset config
uv run train-hgere configs/train/multi_dataset_example.yaml

# Evaluate a saved checkpoint on all datasets
uv run train-hgere configs/train/multi_dataset_example.yaml \
    --do_train false --eval_test true \
    --model_dir saves/multi_head/scier_scinlp_run1
```

---

## Implementation Notes

### `dataset_id` propagation

The string travels through the full data pipeline:

1. `RelationDatasetParams.dataset_id` → stored on `RelationDataset`
2. `RelationDataset.prepare_item()` → each item dict contains `"dataset_id"`
3. `collators._collate_relation_batch()` → propagated to the collated batch dict
4. `MultiDatasetIterableLoader.__iter__()` → batch yielded with `dataset_id` intact
5. Training loop → added to `inputs` dict without `.to(device)`
6. `model.forward(dataset_id=...)` → selects the correct NER and RE head

### `MultiDatasetIterableLoader`

A plain Python iterable (not a `torch.DataLoader`) that holds one iterator per
dataset and advances them according to a pre-computed `schedule` (a list of
dataset names). When a dataset's iterator is exhausted mid-schedule (possible
with non-uniform temperatures), it is automatically reset.

```python
from gsapere.data.multi_dataset import build_sampling_schedule, MultiDatasetIterableLoader

schedule = build_sampling_schedule(
    loader_lengths={"scier": 120, "scinlp": 80},
    sampling_weights={"scier": 1.0, "scinlp": 1.0},
    temperature=0.5,
)
loader = MultiDatasetIterableLoader(loaders={"scier": dl_a, "scinlp": dl_b}, schedule=schedule)
for batch in loader:
    dataset_id = batch["dataset_id"]   # "scier" or "scinlp"
```

### Model weights

On `from_pretrained` from a base SciBERT/ModernBERT checkpoint, all new head
parameters (in `ner_heads` and `rel_heads`) are missing and are randomly
initialised by HuggingFace's standard missing-key handling. No special
initialisation code is needed.

The `config.dataset_heads` dict is serialised as part of `config.json` and
is restored automatically on `from_pretrained`, so the model re-creates the
correct `nn.ModuleDict` structure without any extra arguments.
