# gsapere — Entity and Relation Extraction for Scientific Text

A fork of [HGERE](https://github.com/yanzhh/HGERE) adapted for scientific text, with a two-stage pipeline for **joint entity and relation extraction (ERE)**.

> **Paper under review.**
> Configs used for our experiments are in [`configs/`](configs/).

The pipeline consists of:

1. **Span Pruner** — a lightweight binary classifier that scores all candidate n-grams and filters them down to a manageable set (target: ≥ 98 % entity recall)
2. **HGERE** — a Hypergraph GNN that jointly predicts entity types and relations on the pruned candidates

Supported datasets: **GSAP-ERE**, **SciER**, **SciNLP**, **SciERC**

---

## Changes from the original

- Large-scale code restructuring: Pydantic-first configs, typed signatures throughout, proper package layout under `src/`
- All dependencies updated to current versions
- The transformer package is **no longer hardcoded** — any compatible HuggingFace `transformers` version works
- Added span pruner stage, multi-dataset joint training, and full CLI entry points
- Tests for all major components

---

## Requirements

- **Python 3.9** (tested; `<3.11` required by some dependencies)
- CUDA 12.8 (adjust `pyproject.toml` for other CUDA versions)
- A GPU with at least ~24 GB VRAM for default batch sizes (tested on A40 / 40 GB)

---

## Installation

Install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install:

```bash
git clone <repo-url>
cd HGERE
uv sync
```

### Datasets

Datasets are loaded from their original sources via the download command:

```bash
uv run gsapere-download-dataset --list          # list available datasets
uv run gsapere-download-dataset gsap-ere
uv run gsapere-download-dataset scier
uv run gsapere-download-dataset scinlp
uv run gsapere-download-dataset scierc
uv run gsapere-download-dataset --all           # download everything
```

See [documentation/download-dataset.md](documentation/download-dataset.md) for split details and manual download fallbacks.

#### GSAP-ERE

Entity and relation extraction on scientific text, developed at GESIS.
DOI: <https://doi.org/10.60914/c4c1d-s0587>

> Published at AAAI 2026.

#### SciER

Entity and relation extraction dataset for datasets, methods, and tasks in scientific documents — 106 annotated full-text papers, 24k entities, 12k relations.

> Dziadek et al., "SciER: An Entity and Relation Extraction Dataset for Datasets, Methods, and Tasks in Scientific Documents", EMNLP 2024.
> <https://aclanthology.org/2024.emnlp-main.726/>

#### SciNLP

Full-text entity and relation extraction benchmark for the NLP domain — 60 annotated ACL papers, 6,409 entities, 1,648 relations.

> "SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP", EMNLP 2025.
> <https://aclanthology.org/2025.emnlp-main.732/>

#### SciERC

Scientific information extraction benchmark — 500 annotated AI abstracts,
6 entity types, 7 relation types.

> Luan et al., "Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction", EMNLP 2018.
> <https://aclanthology.org/D18-1360/>

---

## Training

Training is a two-step process: first train the pruner, then train HGERE on the pruner's output.

### Step 1 — Train the span pruner

```bash
uv run gsapere-train-pruner configs/train/gsap/train_gsap_pruner.yaml
```

After training, run pruner inference on train/dev/test to produce the enriched input files for HGERE (see `scripts/pruner/`).

### Step 2 — Train HGERE (single dataset)

```bash
uv run gsapere-train-hgere configs/train/gsap/train_gsap_hgere.yaml
```

Example config:

```yaml
schema_version: "1.0"
label_set: gsap
model_dir: saves/hgere/gsap
base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
ner_prediction_dir: saves/pruner/gsap/output
max_seq_length: 512
n_iter: 3
layernorm: true
attn_self: true

train_params:
  learning_rate: 1e-5
  num_train_epochs: 8
  per_gpu_train_batch_size: 21
  fp16: true
  evaluate_during_training: true
  eval_epochs: 1
  loss_re_weight_alpha: 0.9
  log_wandb: true
```

### Step 2 (alt) — Train HGERE on multiple datasets jointly

Multi-dataset mode trains a shared encoder with per-dataset NER and relation heads. Each dataset must have its own pruner output directory.

```bash
uv run gsapere-train-hgere configs/multi-sciere-scinlp-gsap-ere/train/hgere/train_multi.yaml
```

Example config:

```yaml
schema_version: "1.0"
model_dir: saves/multi/hgere/run1
base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
max_seq_length: 512
n_iter: 3
layernorm: true
attn_self: true
sampling_temperature: 0.8   # 0 = always largest dataset, 1 = proportional to size
seeds: [42, 43, 44]          # run once per seed; _seed<n> appended to model_dir

datasets:
  - label_set: scier
    ner_prediction_dir: saves/pruner/scier/output
    train_file: ent_pred_train.json
    dev_file: ent_pred_dev.json
    test_file: ent_pred_test.json
  - label_set: scinlp
    ner_prediction_dir: saves/pruner/scinlp/output
    train_file: ent_pred_train.json
    dev_file: ent_pred_dev.json   # omit (null) to skip dev evaluation for this dataset
  - label_set: gsap
    ner_prediction_dir: saves/pruner/gsap/output
    train_file: ent_pred_train.json

train_params:
  learning_rate: 1e-5
  num_train_epochs: 8
  per_gpu_train_batch_size: 21
  fp16: true
  evaluate_during_training: true
  log_wandb: true
```

### Rule-based pruner (optional, for performance)

The neural pruner scores all O(n²) candidate spans per sentence, most of which are unambiguous non-entities (punctuation, function-word sequences, purely numeric tokens, etc.).
The rule-based pruner is a lightweight pre-filter that removes these deterministically before training, reducing the candidate set while keeping gold-entity recall at or near 100 %.
This speeds up training and can improve convergence by reducing the imbalance of trivial negatives.

Fit the rule-based pruner from training data:

```bash
uv run gsapere-fit-rulebased-pruner configs/train/gsap/fit_rulebased_pruner.yaml
```

Then reference the saved JSON file in your pruner training config:

```yaml
train_params:
  rulebased_pruner_file: saves/pruner/gsap/rules.json
```

See [documentation/rulebased_pruner_prefiltering.md](documentation/rulebased_pruner_prefiltering.md) for details.

---

## Inference

### Full pipeline (pruner → HGERE)

```bash
CUDA_VISIBLE_DEVICES=0 uv run gsapere-pipeline \
    --config configs/inference/gsap-pipeline-best.yaml \
    --input input/ \
    --output output/
```

`--input` can be a `.jsonl` file or a directory of `.jsonl` files.
Ready-to-use configs for all supported datasets are in [`configs/inference/`](configs/inference/).

The pipeline config combines pruner and HGERE settings in a single YAML file:

```yaml
label_set: gsap

pruner:
  model_dir: saves/pruner/gsap/best
  base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
  model_type: bertspanmarkerpruner
  max_seq_length: 256
  per_gpu_eval_batch_size: 32
  final_pruning:
    method: threshold
    threshold: 0.0005

hgere:
  model_dir: saves/hgere/gsap/best
  base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
  model_type: hyper
  max_seq_length: 512
  per_gpu_eval_batch_size: 32
  n_iter: 3
  layernorm: true
  attn_self: true
  pre_filter_params:
    method: threshold
    value: 0.0125
```

---

## CLI reference

| Command | Description |
|---|---|
| `gsapere-train-pruner` | Train the span pruner |
| `gsapere-train-hgere` | Train the HGERE ERE model |
| `gsapere-pipeline` | Run the full two-stage pipeline on new documents |
| `gsapere-download-dataset` | Download supported datasets |
| `gsapere-tune-pruner` | Threshold sweep and optimisation for the pruner |
| `gsapere-fit-rulebased-pruner` | Fit a rule-based pruner baseline |
| `infer-fixed-spans` | Run HGERE on fixed (gold) spans |
| `infer-pruner-augmented` | Run HGERE on pruner-predicted spans |
| `gsap-ere-benchmark-pipeline` | Benchmark pipeline throughput |
| `gsapere-fix-gold-annos` | Add gold annotations to prediction files |
| `gsapere-analysis-ner-length-distribution` | Analyse entity length distributions |
| `gsapere-generate-pruner-docs` | Regenerate parameter docs in `documentation/api/` |

---

## Development

```bash
uv run pytest                          # run tests
uv run ruff format src/ tests/         # format
uv run ruff check src/ tests/          # lint
```

---

## Building and publishing

```bash
uv build                               # produces dist/ wheel + sdist
bash publish.sh                        # build + upload to PyPI (requires .pypi token file)
```

---

## Citation

Please cite this work and the original HGERE:

```bibtex
@article{Otto2026GSAP-ERE,
  title   = {{GSAP-ERE}: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning},
  author  = {Otto, Wolfgang and Gan, Lu and Upadhyaya, Sharmila and Karmakar, Saurav and Dietze, Stefan},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {38},
  pages   = {32600--32609},
  year    = {2026},
  month   = {Mar.},
  doi     = {10.1609/aaai.v40i38.40537},
  url     = {https://ojs.aaai.org/index.php/AAAI/article/view/40537},
}

@misc{yan2023joint,
  title         = {Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks},
  author        = {Zhaohui Yan and Songlin Yang and Wei Liu and Kewei Tu},
  year          = {2023},
  eprint        = {2310.17238},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
