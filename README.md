# gsapere — Entity and Relation Extraction for Scientific Text

A two-stage pipeline for **entity and relation extraction (ERE)** on scientific text, built on top of [HGERE](https://github.com/yanzhh/HGERE).

The pipeline consists of:
1. **Span Pruner** — a lightweight binary classifier that scores all candidate n-grams and filters them down to a manageable set (target: ≥ 98 % entity recall)
2. **HGERE** — a heterogeneous graph neural network that jointly predicts entity types and relations on the pruned candidates

Supported datasets: **GSAP-ERE**, **SciER**, **SciNLP**, **SciERC**

---

## Requirements

- Python 3.9
- CUDA 12.8 (adjust `pyproject.toml` for other CUDA versions)
- A GPU with at least ~24 GB VRAM for default batch sizes (tested on A40 / 40 GB)

---

## Installation

Install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd HGERE
uv sync
source .venv/bin/activate
```

### Pretrained models

Download SciBERT (used by both stages):

```bash
# requires git-lfs
git lfs install
git clone https://huggingface.co/allenai/scibert_scivocab_uncased pretrained_models/scibert_scivocab_uncased
```

Or ModernBERT:

```bash
git clone https://huggingface.co/answerdotai/ModernBERT-base pretrained_models/modernbert_base
```

### Datasets

Download datasets with the bundled CLI command:

```bash
# list available datasets
uv run download-dataset --list

# download individual datasets
uv run download-dataset gsap-ere
uv run download-dataset scier
uv run download-dataset scierc
uv run download-dataset scinlp
```

See [documentation/download-dataset.md](documentation/download-dataset.md) for full options and dataset details.

---

## Training

Training is a two-step process: first train the pruner, then train HGERE on the pruner's output.

### Step 1 — Train the span pruner

```bash
uv run train-span-classifier configs/train/gsap/train_gsap_pruner.yaml
```

Or pass all parameters directly on the command line:

```bash
uv run train-span-classifier \
    --model_dir saves/pruner/gsap \
    --label_set gsap \
    --base_model_name_or_path pretrained_models/scibert_scivocab_uncased \
    --train_params__data_dir datasets/gsap-ere \
    --train_params__learning_rate 2e-5 \
    --train_params__num_train_epochs 10
```

After training the pruner, run inference to produce enriched dataset files for HGERE:

```bash
bash scripts/pruner/gsap-ere/train_rulebased_pruner.sh
```

### Step 2 — Train HGERE

```bash
uv run train-hgere configs/train/gsap/train_gsap_hgere.yaml
```

Or via CLI flags:

```bash
uv run train-hgere \
    --model_dir saves/hgere/gsap \
    --label_set gsap \
    --base_model_name_or_path pretrained_models/scibert_scivocab_uncased \
    --ner_prediction_dir saves/pruner/gsap/output \
    --train_params__learning_rate 2e-5 \
    --train_params__num_train_epochs 10 \
    --train_params__per_gpu_train_batch_size 18
```

### Config file format

Both commands accept a YAML config file. Example:

```yaml
# configs/train/gsap/train_gsap_hgere.yaml
schema_version: "1.0"
label_set: gsap
model_dir: saves/hgere/gsap
base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
ner_prediction_dir: saves/pruner/gsap/output

train_params:
  learning_rate: 2e-5
  num_train_epochs: 10
  per_gpu_train_batch_size: 18
```

---

## Inference

### Pruner inference

```bash
bash scripts/pruner/gsap-ere/infer_scier_gsap.sh
```

### HGERE inference (fixed spans)

```bash
bash scripts/hgere/gsap-ere/infer_fixed_spans_scier.sh
```

### Full pipeline (pruner + HGERE)

```bash
uv run run-pipeline configs/pipeline_gsap.yaml
```

---

## CLI Commands

| Command | Description |
|---|---|
| `train-span-classifier` | Train the span pruner |
| `train-hgere` | Train the HGERE ERE model |
| `download-dataset` | Download datasets |
| `infer-fixed-spans` | Run HGERE on fixed (gold) spans |
| `infer-pruner-augmented` | Run HGERE on pruner-predicted spans |
| `eval-rulebased-pruner` | Evaluate a rule-based pruner |
| `run-pipeline` | Run the full two-stage pipeline |
| `benchmark-pipeline` | Benchmark pipeline throughput |
| `generate-pruner-docs` | Regenerate API documentation |

---

## Development

Run tests:

```bash
uv run pytest
```

Format and lint:

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
```

---

## Building for PyPI

```bash
uv build
```

This produces a wheel and sdist in `dist/`. Upload with:

```bash
uv run twine upload dist/*
```

Or using the `uv` publish command (uv ≥ 0.4):

```bash
uv publish
```

---

## Citation

If you use this code, please cite the original HGERE paper:

```bibtex
@inproceedings{yan-etal-2023-partition,
    title     = "Partition-Then-Aggregate: A Two-Stage Parsing Method for Higher-Order Coreference Resolution",
    author    = "Yan, Zhenghao and others",
    booktitle = "Findings of ACL 2023",
    year      = "2023",
}
```

---

## License

MIT — see [LICENSE](LICENSE).
