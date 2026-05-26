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

```bash
uv run gsapere-download-dataset --list          # list available datasets
uv run gsapere-download-dataset gsap-ere
uv run gsapere-download-dataset scier
uv run gsapere-download-dataset scinlp
uv run gsapere-download-dataset scierc
```

---

## Training

Training is a two-step process: first train the pruner, then train HGERE on the pruner's output.

### Step 1 — Train the span pruner

```bash
uv run gsapere-train-pruner configs/train/gsap/train_gsap_pruner.yaml
```

Or via CLI flags:

```bash
uv run gsapere-train-pruner \
    --model_dir saves/pruner/gsap \
    --label_set gsap \
    --base_model_name_or_path pretrained_models/scibert_scivocab_uncased \
    --train_params__data_dir datasets/gsap-ere \
    --train_params__learning_rate 2e-5 \
    --train_params__num_train_epochs 10
```

After training, run pruner inference on train/dev/test to produce the input files for HGERE (see `scripts/pruner/`).

### Step 2 — Train HGERE

```bash
uv run gsapere-train-hgere configs/train/gsap/train_gsap_hgere.yaml
```

Or via CLI flags:

```bash
uv run gsapere-train-hgere \
    --model_dir saves/hgere/gsap \
    --label_set gsap \
    --base_model_name_or_path pretrained_models/scibert_scivocab_uncased \
    --ner_prediction_dir saves/pruner/gsap/output \
    --train_params__learning_rate 2e-5 \
    --train_params__num_train_epochs 10 \
    --train_params__per_gpu_train_batch_size 18
```

Both commands accept a YAML config file (positional argument) **or** individual `--field value` flags. Nested fields use `__` as a separator (e.g. `--train_params__learning_rate`).

Example YAML:

```yaml
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

### Full pipeline (pruner → HGERE)

```bash
CUDA_VISIBLE_DEVICES=0 uv run gsapere-pipeline \
    --config configs/inference/gsap-pipeline-best.yaml \
    --input input/ \
    --output output/
```

`--input` can be a `.jsonl` file or a directory of `.jsonl` files.

### Pruner inference only

```bash
bash scripts/pruner/gsap-ere/infer_gsap.sh
```

### Tune pruner threshold

```bash
uv run gsapere-tune-pruner --config config.yaml
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

## Citation

Please cite both this work (paper under review) and the original HGERE:

```bibtex
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
