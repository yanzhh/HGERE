# Inference
 * Tested on our old gpu server `spkgou01`

## install: 
 * 1. install uv
 * 2. `uv sync`

## infer based on config
 * for the example config I put docs.jsonl in the foler input

### 1. Prune (Generate candidates for entity mentions)
 * `CUDA_VISIBLE_DEVICES=3 uv run gsapere-train-pruner configs/inference/gsap/gsap-best-pruner.yaml`


### 2. ERE
 * `CUDA_VISIBLE_DEVICES=3 uv run gsapere-train-hgere configs/inference/gsap/gsap-best-hgere.yaml`

### Output is in 
 * `output/ent_pred_docs.json`



# hints
 * folder names and model paths are stored in the named config files. 
