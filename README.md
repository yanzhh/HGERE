# Inference
 * Tested on our old gpu server `spkgou01`

## install: 
 * 1. install uv
 * 2. `uv sync`

## infer based on config
 * for the example config I put docs.jsonl in the foler input

## Apply the pipeline on a file or folder
 * `CUDA_VISIBLE_DEVICES=3 uv run gsapere-pipeline --config configs/inference/gsap-pipeline-best.yaml --input input --output output` 
 * All jsonl files in the input folder are processed
 * If --input is a file, only this file is processed
