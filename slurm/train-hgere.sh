#!/bin/bash
#SBATCH --job-name=hgere
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=5-01:00:00
#SBATCH --mem 20G
#SBATCH --output=slurm/output/gpu_job_%j.out
#SBATC --nodes 1                  # Number of nodes to request
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 10 > slurm/output/gpu_usage_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID" EXIT

#uv run gsapere-train-hgere configs/$1/train/hgere/best_seeds.yaml
uv run gsapere-pipeline \
	--config configs/$1/infer/best_seeds.yaml \
    --skip_train \
	--input datasets/gsap-ere/2025-05-15 \
	--output datasets/gsap-ere/pred_${1}
uv run gsapere-pipeline \
	--config configs/$1/infer/best_seeds.yaml \
    --skip_train \
	--input datasets/scier/ \
	--output datasets/scier/pred_${1}
uv run gsapere-pipeline \
	--config configs/$1/infer/best_seeds.yaml \
    --skip_train \
	--input datasets/scinlp/ \
	--output datasets/scinlp/pred_${1}
