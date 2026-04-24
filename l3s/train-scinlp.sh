#!/bin/bash
#SBATCH --job-name=ottowg_gsap-ere_hgere
#SBATCH --gpus=1  # 1 GPU anfordern
#SBATCH --nodes=1
#SBATCH --time=8-20:00:00  # Laufzeitbegrenzung (z.B. 10 Minuten)
#SBATCH --cpus-per-task=16
#SBATCH --mem 30G
#SBATCH --output=l3s/logs/gpu_job_%j.out  # Ausgabe-Datei

#nvidia-smi  # GPU-Info anzeigen
uv run gsapere-train-hgere configs/scinlp/train/hgere/best_seeds.yaml
