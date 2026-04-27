#nvidia-smi  # GPU-Info anzeigen
uv run gsapere-train-hgere configs/scinlp/train/hgere/best_seeds.yaml
#!/bin/bash
#SBATCH --job-name=hgere
#SBATCH --nodes=1
#SBATCH --gpus=1  # 1 GPU anfordern
#SBATCH --cpus-per-task=32
#SBATCH --time=2-01:00:00
#SBATCH --mem 30G
#SBATCH --output=l3s/logs/gpu_job_%j.out  # Ausgabe-Datei

uv run gsapere-train-hgere configs/$1/train/hgere/best_seeds.yaml
uv run gsapere-pipeline \
       --config configs/$1/infer/best_seeds.yaml \
       --input datasets/gsap-ere/ \
       --output datasets/gsap-ere/pred_gsap-ere
uv run gsapere-pipeline \
       --config configs/$1/infer/best_seeds.yaml \
       --input datasets/scier/ \
       --output datasets/scier/pred_gsap-ere
uv run gsapere-pipeline \
       --config configs/$1/infer/best_seeds.yaml \
       --input datasets/scinlp/ \
       --output datasets/scinlp/pred_gsap-ere
