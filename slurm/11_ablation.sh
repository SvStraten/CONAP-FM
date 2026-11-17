#!/bin/bash
#SBATCH --job-name=conap_ablation
#SBATCH --array=0-99               
#SBATCH --output=OutputLogs/conap_ablation_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "$SLURM_SUBMIT_DIR"
CONFIGS="full forward neither with_mask no_caga"
RUN_ARGS=""
source slurm/array_body.sh
