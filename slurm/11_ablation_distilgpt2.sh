#!/bin/bash
#SBATCH --job-name=conap_ablation_gpt2
#SBATCH --array=0-99               
#SBATCH --output=OutputLogs/conap_ablation_distilgpt2_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "$SLURM_SUBMIT_DIR"
export BACKBONE=distilgpt2   
unset DRY_RUN
CONFIGS="full forward neither with_mask no_caga"
RUN_ARGS=""
source slurm/array_body.sh
