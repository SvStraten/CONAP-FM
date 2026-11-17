#!/bin/bash
#SBATCH --job-name=conap_triggers
#SBATCH --array=0-99              
#SBATCH --output=OutputLogs/conap_triggers_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "$SLURM_SUBMIT_DIR"
CONFIGS="full oracle fixed adwin page_hinkley"
RUN_ARGS=""
if [ "${BACKBONE:-tinyllm}" = "distilgpt2" ]; then
  echo "BACKBONE=distilgpt2 only runs slurm/10_main.sh (CONAP-FM) and slurm/11_ablation.sh. Nothing to do here."; exit 0
fi
source slurm/array_body.sh
