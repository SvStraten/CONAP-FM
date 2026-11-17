#!/bin/bash
#SBATCH --job-name=conap_granular
#SBATCH --array=0-9                
#SBATCH --output=OutputLogs/conap_granular_%A_%a.txt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "${SLURM_SUBMIT_DIR:-.}"
export BACKBONE="${BACKBONE:-tinyllm}"
export OUT="${OUT:-runs/granular_${BACKBONE}}"
export SEEDS="${SEEDS:-40}"
source slurm/env.sh
if [ -z "$CONFIGS" ]; then
  CONFIGS="full"
fi
export GRANULAR_DIR="${GRANULAR_DIR:-OutputLogs/granular}"
RUN_ARGS=""
source slurm/array_body.sh
