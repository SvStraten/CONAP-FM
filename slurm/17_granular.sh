#!/bin/bash
#SBATCH --job-name=conap_granular
#SBATCH --array=0-9                # 10 logs, one seed. Task t is log t modulo 10
#SBATCH --output=OutputLogs/conap_granular_%A_%a.txt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# One run of CONAP-FM on every log (seed 40) to get the granular accuracy over the stream for the accuracy plots.
#   CONAP-FM (full), Tiny-LLM by default, or BACKBONE=distilgpt2 sbatch slurm/17_granular.sh
# The runs go to their own folder (runs/granular_<backbone>), so nothing from the earlier runs is skipped or overwritten.
# Every task writes its own file OutputLogs/granular/<backbone>__<log>__seed<seed>.csv, merge them afterwards with
#   python Analysis/merge_granular.py
# usage, from the repo root
#   sbatch slurm/17_granular.sh
#   BACKBONE=distilgpt2 sbatch slurm/17_granular.sh
#   SEEDS=41 sbatch slurm/17_granular.sh                                          another seed
#   CONFIGS="full forward full_ft" sbatch slurm/17_granular.sh                    other methods, for example from the ablation
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=6 bash slurm/17_granular.sh                     prints the commands of one task
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
