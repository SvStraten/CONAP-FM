#!/bin/bash
#SBATCH --job-name=conap_ablation_gpt2
#SBATCH --array=0-99               # 10 logs times 10 seeds. Task t is log t modulo 10 and seed t divided by 10
#SBATCH --output=OutputLogs/conap_ablation_distilgpt2_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# Step 2c. Ablation of memory, CaGA and the importance mask (Forward is plain sequential LoRA).
#   full = trigger + memory + CaGA, forward = plain LoRA, neither = memory that is never used, with_mask = full + importance mask, no_caga = mask only
# Second backbone  BACKBONE=distilgpt2 sbatch slurm/11_ablation.sh
# usage, from the repo root
#   sbatch slurm/11_ablation.sh
#   sbatch --array=0,6 slurm/11_ablation.sh       only task 0 and 6, for a quick test
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=6 bash slurm/11_ablation.sh     prints the command of one task
# Results go to runs/all/results/<log>__<configuration>__seed<seed>.json, analyse with slurm/30_analyze.sh
cd "$SLURM_SUBMIT_DIR"
export BACKBONE=distilgpt2   # fixed in this script, nothing has to be passed on the command line
unset DRY_RUN
CONFIGS="full forward neither with_mask no_caga"
RUN_ARGS=""
source slurm/array_body.sh
