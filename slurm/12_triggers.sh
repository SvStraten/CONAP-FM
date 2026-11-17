#!/bin/bash
#SBATCH --job-name=conap_triggers
#SBATCH --array=0-99               # 10 logs times 10 seeds. Task t is log t modulo 10 and seed t divided by 10
#SBATCH --output=OutputLogs/conap_triggers_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# Step 2d. What triggers consolidation. full = plateau trigger of the method, oracle = true drift points, fixed = every 20 windows, adwin and page_hinkley = drift detectors on the window loss.
# adwin and page_hinkley need the python package river (requirements.txt).
# Second backbone  BACKBONE=distilgpt2 sbatch slurm/12_triggers.sh
# usage, from the repo root
#   sbatch slurm/12_triggers.sh
#   sbatch --array=0,6 slurm/12_triggers.sh       only task 0 and 6, for a quick test
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=6 bash slurm/12_triggers.sh     prints the command of one task
# Results go to runs/all/results/<log>__<configuration>__seed<seed>.json, analyse with slurm/30_analyze.sh
cd "$SLURM_SUBMIT_DIR"
CONFIGS="full oracle fixed adwin page_hinkley"
RUN_ARGS=""
if [ "${BACKBONE:-tinyllm}" = "distilgpt2" ]; then
  echo "BACKBONE=distilgpt2 only runs slurm/10_main.sh (CONAP-FM) and slurm/11_ablation.sh. Nothing to do here."; exit 0
fi
source slurm/array_body.sh
