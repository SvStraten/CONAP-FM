#!/bin/bash
#SBATCH --job-name=conap_merge_granular
#SBATCH --output=OutputLogs/conap_merge_granular_%j.txt
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=mcs.gpu.q
# Merges the per-task granular files into OutputLogs/granular_accuracy.csv. Run it after slurm/17_granular.sh, for example
#   J1=$(sbatch --parsable slurm/17_granular.sh)
#   J2=$(BACKBONE=distilgpt2 sbatch --parsable slurm/17_granular.sh)
#   sbatch --dependency=afterany:$J1:$J2 slurm/18_merge_granular.sh
cd "${SLURM_SUBMIT_DIR:-.}"
USE_GPU=0 source slurm/env.sh
python Analysis/merge_granular.py
