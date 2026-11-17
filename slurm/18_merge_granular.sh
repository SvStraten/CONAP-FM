#!/bin/bash
#SBATCH --job-name=conap_merge_granular
#SBATCH --output=OutputLogs/conap_merge_granular_%j.txt
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=mcs.gpu.q

cd "${SLURM_SUBMIT_DIR:-.}"
USE_GPU=0 source slurm/env.sh
python Analysis/merge_granular.py
