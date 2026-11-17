#!/bin/bash
#SBATCH --job-name=conap_smoke
#SBATCH --output=OutputLogs/conap_smoke_%j.txt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python smoke_test.py --dataset RequestForPayment
