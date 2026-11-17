#!/bin/bash
#SBATCH --job-name=conap_smoke
#SBATCH --output=OutputLogs/conap_smoke_%j.txt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# Optional. Quick end to end check of the main configurations on the first 3000 events of one log.
#   sbatch slurm/01_smoke.sh
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python smoke_test.py --dataset RequestForPayment
