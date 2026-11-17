#!/bin/bash
#SBATCH --job-name=conap_prep
#SBATCH --output=OutputLogs/conap_prep_%j.txt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mcs.default.q

mkdir -p OutputLogs runs checkpoints
cd "$SLURM_SUBMIT_DIR"
USE_GPU=0 source slurm/env.sh

SOURCES=$(for p in $PAIRS; do echo "${p#*=}"; done)
for ds in $DATASETS $SOURCES; do
    echo "preprocessing $ds"
    python -c "from Utils.preprocess import preprocess; preprocess('Data/$ds.csv')" || echo "FAILED $ds"
done
echo "Preprocessing finished."
