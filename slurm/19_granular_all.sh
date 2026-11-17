#!/bin/bash
# One command for the granular accuracy file. Run it on the login node from the repo root
#   bash slurm/19_granular_all.sh
# It submits slurm/17_granular.sh for TinyLLM and for DistilGPT2 (one run per method and log, seed 40) and a merge job that starts
# when both have ended. The result is OutputLogs/granular_accuracy.csv.
cd "$(dirname "$0")/.."
J1=$(sbatch --parsable slurm/17_granular.sh)
J2=$(BACKBONE=distilgpt2 sbatch --parsable slurm/17_granular.sh)
J3=$(sbatch --parsable --dependency=afterany:$J1:$J2 slurm/18_merge_granular.sh)
echo "submitted  tinyllm $J1  distilgpt2 $J2  merge $J3 (starts after both)"
