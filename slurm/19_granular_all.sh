#!/bin/bash

cd "$(dirname "$0")/.."
J1=$(sbatch --parsable slurm/17_granular.sh)
J2=$(BACKBONE=distilgpt2 sbatch --parsable slurm/17_granular.sh)
J3=$(sbatch --parsable --dependency=afterany:$J1:$J2 slurm/18_merge_granular.sh)
echo "submitted  tinyllm $J1  distilgpt2 $J2  merge $J3 (starts after both)"
