#!/bin/bash
# Step 2 and later. Submits every experiment, run it from the repo root on the login node.
#   bash slurm/setup_env.sh              once
#   sbatch slurm/00_preprocess.sh        wait until it finished
#   bash slurm/submit_all.sh             this script
set -e
mkdir -p OutputLogs runs checkpoints

P=$(sbatch --parsable slurm/02_pretrain.sh)
J1=$(sbatch --parsable slurm/10_main.sh)
J2=$(sbatch --parsable slurm/11_ablation.sh)
J3=$(sbatch --parsable slurm/12_triggers.sh)
J4=$(sbatch --parsable slurm/13_backbone.sh)
J5=$(sbatch --parsable --dependency=afterok:$P slurm/14_pretrained.sh)
DEPS="$J1:$J2:$J3:$J4:$J5"
# afterany, so that the analysis also runs when single tasks failed. Check OutputLogs for failures.
A=$(sbatch --parsable --dependency=afterany:$DEPS slurm/30_analyze.sh)
echo "submitted pretrain $P, main $J1, ablation $J2, triggers $J3, backbone $J4, pretrained $J5, analysis $A"
