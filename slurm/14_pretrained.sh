#!/bin/bash
#SBATCH --job-name=conap_pretrained
#SBATCH --array=0-9              # (number of pairs in PAIRS) times 10 seeds minus 1. Task t is pair t modulo pairs and seed t divided by pairs
#SBATCH --output=OutputLogs/conap_pretrained_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# Step 3. CONAP-FM and Forward starting from the pretrained adapters (and memory subspace), tags full_pre and forward_pre.
# Needs slurm/02_pretrain.sh to be finished. The target logs are taken from PAIRS in slurm/env.sh.
#   sbatch --dependency=afterok:<pretrain job id> slurm/14_pretrained.sh
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
DATASETS=$(for p in $PAIRS; do echo "${p%%=*}"; done | tr '\n' ' ')
PRE=$(for p in $PAIRS; do t=${p%%=*}; s=${p#*=}; echo -n "$t=$CKPT_DIR/${t}_from_${s} "; done)
CONFIGS="full forward"
RUN_ARGS="--pretrained $PRE"
source slurm/array_body.sh
