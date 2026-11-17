#!/bin/bash
#SBATCH --job-name=conap_pretrained
#SBATCH --array=0-9              
#SBATCH --output=OutputLogs/conap_pretrained_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
DATASETS=$(for p in $PAIRS; do echo "${p%%=*}"; done | tr '\n' ' ')
PRE=$(for p in $PAIRS; do t=${p%%=*}; s=${p#*=}; echo -n "$t=$CKPT_DIR/${t}_from_${s} "; done)
CONFIGS="full forward"
RUN_ARGS="--pretrained $PRE"
source slurm/array_body.sh
