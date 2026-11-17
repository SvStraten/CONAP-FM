#!/bin/bash

source slurm/env.sh
DS_ARR=($DATASETS); SEED_ARR=($SEEDS)
ND=${#DS_ARR[@]}
t=${SLURM_ARRAY_TASK_ID:?this script must be submitted with sbatch --array}
DS=${DS_ARR[$((t % ND))]}
SEED=${SEED_ARR[$((t / ND))]}
if [ -z "$SEED" ]; then echo "array index $t is outside datasets x seeds ($ND x ${#SEED_ARR[@]})"; exit 1; fi
if [ -n "$GRANULAR_DIR" ]; then mkdir -p "$GRANULAR_DIR"; export GRANULAR_OUT="$GRANULAR_DIR/${BACKBONE}__${DS}__seed${SEED}.csv"; fi
echo "task $t | dataset $DS | seed $SEED | configs $CONFIGS | backbone $MODEL_NAME | out $OUT"
CMD=(python run_experiments.py --datasets "$DS" --seeds "$SEED" --configs $CONFIGS --out "$OUT" --model-name "$MODEL_NAME" $RUN_ARGS)
if [ -n "$DRY_RUN" ]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
