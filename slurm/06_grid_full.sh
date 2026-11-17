#!/bin/bash
#SBATCH --job-name=conap_gridfull
#SBATCH --array=0-2                
#SBATCH --output=OutputLogs/conap_gridfull_%A_%a.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

source slurm/env.sh
DS=${GRID_LOG:-IRO5000}
WINS=(50 100 500)
t=${SLURM_ARRAY_TASK_ID:?this script must be submitted with sbatch --array}
if [ "$t" -ge 3 ]; then echo "array index $t is outside 0-2"; exit 1; fi
W=${WINS[$t]}
echo "task $t | dataset $DS | W $W | epsilon 0.05 | split ${SPLIT:-evaluation}"
CMD=(python grid_full.py --datasets "$DS" --window "$W" --epsilon 0.05 --seeds "${GRID_SEED:-40}" --data-split "${SPLIT:-evaluation}" --out runs/grid_full)
if [ "$LORA_16_32" = "1" ]; then CMD+=(--lora-16-32); fi
if [ -n "$DRY_RUN" ]; then CMD+=(--dry-run); fi
"${CMD[@]}"
