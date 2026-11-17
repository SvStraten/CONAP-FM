#!/bin/bash
#SBATCH --job-name=conap_gridfull
#SBATCH --array=0-2                # one task per window size. Each task runs the 9 cells of tau x cooldown, one seed
#SBATCH --output=OutputLogs/conap_gridfull_%A_%a.txt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# Grid over W {50 100 500}, tau {5e-4 5e-3 5e-2} and cooldown {5 10 20}, epsilon fixed at 0.05 on IRO5000, evaluation stream (the 85% of the main table).
# One seed (40), 27 runs in total (see grid_full.py). LoRA rank and alpha are those of Table 1 (IRO5000 8/16).
#   sbatch slurm/06_grid_full.sh
#   SPLIT=full sbatch --export=ALL slurm/06_grid_full.sh            the whole log instead of the evaluation stream
#   LORA_16_32=1 sbatch --export=ALL slurm/06_grid_full.sh          rank 16 and alpha 32 instead
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash slurm/06_grid_full.sh | wc -l     prints the 9 commands of one task
# Finished runs are skipped, so a timed out array can simply be submitted again. Results go to runs/grid_full/results.
cd "${SLURM_SUBMIT_DIR:-.}"
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
