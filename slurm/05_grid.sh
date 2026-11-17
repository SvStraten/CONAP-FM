#!/bin/bash
#SBATCH --job-name=conap_grid
#SBATCH --array=0-29               
#SBATCH --output=OutputLogs/conap_grid_%A_%a.txt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
DS_ARR=($DATASETS); GSEEDS=(${GRID_SEEDS:-40 41 42})
ND=${#DS_ARR[@]}
t=${SLURM_ARRAY_TASK_ID:?this script must be submitted with sbatch --array}
DS=${DS_ARR[$((t % ND))]}; SEED=${GSEEDS[$((t / ND))]}
if [ -z "$SEED" ]; then echo "array index $t is outside datasets x seeds"; exit 1; fi
echo "task $t | dataset $DS | seed $SEED"
CMD=(python grid_search.py --datasets "$DS" --seeds "$SEED" --out runs/grid)
if [ -n "$DRY_RUN" ]; then CMD+=(--dry-run); fi
"${CMD[@]}"
