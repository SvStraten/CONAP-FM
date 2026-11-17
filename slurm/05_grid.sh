#!/bin/bash
#SBATCH --job-name=conap_grid
#SBATCH --array=0-29               # 10 logs times 3 seeds. Task t is log t modulo 10 and seed t divided by 10
#SBATCH --output=OutputLogs/conap_grid_%A_%a.txt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q
# Step 1b. Grid search on the 15% validation split, run BEFORE the main experiments.
# Three parameters with three values each, 36 combinations per log and seed (see grid_search.py). W = 50 was added after the first run, finished cells are skipped.
#   window 50 100 500 1000 | tau 5e-4 5e-3 5e-2 | cooldown 5 10 20 (epsilon fixed at 0.05). Learning rate and Adam steps stay as in Setup/hparams.py.
#   sbatch slurm/05_grid.sh
#   then, on the login node, from the repo root   python -m Analysis.select_grid      (writes Setup/final.py)
# run_experiments.py reads Setup/final.py by itself, so the main runs use the chosen values per log.
# Finished runs are skipped, so a timed out array can simply be submitted again.
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash slurm/05_grid.sh | wc -l     prints the 36 commands of one task
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
