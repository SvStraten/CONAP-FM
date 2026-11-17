#!/bin/bash
# Sourced by all scripts. Adjust the module names if your cluster differs.
if command -v module >/dev/null 2>&1; then
    module purge
    module load Umbrella/2024
    module load Python/3.12.3-GCCcore-13.3.0
    if [ "${USE_GPU:-1}" = "1" ]; then module load CUDA/12.1.1; fi
fi
VENV="${VENV:-$HOME/venvs/conapfm}"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi

# compute nodes may have no internet, the model is cached by setup_env.sh on the login node
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# the ten public logs of the paper and the ten seeds. Override with DATASETS="..." SEEDS="..." for a test.
DATASETS="${DATASETS:-IRO5000 ORI5000 ROI5000 OIR5000 RIO5000 BPIC2015_Recurrent RequestForPayment InternationalDeclarations DomesticDeclarations BPIC2017}"
SEEDS="${SEEDS:-40 41 42 43 44 45 46 47 48 49}"

# Backbone. BACKBONE=tinyllm (default) or BACKBONE=distilgpt2, for example  BACKBONE=distilgpt2 sbatch slurm/10_main.sh
# Every backbone has its own result, checkpoint and analysis folder, so the two never mix.
BACKBONE="${BACKBONE:-tinyllm}"
case "$BACKBONE" in
    tinyllm)    MODEL_NAME="${MODEL_NAME:-arnir0/Tiny-LLM}"; OUT="${OUT:-runs/all}"; CKPT_DIR="${CKPT_DIR:-checkpoints}"; ANALYSIS="${ANALYSIS:-runs/analysis}" ;;
    distilgpt2) MODEL_NAME="${MODEL_NAME:-distilbert/distilgpt2}"; OUT="${OUT:-runs/distilgpt2}"; CKPT_DIR="${CKPT_DIR:-checkpoints/distilgpt2}"; ANALYSIS="${ANALYSIS:-runs/analysis_distilgpt2}" ;;
    *) echo "unknown BACKBONE $BACKBONE (tinyllm or distilgpt2)"; exit 1 ;;
esac
export BACKBONE MODEL_NAME OUT CKPT_DIR ANALYSIS

# Pretraining pairs TARGET=SOURCE (the log that is pretrained on). Fill in the pairs of Table 1.
PAIRS="${PAIRS:-RequestForPayment=PrepaidTravelCost}"
