#!/bin/bash
if command -v module >/dev/null 2>&1; then
    module purge
    module load Umbrella/2024
    module load Python/3.12.3-GCCcore-13.3.0
    if [ "${USE_GPU:-1}" = "1" ]; then module load CUDA/12.1.1; fi
fi
VENV="${VENV:-$HOME/venvs/conapfm}"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

DATASETS="${DATASETS:-IRO5000 ORI5000 ROI5000 OIR5000 RIO5000 BPIC2015_Recurrent RequestForPayment InternationalDeclarations DomesticDeclarations BPIC2017}"
SEEDS="${SEEDS:-40 41 42 43 44 45 46 47 48 49}"

BACKBONE="${BACKBONE:-tinyllm}"
case "$BACKBONE" in
    tinyllm)    MODEL_NAME="${MODEL_NAME:-arnir0/Tiny-LLM}"; OUT="${OUT:-runs/all}"; CKPT_DIR="${CKPT_DIR:-checkpoints}"; ANALYSIS="${ANALYSIS:-runs/analysis}" ;;
    distilgpt2) MODEL_NAME="${MODEL_NAME:-distilbert/distilgpt2}"; OUT="${OUT:-runs/distilgpt2}"; CKPT_DIR="${CKPT_DIR:-checkpoints/distilgpt2}"; ANALYSIS="${ANALYSIS:-runs/analysis_distilgpt2}" ;;
    *) echo "unknown BACKBONE $BACKBONE (tinyllm or distilgpt2)"; exit 1 ;;
esac
export BACKBONE MODEL_NAME OUT CKPT_DIR ANALYSIS

PAIRS="${PAIRS:-RequestForPayment=PrepaidTravelCost}"
