#!/bin/bash
# Run ONCE on the login node (needs internet), from the repo root. bash slurm/setup_env.sh
# Creates the virtual environment, installs the packages and caches Tiny-LLM for the offline compute nodes.
set -e
module purge
module load Umbrella/2024
module load Python/3.12.3-GCCcore-13.3.0
VENV="${VENV:-$HOME/venvs/conapfm}"
mkdir -p "$(dirname "$VENV")"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
# The default torch wheel is built for CUDA 13 and has no kernels for V100 (compute capability 7.0).
# The CUDA 12.6 build still supports it. TORCH_INDEX can be changed for other GPUs.
pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
pip install -r requirements.txt
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
python - <<'PY'
from transformers import AutoConfig, AutoModel
AutoConfig.from_pretrained("arnir0/Tiny-LLM")
AutoModel.from_pretrained("arnir0/Tiny-LLM")
print("Tiny-LLM cached")
AutoConfig.from_pretrained("distilbert/distilgpt2")
AutoModel.from_pretrained("distilbert/distilgpt2")
print("distilgpt2 cached")
PY
mkdir -p OutputLogs results cache runs checkpoints
echo "Done. Next, sbatch slurm/00_preprocess.sh"
