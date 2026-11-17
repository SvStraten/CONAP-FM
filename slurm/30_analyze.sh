#!/bin/bash
#SBATCH --job-name=conap_analyze
#SBATCH --output=OutputLogs/conap_analyze_%j.txt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=mcs.default.q

cd "$SLURM_SUBMIT_DIR"
USE_GPU=0 source slurm/env.sh
R=$OUT/results
A=$ANALYSIS      
python -m Analysis.analyze --results $R --reference full --out $A/main --tags full
python -m Analysis.analyze --results $R --reference full --out $A/ablation \
    --tags full forward neither with_mask no_caga
python -m Analysis.analyze --results $R --reference full --out $A/triggers \
    --tags full oracle fixed adwin page_hinkley
python -m Analysis.analyze --results $R --reference full --out $A/backbone \
    --tags full forward full_ft random_full random_full_lowlr random_lora
python -m Analysis.analyze --results $R --reference full --out $A/pretrained \
    --tags full full_pre forward forward_pre
