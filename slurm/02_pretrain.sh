#!/bin/bash
#SBATCH --job-name=conap_pretrain
#SBATCH --output=OutputLogs/conap_pretrain_%j.txt
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=mcs.gpu.q

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
for pair in $PAIRS; do
    TARGET=${pair%%=*}; SOURCE=${pair#*=}
    R=$(python -c "from Setup.hparams import HPARAMS; print(HPARAMS['$TARGET']['r'])")
    A=$(python -c "from Setup.hparams import HPARAMS; print(HPARAMS['$TARGET']['alpha'])")
    DIR=$CKPT_DIR/${TARGET}_from_${SOURCE}
    if [ -f "$DIR/offline_lora.pt" ]; then echo "found $DIR, skipping"; continue; fi
    echo "pretraining on $SOURCE for $TARGET (rank $R, alpha $A)"
    python -m Methods.CONAP_FM.pretrain_offline --dataset "Data/$SOURCE.csv" --output-dir "$DIR" \
        --epochs 10 --batch-size 32 --calora-energy 0.95 --calora-batches 50 --lora-r "$R" --lora-alpha "$A" --seed 40 --model-name "$MODEL_NAME"
done
