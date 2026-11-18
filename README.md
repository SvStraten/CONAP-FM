# CONAP-FM

**Sustainable Online Continual Learning for Foundation Process Prediction Models**

Code for the ICPM paper *Towards Foundation Process Prediction Models: Sustainable Online Continual Learning*.

CONAP-FM keeps a pretrained language model frozen and adapts it online to a stream of process events for next activity prediction.
Only LoRA adapters and a linear head are trained, on top of Tiny-LLM (about 10M parameters) or DistilGPT2 (about 82M parameters).

![Architecture of CONAP-FM](architecture.png)

1. **Test then train.** Every window of the event stream is first used for prediction and then for training.
2. **Plateau trigger.** When the variance of the last N = 5 window losses drops below tau, the current concept is consolidated. A cooldown of C windows follows. No drift detector or task boundary is needed.
3. **Gradient memory and CaGA.** Consolidated gradient subspaces are kept as an SVD basis (95% energy). If the share rho of a new gradient inside this memory exceeds epsilon = 0.05, the part inside the memory is removed (orthogonal projection) before the Adam step, so new concepts do not overwrite old ones.

## Results

Accuracy over the event stream (evaluation split, seed 40) for CONAP-FM with Tiny-LLM and the competing methods. Dashed lines mark known drifts.
In the paper, CONAP-FM reaches the highest accuracy on nine of the ten logs (Table 2).

![Accuracy at a given event index](accuracy_at_given_index.png)

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The backbones `arnir0/Tiny-LLM` and `distilbert/distilgpt2` are downloaded from the Hugging Face hub on first use. A GPU is recommended.

## Data

The repository contains no event logs. Put each log in `Data/<name>.csv` with the columns `case`, `event` and `completeTime`.
The ten logs and their names are listed in [`Data/README.md`](Data/README.md).
Hyperparameters are chosen on the first 15% of the cases (`--data-split validation`). All results use the remaining 85% (`--data-split evaluation`).

## Usage

```bash
python smoke_test.py --dataset DomesticDeclarations                 
python run_experiments.py --datasets DomesticDeclarations --configs full --seeds 40 41 42 --out runs/all
python run_experiments.py --datasets DomesticDeclarations --configs full \
    --model-name distilbert/distilgpt2 --out runs/distilgpt2        # second backbone
```

## SLURM

`slurm/` has one job array script per experiment (main runs, ablation, triggers, backbones, pretraining, grids) and `submit_all.sh` to submit everything.
Edit the module names and partition in `slurm/env.sh` and the `#SBATCH` lines first. `BACKBONE=distilgpt2` switches the backbone. Seeds are 40 to 49.
