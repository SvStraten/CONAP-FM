# CONAP-FM: Sustainable Online Continual Learning for Foundation Process Prediction Models

Code of the paper *Towards Foundation Process Prediction Models: Sustainable Online Continual Learning* (ICPM).

CONAP-FM keeps a pretrained language model frozen and adapts it online to a stream of process events with LoRA adapters.
Training and drift handling follow three ideas.

1. **Parameter efficient adaptation.** Only the LoRA adapters and a linear next activity head are trained, either on top of Tiny-LLM (about 10M parameters) or DistilGPT2 (about 82M parameters).
2. **Unsupervised plateau trigger.** The loss of each window is tested before it is trained on (test then train). When the variance of the last N = 5 window losses falls below a threshold tau, the adapters have settled and the gradient subspace of the current concept is consolidated. A cooldown of C windows prevents repeated triggers. No drift detector and no task boundaries are needed.
3. **Gradient memory and CaGA.** Consolidated subspaces are stored as an SVD basis (95% energy, new directions are merged and the memory is capped at half of the columns). Causal aware gradient adaptation (CaGA) projects the new gradient orthogonally to the memory when the share rho of the gradient that lies in the memory exceeds epsilon = 0.05, so that new concepts are learned without overwriting old ones. The adapted gradient is applied with Adam.

The importance mask of earlier versions is not part of the method. It is only kept as an ablation flag (`--paca`).

## Repository structure

```text
Methods/CONAP_FM/   run.py              one run (one log, one seed, one configuration)
                    engine.py           test then train loop, plateau trigger and the alternative triggers of the ablation
                    calora_handler.py   SVD gradient memory and CaGA
                    pretrain_offline.py offline pretraining of the adapters on a second log
Setup/              hparams.py          per log settings of Table 1 (lr, rank, alpha, tau, W) and shared settings
                    final.py            W, tau, epsilon and cooldown C per log used by run_experiments.py
                    drifts.py           known drift points of the logs (only for the oracle trigger and the drift analysis)
                    LogFile.py, transform.py, sampler.py, setting.py, method.py   log handling
Utils/              preprocess.py, stream.py, metrics.py
Analysis/           analyze.py          Wilcoxon + Holm, Friedman + Nemenyi, mean ranks, drift recovery, early stream, efficiency
                    runtime_table.py, runtime_throughput.py   runtime (Fig. 5) and throughput (Table 4) tables
                    select_grid.py, plot_grid.py, plot_grid_full.py, merge_granular.py
Data/               raw event logs (see Data/README.md)
Results/            runtime and throughput tables of the paper (csv, md, tex)
slurm/              one script per experiment for a SLURM cluster
run_experiments.py  datasets x configurations x seeds
grid_search.py      grid of W, tau and cooldown on the 15% validation split
grid_full.py        sensitivity grid on the evaluation stream
smoke_test.py       quick end to end check
```

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

A GPU is recommended. The backbones `arnir0/Tiny-LLM` and `distilbert/distilgpt2` are downloaded from the Hugging Face hub on first use.

## Data

Event logs are `.csv` files in `Data/` with the columns `case` (case id), `event` (activity) and `completeTime` (timestamp).
The repository includes no logs. See `Data/README.md` for the expected files and how to obtain all ten logs of the paper.
Hyperparameters are chosen on the first 15% of the cases (`--data-split validation`), all reported results use the remaining 85% (`--data-split evaluation`).
Each prefix is a vector of integer activity ids, most recent event first, zero padded to the length of the longest case.
The ids index the frozen token embeddings of the backbone and a new linear head on the last position is trained online.

## Quick start

```bash
python smoke_test.py --dataset DomesticDeclarations           # short end to end check, accuracies are meaningless
python -m Methods.CONAP_FM.run --dataset Data/DomesticDeclarations.csv --data-split evaluation \
    --window-size 100 --lr 2e-4 --lora-r 256 --lora-alpha 512 --loss-variance-threshold 5e-4 --seed 40
```

The settings of the second command are the ones of Table 1 for this log. `run_experiments.py` fills them in automatically from `Setup/hparams.py` and `Setup/final.py`.

```bash
python run_experiments.py --datasets DomesticDeclarations --configs full --seeds 40 41 42 --out runs/all
python run_experiments.py --configs all_ablation --dry-run            # print the commands only
python run_experiments.py --datasets DomesticDeclarations --configs full --model-name distilbert/distilgpt2 --out runs/distilgpt2
```

Each run writes `runs/<out>/results/<log>__<configuration>__seed<seed>.json`. Finished runs are skipped, so an interrupted job can be restarted.

### Configurations

| Tag | Description |
|---|---|
| `full` | CONAP-FM, plateau trigger, memory and CaGA |
| `forward` | plain sequential LoRA without memory |
| `neither` | trigger and memory, but the memory is never used |
| `with_mask`, `no_caga` | importance mask added to the method, mask without CaGA (ablation) |
| `oracle`, `fixed`, `adwin`, `page_hinkley` | other consolidation triggers (true drift points, every 20 windows, drift detectors on the window loss) |
| `full_ft`, `random_full`, `random_full_lowlr`, `random_lora` | full fine tuning, random initialisation and a random frozen backbone, to separate the effect of the backbone from drift awareness |
| `<tag>_pre` | start from adapters pretrained on a second log, `--pretrained TARGET=checkpoint_dir` |

Groups: `all_ablation`, `triggers`, `backbone`, `all`.

### Pretraining

```bash
python -m Methods.CONAP_FM.pretrain_offline --dataset Data/PrepaidTravelCost.csv \
    --output-dir checkpoints/RequestForPayment_from_PrepaidTravelCost --lora-r 128 --lora-alpha 256 --epochs 10 --seed 40
python run_experiments.py --datasets RequestForPayment --configs full forward \
    --pretrained RequestForPayment=checkpoints/RequestForPayment_from_PrepaidTravelCost
```

The head is not reused online, because the activity set of the target log differs.

## Analysis

```bash
python -m Analysis.analyze --results runs/all/results --reference full --out runs/analysis/main --tags full
python -m Analysis.analyze --results runs/all/results --reference full --out runs/analysis/ablation --tags full forward neither with_mask no_caga
```

Written files: `accuracy_table.csv`, `wilcoxon.csv` (paired Wilcoxon signed-rank test of every configuration against the reference, Holm corrected), `friedman_nemenyi.csv`, `mean_ranks.csv`, `drift_recovery.csv`, `early_stream.csv`, `efficiency.csv`, `report.txt`.
If you add result files of competing methods in the same json format to the folder, `--baselines <tags>` also writes `wilcoxon_main.csv`, which tests the reference against the strongest of them per log.
The competing methods of the paper (DoNothing, LastDrift, DynaTrainCDD, TFCLPM, CNAPwP) are not part of this repository. The competitor columns in `Results/` come from the runs of the paper.

## Cluster (SLURM)

The scripts in `slurm/` run the experiments as job arrays. Adjust the module names and the partition in `slurm/env.sh` and the `#SBATCH` lines to your cluster.

| Script | Experiment |
|---|---|
| `setup_env.sh` | once, creates the environment and caches the backbone |
| `00_preprocess.sh` | preprocesses and caches all logs |
| `01_smoke.sh` | end to end check |
| `02_pretrain.sh`, `14_pretrained.sh` | offline pretraining and the runs that start from it |
| `05_grid.sh` | grid of W, tau and C on the validation split, then `python -m Analysis.select_grid` |
| `06_grid_full.sh` | sensitivity grid on the evaluation stream |
| `10_main.sh`, `10_main_distilgpt2.sh` | CONAP-FM on all logs and seeds, Tiny-LLM and DistilGPT2 |
| `11_ablation.sh`, `11_ablation_distilgpt2.sh` | ablation of memory and CaGA |
| `12_triggers.sh`, `13_backbone.sh` | trigger and backbone variants |
| `17_granular.sh`, `18_merge_granular.sh`, `19_granular_all.sh` | accuracy over the stream for the plots |
| `30_analyze.sh` | all statistics |
| `submit_all.sh` | submits everything with the right dependencies |

Use `DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash slurm/10_main.sh` to print the command of one array task. `BACKBONE=distilgpt2` switches the backbone. Seeds are 40 to 49.

## Runtime results

End-to-end runtime of the whole evaluation stream (training and inference) in minutes, mean ± standard deviation over 10 seeds (40 to 49). The number of seeds per cell is in brackets. Medians, minimum and maximum are in `Results/runtime_results.csv` and a LaTeX version is in `Results/runtime_results.tex`.

| Log | CONAP-FM TinyLLM | CONAP-FM DistilGPT2 | TFCLPM | CNAPwP | DynaTrainCDD | LastDrift |
|---|---|---|---|---|---|---|
| IRO5000 | 1.92 ± 0.17 (10) | 8.79 ± 0.56 (10) | 0.22 ± 0.01 (10) | 1.10 ± 0.09 (10) | 0.21 ± 0.01 (10) | 1.02 ± 0.11 (10) |
| ORI5000 | 2.16 ± 0.25 (10) | 12.51 ± 0.60 (10) | 0.24 ± 0.02 (10) | 1.26 ± 0.15 (10) | 0.22 ± 0.01 (10) | 1.08 ± 0.09 (10) |
| ROI5000 | 1.85 ± 0.11 (10) | 7.32 ± 0.43 (10) | 0.26 ± 0.02 (10) | 1.10 ± 0.14 (10) | 0.20 ± 0.02 (10) | 1.05 ± 0.15 (10) |
| OIR5000 | 2.05 ± 0.15 (10) | 10.80 ± 0.75 (10) | 0.29 ± 0.02 (10) | 1.22 ± 0.12 (10) | 0.21 ± 0.01 (10) | 1.22 ± 0.08 (10) |
| RIO5000 | 1.93 ± 0.18 (10) | 7.53 ± 0.31 (10) | 0.22 ± 0.02 (10) | 1.14 ± 0.12 (10) | 0.20 ± 0.02 (10) | 1.05 ± 0.12 (10) |
| BPI15-REC | 5.62 ± 0.30 (10) | 77.98 ± 5.37 (10) | 0.65 ± 0.07 (10) | 3.76 ± 0.32 (10) | 0.77 ± 0.06 (10) | 5.05 ± 0.23 (10) |
| BPI20-RFP | 1.37 ± 0.13 (10) | 5.79 ± 0.66 (10) | 0.18 ± 0.01 (10) | 1.08 ± 0.11 (10) | 0.15 ± 0.01 (10) | 5.94 ± 0.68 (10) |
| BPI20-ID | 2.80 ± 0.22 (10) | 15.25 ± 0.86 (10) | 0.29 ± 0.02 (10) | 1.57 ± 0.15 (10) | 0.27 ± 0.01 (10) | 11.81 ± 1.09 (10) |
| BPI20-DD | 2.13 ± 0.21 (10) | 10.22 ± 0.41 (10) | 0.24 ± 0.02 (10) | 1.54 ± 0.18 (10) | 0.23 ± 0.02 (10) | 10.80 ± 0.64 (10) |
| BPI17 | 11.40 ± 0.53 (10) | 229.79 ± 105.99 (10) | 0.67 ± 0.05 (10) | 7.03 ± 0.54 (10) | 1.30 ± 0.09 (10) | 72.79 ± 4.95 (10) |

The DistilGPT2 mean on BPI17 is inflated by one slow run (seed 41, 521 min), the median in the csv is the safer summary there.
Regenerate with `python -m Analysis.runtime_table --tinyllm <csv> --distilgpt2 <csv> --out Results`.
