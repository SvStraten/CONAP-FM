"""Run all configurations x datasets x seeds. Finished runs are skipped, so it can be restarted.

  python run_experiments.py --datasets IRO5000 RequestForPayment --configs full --seeds 40 41 42 43 44 45
  python run_experiments.py --configs all_ablation --dry-run

Pretraining pairs are given as DATASET=checkpoint_dir, for example
  --pretrained RequestForPayment=checkpoints/PTC_seed40
and are run as extra configurations with the suffix _pre.
"""
import argparse
import os
import subprocess
import sys

try:
    from Setup.final import FINAL     # W, tau, epsilon and cooldown C per log, written by Analysis/select_grid.py
except ImportError:
    FINAL = {}
from Setup.hparams import HPARAMS, EPOCHS, SHARED

CONFIGS = {
    # main method and the ablation of its components. The importance mask is NOT part of the final method.
    #   full        trigger + memory + CaGA (the method)
    #   forward     plain LoRA, no memory (sequential fine-tuning without consolidation)
    #   neither     trigger and memory, but the memory is never used (no CaGA), so it equals forward
    #   with_mask   the method plus the parameter importance mask
    #   no_caga     the importance mask without CaGA (mask only)
    "full":            [],
    "forward":         ["--no-backward-transfer"],
    "neither":         ["--no-caga"],
    "with_mask":       ["--paca"],
    "no_caga":         ["--paca", "--no-caga"],
    "mask_legacy":     ["--paca", "--paca-mode", "legacy"],
    "caga_affinity":   ["--caga-mode", "affinity"],
    "order_old":       ["--paca", "--adapt-order", "paca_caga"],   # old order, mask before projection (diverges, kept to show why)
    # who triggers consolidation
    "oracle":          ["--trigger", "oracle"],
    "fixed":           ["--trigger", "fixed"],
    "adwin":           ["--trigger", "adwin"],
    "page_hinkley":    ["--trigger", "page_hinkley"],
    # backbone versus drift awareness
    "full_ft":         ["--variant", "full_ft", "--lr", "2e-5"],
    "random_full":     ["--variant", "random_full"],
    "random_full_lowlr": ["--variant", "random_full", "--lr", "2e-5"],   # same lr as full_ft, so the pretraining comparison is fair
    "random_lora":     ["--variant", "random_lora"],
}
# Every configuration uses the same budget per window, EPOCHS in Setup/hparams.py (10).
GROUPS = {
    "all_ablation": ["full", "forward", "neither", "with_mask", "no_caga"],
    "triggers": ["full", "oracle", "fixed", "adwin", "page_hinkley"],
    "backbone": ["full", "forward", "full_ft", "random_full", "random_full_lowlr", "random_lora"],
    "all": list(CONFIGS.keys()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(HPARAMS.keys()))
    ap.add_argument("--configs", nargs="+", default=["full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(40, 50)))
    ap.add_argument("--pretrained", nargs="*", default=[])
    ap.add_argument("--out", default="runs/all", help="result folder, use runs/distilgpt2 for the second backbone")
    ap.add_argument("--model-name", default="arnir0/Tiny-LLM", help="LLM backbone, for example distilbert/distilgpt2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags passed to every run, put last")
    args = ap.parse_args()

    configs = []
    for c in args.configs:
        configs += GROUPS.get(c, [c])
    pre = dict(x.split("=") for x in args.pretrained)

    for ds in args.datasets:
        hp = HPARAMS[ds]
        for cfg in configs:
            variants = [(cfg, [])]
            if ds in pre:
                ck = pre[ds]
                variants.append((cfg + "_pre", ["--pretrained-lora-path", f"{ck}/offline_lora.pt",
                                                "--pretrained-subspace-path", f"{ck}/offline_subspace.pt"]))
            for tag, extra in variants:
                for seed in args.seeds:
                    res = os.path.join(args.out, "results", f"{ds}__{tag}__seed{seed}.json")
                    if os.path.exists(res):
                        continue
                    tu = dict(window=hp["W"], tau=hp["tau"], epsilon=0.05, cooldown=SHARED["cooldown_C"]) | FINAL.get(ds, {})
                    cmd = [sys.executable, "-m", "Methods.CONAP_FM.run", "--dataset", f"Data/{ds}.csv",
                           "--data-split", "evaluation", "--seed", str(seed), "--tag", tag, "--model-name", args.model_name,
                           "--window-size", str(tu["window"]), "--lr", str(hp["lr"]),
                           "--loss-variance-threshold", str(tu["tau"]), "--epsilon", str(tu["epsilon"]), "--svd-cooldown-windows", str(tu["cooldown"]),
                           "--epochs-per-window", str(EPOCHS), "--lora-r", str(hp["r"]),
                           "--lora-alpha", str(hp["alpha"]), "--plots-dir", args.out, "--quiet"]
                    cmd += CONFIGS[cfg] + extra + args.extra   # config flags may override --lr
                    print(" ".join(cmd))
                    if not args.dry_run:
                        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
