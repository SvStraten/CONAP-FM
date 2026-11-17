import argparse
import os
import subprocess
import sys

try:
    from Setup.final import FINAL    
except ImportError:
    FINAL = {}
from Setup.hparams import HPARAMS, EPOCHS, SHARED

CONFIGS = {
    "full":            [],
    "forward":         ["--no-backward-transfer"],
    "neither":         ["--no-caga"],
    "with_mask":       ["--paca"],
    "no_caga":         ["--paca", "--no-caga"],
    "mask_legacy":     ["--paca", "--paca-mode", "legacy"],
    "caga_affinity":   ["--caga-mode", "affinity"],
    "order_old":       ["--paca", "--adapt-order", "paca_caga"],   
    "oracle":          ["--trigger", "oracle"],
    "fixed":           ["--trigger", "fixed"],
    "adwin":           ["--trigger", "adwin"],
    "page_hinkley":    ["--trigger", "page_hinkley"],
    # backbone versus drift awareness
    "full_ft":         ["--variant", "full_ft", "--lr", "2e-5"],
    "random_full":     ["--variant", "random_full"],
    "random_full_lowlr": ["--variant", "random_full", "--lr", "2e-5"],   
    "random_lora":     ["--variant", "random_lora"],
}

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
                    cmd += CONFIGS[cfg] + extra + args.extra  
                    print(" ".join(cmd))
                    if not args.dry_run:
                        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
