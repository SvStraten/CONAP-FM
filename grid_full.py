import argparse
import itertools
import os
import subprocess
import sys

from Setup.hparams import EPOCHS, HPARAMS

GRID = dict(window=[50, 100, 500], tau=[5e-4, 5e-3, 5e-2], cooldown=[5, 10, 20])
LOGS = ["IRO5000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=LOGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[40])
    for k, v in GRID.items():
        ap.add_argument(f"--{k}", nargs="+", type=int if k in ("window", "cooldown") else float, default=v)
    ap.add_argument("--epsilon", type=float, default=0.05, help="CaGA threshold, fixed for the whole grid")
    ap.add_argument("--data-split", default="evaluation", choices=["full", "evaluation", "validation"])
    ap.add_argument("--lora-16-32", action="store_true", help="rank 16 and alpha 32 instead of the values of Table 1")
    ap.add_argument("--out", default="runs/grid_full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags for every run, put last")
    a = ap.parse_args()

    for ds in a.datasets:
        hp = HPARAMS[ds]
        r, alpha = (16, 32) if a.lora_16_32 else (hp["r"], hp["alpha"])
        for W, tau, C in itertools.product(a.window, a.tau, a.cooldown):
            tag = f"f_w{W}_tau{tau:g}_c{C}"
            for seed in a.seeds:
                if os.path.exists(os.path.join(a.out, "results", f"{ds}__{tag}__seed{seed}.json")):
                    continue
                cmd = [sys.executable, "-m", "Methods.CONAP_FM.run", "--dataset", f"Data/{ds}.csv", "--data-split", a.data_split,
                       "--seed", str(seed), "--tag", tag, "--window-size", str(W), "--lr", str(hp["lr"]),
                       "--loss-variance-threshold", str(tau), "--epsilon", str(a.epsilon), "--svd-cooldown-windows", str(C),
                       "--epochs-per-window", str(EPOCHS), "--lora-r", str(r), "--lora-alpha", str(alpha),
                       "--plots-dir", a.out, "--quiet"] + a.extra
                print(" ".join(cmd))
                if not a.dry_run:
                    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
