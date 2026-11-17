"""Grid over window size, tau and cooldown (epsilon fixed at 0.05, plain orthogonal CaGA rule) on the evaluation stream, one run (one seed) per cell.

  python grid_full.py --dry-run | wc -l                      27 commands (1 log x 27 cells)
  python grid_full.py --datasets IRO5000 --window 50      one slice

Parameters and values (27 cells per log)
  window   window size W in events                     50     100    500
  tau      variance threshold of the plateau trigger    5e-4   5e-3   5e-2
  epsilon  CaGA threshold on rho, fixed                 0.05 (--epsilon changes it for the whole grid)
  cooldown windows between two consolidations C         5      10     20
Log  IRO5000 (other logs with --datasets).
The stream is the 85% evaluation stream of the main table (--data-split evaluation, --data-split full takes the whole log), the seed is 40.
Everything else follows Table 1 and Setup/hparams.py: the learning rate eta of the log and 10 Adam steps per window.
LoRA rank and alpha are those of Table 1 (Setup/hparams.py), BPIC2015_Recurrent 256/512, RequestForPayment 128/256, DomesticDeclarations 256/512.
--lora-16-32 uses rank 16 and alpha 32 instead.
Results go to runs/grid_full/results as <log>__<tag>__seed<seed>.json with tags like f_w50_tau0.0005_c5. Finished runs are skipped.
Because the grid uses the evaluation data too, it is a sensitivity analysis. Do not pick the final setting of the paper from it without saying so.
"""
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
