"""Quick check that everything runs. Uses the first 3000 events of one log, one seed, small settings.

  python smoke_test.py                              # RequestForPayment, Tiny-LLM from the Hugging Face hub
  python smoke_test.py --dataset BPIC2015_Recurrent --events 5000
  python smoke_test.py --model-name /path/to/local/model --methods full forward

Prints one line per method with accuracy and time, and PASS or FAIL. Results go to runs/smoke.
Accuracies of such a short run mean nothing, the test only checks that the code runs end to end.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
ALL = ["full", "forward", "oracle", "random_full"]


def cmd(m, a, out):
    base = ["--dataset", f"Data/{a.dataset}.csv", "--seed", "40", "--tag", f"smoke_{m}", "--plots-dir", out,
            "--max-events", str(a.events), "--device", a.device]
    llm = ["--model-name", a.model_name]
    extra = {"full": [], "forward": ["--no-backward-transfer"], "oracle": ["--trigger", "oracle"],
             "random_full": ["--variant", "random_full"]}[m]
    return [PY, "-m", "Methods.CONAP_FM.run", "--quiet", "--window-size", "100", "--epochs-per-window", "2",
            "--lora-r", "16", "--lora-alpha", "32"] + llm + base + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="RequestForPayment")
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--model-name", default="arnir0/Tiny-LLM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--methods", nargs="+", default=ALL)
    ap.add_argument("--out", default="runs/smoke")
    a = ap.parse_args()
    if a.device == "auto":
        try:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            a.device = "cpu"

    bad = 0
    print(f"{'method':14s} {'status':6s} {'accuracy':>9s} {'seconds':>8s}")
    for m in a.methods:
        t0 = time.time()
        p = subprocess.run(cmd(m, a, a.out), capture_output=True, text=True)
        f = os.path.join(a.out, "results", f"{a.dataset}__smoke_{m}__seed40.json")
        if p.returncode == 0 and os.path.exists(f):
            r = json.load(open(f))
            print(f"{m:14s} {'PASS':6s} {r['overall_accuracy']:9.4f} {time.time() - t0:8.1f}")
        else:
            bad += 1
            print(f"{m:14s} {'FAIL':6s} {'':9s} {time.time() - t0:8.1f}")
            print("   " + "\n   ".join((p.stderr or p.stdout).strip().splitlines()[-6:]))
    if bad == 0:
        r = subprocess.run([PY, "-m", "Analysis.analyze", "--results", os.path.join(a.out, "results"),
                            "--reference", "smoke_full", "--out", os.path.join(a.out, "analysis")],
                           capture_output=True, text=True)
        print("analysis", "PASS" if r.returncode == 0 else "FAIL\n" + r.stderr[-400:])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
