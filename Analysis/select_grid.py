"""Pick the best parameter combination per log from the validation grid and write Setup/final.py.

  python -m Analysis.select_grid --results runs/grid/results

For every log the combination with the highest mean validation accuracy over seeds is chosen. The report shows
  the best three combinations, the gain over the current setting of Setup/hparams.py (window, tau and cooldown),
  the mean accuracy per value of each parameter (averaged over the others), which shows how much each parameter matters,
  and how often the plateau trigger fired, since a log without consolidations says nothing about tau and the cooldown.
Files written  runs/grid/grid_table.csv (every combination), Setup/final.py (used by run_experiments.py).
"""
import argparse
import glob
import json
import os

import pandas as pd

KEYS = ["window", "tau", "cooldown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/grid/results")
    ap.add_argument("--out", default="Setup/final.py")
    a = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(a.results, "*.json")):
        r = json.load(open(f))
        if str(r["tag"]).startswith("g_"):
            rows.append(dict(dataset=r["dataset"], seed=r["seed"], window=r["window_size"], tau=r["tau"], cooldown=r["cooldown"],
                             acc=r["overall_accuracy"], triggers=r["plateau_triggers"]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no grid results in {a.results}")
    g = df.groupby(["dataset"] + KEYS).agg(mean=("acc", "mean"), std=("acc", "std"), seeds=("acc", "count"),
                                           triggers=("triggers", "mean")).reset_index()
    g.to_csv(os.path.join(os.path.dirname(a.results), "grid_table.csv"), index=False)

    from Setup.hparams import HPARAMS, SHARED
    tuned = {}
    for ds, h in g.groupby("dataset"):
        # ties (also differences below 1e-4) go to the default setting, so a log without consolidations keeps tau and C of Setup/hparams.py
        h = h.assign(_acc=h["mean"].round(4), _off=(h.tau != HPARAMS[ds]["tau"]).astype(int) + (h.cooldown != SHARED["cooldown_C"]).astype(int)
                     + (h.window != HPARAMS[ds]["W"]).astype(int))
        h = h.sort_values(["_acc", "_off", "window", "tau", "cooldown"], ascending=[False, True, True, True, True]).drop(columns=["_acc", "_off"])
        b = h.iloc[0]
        tuned[ds] = dict(window=int(b.window), tau=float(b.tau), cooldown=int(b.cooldown))
        d = h[(h.window == HPARAMS[ds]["W"]) & (h.tau == HPARAMS[ds]["tau"]) & (h.cooldown == SHARED["cooldown_C"])]
        dtxt = f"current setting {d['mean'].iloc[0]:.4f}, gain {b['mean'] - d['mean'].iloc[0]:+.4f}" if len(d) else "current setting not in the grid"
        print(f"\n{ds}  ({int(h.seeds.min())} seeds, {len(h)} combinations)  {dtxt}")
        for _, r in h.head(3).iterrows():
            print(f"   W {int(r.window):<5d} tau {r.tau:<7g} C {int(r.cooldown):<3d} acc {r['mean']:.4f} +- {0 if pd.isna(r['std']) else r['std']:.4f}  consolidations {r.triggers:.1f}")
        print("   marginal means  " + "   ".join(f"{k} " + " ".join(f"{v:g}:{m:.3f}" for v, m in h.groupby(k)["mean"].mean().items()) for k in KEYS))
        if h["triggers"].max() == 0:
            print("   WARNING no consolidation in any run, tau and the cooldown have no effect here")
    allm = g.groupby(["dataset"] + KEYS)["mean"].mean().reset_index()
    print("\nAcross all logs, mean accuracy per value")
    for k in KEYS:
        print("   " + k + "   " + "   ".join(f"{v:g}: {m:.4f}" for v, m in allm.groupby(k)["mean"].mean().items()))
    with open(a.out, "w") as f:
        f.write('"""Values of W, tau, CaGA threshold epsilon and cooldown C per log, read by run_experiments.py.\n\n'
                'Written by Analysis/select_grid.py from the grid on the 15% validation split. epsilon is fixed at the value of Setup/hparams.py."""\n\n'
                'FINAL = {\n')
        for k, v in sorted(tuned.items()):
            f.write(f'    "{k}": dict(window={v["window"]}, tau={v["tau"]:g}, epsilon={SHARED["epsilon"]}, cooldown={v["cooldown"]}),\n')
        f.write("}\n")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
