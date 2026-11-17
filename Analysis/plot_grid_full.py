"""Figure 1 of the sensitivity analysis. Effect of W, tau and the cooldown C (epsilon fixed at 0.05) on the grid of grid_full.py.

  python -m Analysis.plot_grid_full --grid runs/grid_full/results --runs runs/all/results --out runs/grid_full/figures

One row per log, one column per parameter. In a panel the x axis holds the values of that parameter. Grey dots are the cells that share the value
(9 per dot column, one run each), the blue line is their mean. The dotted line per row is Forward (plain LoRA, no consolidation).
The lines come from runs/all, so run them on the same stream as the grid. Every row has its own y axis.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, DOT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b0"
TITLE = {"BPIC2015_Recurrent": "BPIC2015"}
PARAMS = [("window", "Window size W"), ("tau", "Variance threshold tau"), ("cooldown", "Cooldown C")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 150})


def tick_label(k, v):
    if k == "epsilon" and v >= 1:
        return "1 (off)"
    return f"{v:g}"


def load(grid_dir, runs_dir):
    rows = []
    for f in glob.glob(os.path.join(grid_dir, "*.json")):
        r = json.load(open(f))
        if str(r["tag"]).startswith("f_"):
            rows.append(dict(dataset=r["dataset"], window=r["window_size"], tau=r["tau"], epsilon=r["epsilon"], cooldown=r["cooldown"],
                             acc=r["overall_accuracy"], triggers=r["plateau_triggers"]))
    g = pd.DataFrame(rows)
    if g.empty:
        raise SystemExit(f"no grid results in {grid_dir}")
    ref = {}
    for f in glob.glob(os.path.join(runs_dir, "*.json")):
        r = json.load(open(f))
        ref.setdefault((r["dataset"], r["tag"]), []).append(r["overall_accuracy"])
    ref = {k: float(np.mean(v)) for k, v in ref.items()}
    return g, ref


def figure(g, ref, out):
    LOGS = [d for d in g.dataset.unique()]
    fig, axes = plt.subplots(len(LOGS), len(PARAMS), figsize=(7.6, 2.35 * len(LOGS) + 0.9), sharey="row", squeeze=False)
    rng = np.random.default_rng(0)
    for i, ds in enumerate(LOGS):
        h = g[g.dataset == ds]
        fwd = ref.get((ds, "forward"), np.nan)
        for j, (k, label) in enumerate(PARAMS):
            ax = axes[i, j]
            vals = sorted(h[k].unique())
            xs = np.arange(len(vals))
            for x, v in zip(xs, vals):
                a = h[h[k] == v]["acc"].values
                ax.scatter(x + rng.uniform(-0.17, 0.17, len(a)), a, s=9, color=DOT, edgecolor="none", zorder=2)
            m = [h[h[k] == v]["acc"].mean() for v in vals]
            ax.plot(xs, m, color=BLUE, lw=2, marker="o", ms=6, mfc=BLUE, mec="white", mew=1, zorder=4)
            if not np.isnan(fwd):
                ax.axhline(fwd, color=MUTED, lw=1.2, ls=(0, (1, 2)), zorder=3)
            ax.set_xticks(xs, [tick_label(k, v) for v in vals])
            ax.set_xlim(-0.5, len(vals) - 0.5)
            if i == len(LOGS) - 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel(f"{TITLE.get(ds, ds)}\nAccuracy")
    for i in range(len(LOGS)):
        lo, hi = axes[i, 0].get_ylim()
        axes[i, 0].set_ylim(lo, hi)
    handles = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=6, mfc=BLUE, mec="white", label="Mean over the other parameters"),
               plt.Line2D([], [], color=DOT, lw=0, marker="o", ms=4, label="Single cell (one run)"),
               plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (1, 2)), label="Forward (LoRA without consolidation)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"grid_full_effects.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="runs/grid_full/results")
    ap.add_argument("--runs", default="runs/all/results")
    ap.add_argument("--out", default="runs/grid_full/figures")
    a = ap.parse_args()
    g, ref = load(a.grid, a.runs)
    figure(g, ref, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
