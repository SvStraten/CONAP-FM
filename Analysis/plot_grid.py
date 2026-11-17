"""Figures for the validation grid search (window size, tau, cooldown).

  python -m Analysis.plot_grid --results runs/grid/results --out runs/grid/figures

Writes PNG and PDF files
  grid_effects      how much each parameter matters. One thin grey line per log (accuracy relative to the mean of that log),
                    the blue line is the mean over logs with the spread across logs as error bars.
  grid_heatmap_<log>  window size by tau, one panel per cooldown, cells show the mean validation accuracy, the best cell has a frame.
  grid_cube_<log>   the same grid as 27 cubes, window by tau by cooldown, colour is the accuracy. Seen from two sides so that no cube stays hidden.
                    Every cube has a grey edge, the best one an orange edge. All cubes are opaque so the colour always matches the colour bar.
  grid_cube_all     all logs in one figure, two rows by five columns, for the paper. Colour scale per log, its range is in the panel title.
Accuracy is the mean over seeds. Heatmap cells with no consolidation during the run are hatched, tau and the cooldown cannot matter there.
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
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eaf2fb", "#9cc3ee", "#2a78d6", "#103f78"])
KEYS = ["window", "tau", "cooldown"]
LABEL = dict(window="Window size W (events)", tau="Variance threshold tau", cooldown="Cooldown C (windows)")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 150})


def load(results_dir):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        r = json.load(open(f))
        if str(r["tag"]).startswith("g_"):
            rows.append(dict(dataset=r["dataset"], seed=r["seed"], window=r["window_size"], tau=r["tau"], cooldown=r["cooldown"],
                             acc=r["overall_accuracy"], triggers=r["plateau_triggers"]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no grid results in {results_dir}")
    return df.groupby(["dataset"] + KEYS).agg(mean=("acc", "mean"), std=("acc", "std"), triggers=("triggers", "mean")).reset_index()


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fmt(k, v):
    return f"{v:g}"


def effects(g, out):
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), sharey=True)
    g = g.copy()
    g["rel"] = g["mean"] - g.groupby("dataset")["mean"].transform("mean")
    for ax, k in zip(axes, KEYS):
        per = g.groupby(["dataset", k])["rel"].mean().unstack(k)          # log x value
        xs = np.arange(per.shape[1])
        for _, row in per.iterrows():
            ax.plot(xs, row.values, color="#c3c2b7", lw=0.8, zorder=1)
        m, s = per.mean(), per.std()
        ax.errorbar(xs, m.values, yerr=s.values, color=BLUE, lw=2, marker="o", ms=5, capsize=3, zorder=3, mfc=BLUE, mec="white", mew=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(xs, [fmt(k, v) for v in per.columns])
        ax.set_xlabel(LABEL[k])
        ax.set_xlim(-0.3, len(xs) - 0.7)
    axes[0].set_ylabel("Accuracy relative to the log mean")
    fig.suptitle("Effect of each parameter (grey lines are single logs, blue is the mean over logs with spread)", fontsize=9, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, out, "grid_effects")


def heatmaps(g, out):
    eps = sorted(g.cooldown.unique())
    for ds, h in g.groupby("dataset"):
        wins, taus = sorted(h.window.unique()), sorted(h.tau.unique())
        lo, hi = h["mean"].min(), h["mean"].max()
        fig, axes = plt.subplots(1, len(eps), figsize=(3.0 * len(eps) + 0.6, 2.8), sharey=True)
        axes = np.atleast_1d(axes)
        best = h.loc[h["mean"].idxmax()]
        for ax, e in zip(axes, eps):
            ax.grid(False)
            for i, w in enumerate(wins):
                for j, t in enumerate(taus):
                    c = h[(h.window == w) & (h.tau == t) & (h.cooldown == e)]
                    if c.empty:
                        continue
                    v, trig = float(c["mean"].iloc[0]), float(c["triggers"].iloc[0])
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=SEQ((v - lo) / (hi - lo + 1e-12)), edgecolor="white", lw=2,
                                               hatch="////" if trig == 0 else None))
                    dark = (v - lo) / (hi - lo + 1e-12) > 0.55
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if dark else INK)
                    if (w, t, e) == (best.window, best.tau, best.cooldown):
                        ax.add_patch(plt.Rectangle((j - .43, i - .43), .86, .86, fill=False, edgecolor=ORANGE, lw=2.5, zorder=5))
            ax.set_xlim(-.5, len(taus) - .5)
            ax.set_ylim(len(wins) - .5, -.5)
            ax.set_xticks(range(len(taus)), [fmt("tau", t) for t in taus])
            ax.set_yticks(range(len(wins)), [fmt("window", w) for w in wins])
            ax.set_xlabel("tau")
            ax.set_title(f"cooldown C {fmt('cooldown', e)}", fontsize=9, color=INK)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_ylabel("Window size W")
        fig.suptitle(f"{ds}, validation accuracy. Orange frame is the best cell, hatched cells had no consolidation.", fontsize=9, color=INK, y=1.04)
        fig.tight_layout()
        save(fig, out, f"grid_heatmap_{ds}")


GREY_EDGE = "#6b6a66"
STEP = 3          # grid cells between two cubes, 3 leaves a gap of two cube widths
# Viewing angle (elevation 27, azimuth -76). On screen every pair of cubes is at least 1.45 cube widths apart, so no cube hides another
# one. This was found by a search over angles, the usual angles such as azimuth -58 line up cubes exactly behind each other.
# the ten logs in the order of the paper, the five synthetic ones on the top row
ORDER = ["IRO5000", "ORI5000", "ROI5000", "OIR5000", "RIO5000", "BPIC2015_Recurrent", "RequestForPayment",
         "InternationalDeclarations", "DomesticDeclarations", "BPIC2017"]


def _cube_data(h):
    """Voxel arrays for the 27 cells of one log. Colours are opaque, every cube has a grey edge, the best one an orange edge."""
    wins, taus, eps = sorted(h.window.unique()), sorted(h.tau.unique()), sorted(h.cooldown.unique())
    n = (len(wins), len(taus), len(eps))
    norm = matplotlib.colors.Normalize(h["mean"].min(), h["mean"].max() + 1e-12)
    best = h.loc[h["mean"].idxmax()]
    shape = tuple(STEP * k - (STEP - 1) for k in n)          # cubes every STEP cells, gaps in between
    filled = np.zeros(shape, dtype=bool)
    colors, edges = np.zeros(shape + (4,)), np.zeros(shape + (4,))
    for i, w in enumerate(wins):
        for j, t in enumerate(taus):
            for k, e in enumerate(eps):
                c = h[(h.window == w) & (h.tau == t) & (h.cooldown == e)]
                if c.empty:
                    continue
                idx = (STEP * i, STEP * j, STEP * k)
                filled[idx] = True
                colors[idx] = SEQ(norm(float(c["mean"].iloc[0])))
                is_best = (w, t, e) == (best.window, best.tau, best.cooldown)
                edges[idx] = (*matplotlib.colors.to_rgb(ORANGE if is_best else GREY_EDGE), 1)
    return dict(wins=wins, taus=taus, eps=eps, n=n, norm=norm, best=best, filled=filled, colors=colors, edges=edges)


def _draw_cube(ax, d, view, fs=8, lw=0.8, labelpad=(6, 6, 2), short=False):
    ax.voxels(d["filled"], facecolors=d["colors"], edgecolors=d["edges"], linewidth=lw, shade=False)   # no shading, colour carries the value
    n = d["n"]
    ax.set_xticks([STEP * i + .5 for i in range(n[0])], [f"{w:g}" for w in d["wins"]], fontsize=fs)
    ax.set_yticks([STEP * j + .5 for j in range(n[1])], [f"{t:g}" for t in d["taus"]], fontsize=fs)
    ax.set_zticks([STEP * k + .5 for k in range(n[2])], [f"{e:g}" for e in d["eps"]], fontsize=fs)
    ax.set_xlabel("W", labelpad=labelpad[0], fontsize=fs + 1) if short else ax.set_xlabel("Window size W", labelpad=labelpad[0], fontsize=fs + 1)
    ax.set_ylabel("tau", labelpad=labelpad[1], fontsize=fs + 1)
    ax.set_zlabel("C" if short else "Cooldown C (windows)", labelpad=labelpad[2], fontsize=fs + 1)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = GRID


def cube(g, out):
    """One figure per log, the cube seen from two sides."""
    for ds, h in g.groupby("dataset"):
        d = _cube_data(h)
        fig = plt.figure(figsize=(10.5, 4.8))
        for pos, view in enumerate([(27, -76), (27, 104)], start=1):
            _draw_cube(fig.add_subplot(1, 2, pos, projection="3d"), d, view, fs=9, lw=0.8, labelpad=(8, 8, 2))
        fig.subplots_adjust(left=0.0, right=0.9, wspace=0.18, top=0.9, bottom=0.02)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=d["norm"], cmap=SEQ), cax=fig.add_axes([0.93, 0.25, 0.015, 0.5]))
        cb.set_label("Mean validation accuracy")
        cb.outline.set_visible(False)
        b = d["best"]
        fig.suptitle(f"{ds}. Orange edge is the best cell (W {int(b.window)}, tau {b.tau:g}, C {int(b.cooldown)}, {b['mean']:.3f}).",
                     fontsize=9, color=INK)
        save(fig, out, f"grid_cube_{ds}")


def cube_all(g, out, ncols=5):
    """All logs in one figure for the paper, two rows by five columns, one view per log.
    The colour scale is set per log (its lowest to its highest accuracy) because the logs differ a lot in level.
    The range of each log is printed in its title."""
    logs = [d for d in ORDER if d in set(g.dataset)] + sorted(set(g.dataset) - set(ORDER))
    nrows = int(np.ceil(len(logs) / ncols))
    fig = plt.figure(figsize=(3.7 * ncols, 3.9 * nrows + 0.6))
    for k, ds in enumerate(logs):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        d = _cube_data(g[g.dataset == ds])
        _draw_cube(ax, d, (27, -76), fs=6.5, lw=0.6, labelpad=(-2, 9, 0), short=True)
        lo, hi = d["norm"].vmin, d["norm"].vmax
        ax.set_title(f"{ds.replace('_Recurrent', '')}\n{lo:.3f} to {hi:.3f}", fontsize=9, color=INK, pad=4)
    fig.subplots_adjust(left=0.01, right=0.925, wspace=0.16, hspace=0.2, top=0.9, bottom=0.03)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=SEQ),
                      cax=fig.add_axes([0.95, 0.3, 0.008, 0.4]), ticks=[0, 1])
    cb.ax.set_yticklabels(["lowest\nin log", "highest\nin log"], fontsize=7.5)
    cb.set_label("Validation accuracy (scale set per log)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Grid search on the validation split. Axes are window size W, tau and cooldown C. Orange edge marks the best cell.",
                 fontsize=9, color=INK, y=0.985)
    save(fig, out, "grid_cube_all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/grid/results")
    ap.add_argument("--out", default="runs/grid/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = load(a.results)
    effects(g, a.out)
    heatmaps(g, a.out)
    cube(g, a.out)
    cube_all(g, a.out)
    print("wrote", len(os.listdir(a.out)), "files to", a.out)


if __name__ == "__main__":
    main()
