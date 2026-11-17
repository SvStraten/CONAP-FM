"""Statistics and drift analysis over the JSON files written by Methods/CONAP_FM/run.py.

  python -m Analysis.analyze --results runs/CONAP_FM/results --reference full --out runs/analysis

Outputs (CSV files and one text report)
  accuracy_table.csv       mean and std of accuracy per dataset and configuration
  wilcoxon.csv             paired Wilcoxon signed-rank test of every configuration against the
                           reference per dataset, Holm corrected over all tests (ablation)
  wilcoxon_main.csv        main table. Reference against the strongest baseline per log only,
                           Holm corrected over the logs (use this for the significance marks)
  friedman_nemenyi.csv     Friedman test over datasets and Nemenyi p-values (needs scikit-posthocs)
  mean_ranks.csv           mean rank of each configuration, for the critical difference diagram
  drift_recovery.csv       accuracy drop and recovery time after each known drift
  early_stream.csv         accuracy in the first K windows (cold start)
  efficiency.csv           latency, update time, memory, trainable parameters
With n paired seeds the smallest two-sided Wilcoxon p-value is 2 / 2^n, so n >= 6 is needed for p < .05.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def load(results_dir):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def accuracy_table(df):
    t = df.groupby(["dataset", "tag"])["overall_accuracy"].agg(["mean", "std", "count"]).reset_index()
    return t


def wilcoxon_table(df, reference):
    out = []
    for ds, g in df.groupby("dataset"):
        ref = g[g.tag == reference].set_index("seed")["overall_accuracy"]
        for tag, h in g.groupby("tag"):
            if tag == reference:
                continue
            other = h.set_index("seed")["overall_accuracy"]
            common = ref.index.intersection(other.index)
            if len(common) < 2:
                continue
            a, b = ref.loc[common].values, other.loc[common].values
            diff = a - b
            p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(a, b, zero_method="wilcox").pvalue
            out.append(dict(dataset=ds, reference=reference, other=tag, n_seeds=len(common),
                            mean_diff=float(diff.mean()), p=p, min_possible_p=2.0 / 2 ** len(common)))
    t = pd.DataFrame(out)
    if len(t):
        t["p_holm"] = holm(t["p"].values)
        t["significant_05"] = t["p_holm"] < 0.05
    return t


def strongest_baseline_table(df, reference, baselines):
    """Main table test. Per log, CONAP-FM (reference) is compared with the one baseline that has the highest mean
    accuracy on that log, using a paired Wilcoxon signed-rank test over the seeds. Holm correction over the logs only,
    so with 10 seeds (smallest p 2 / 2^10 = .002) and 10 logs the threshold of the best p is .005 and can be reached."""
    out = []
    for ds, g in df.groupby("dataset"):
        ref = g[g.tag == reference].set_index("seed")["overall_accuracy"]
        best = None
        for tag in baselines:
            h = g[g.tag == tag]
            common = ref.index.intersection(h.seed)
            if len(common) < 2:
                continue
            m = h.set_index("seed").loc[common, "overall_accuracy"].mean()
            if best is None or m > best[1]:
                best = (tag, m, common)
        if best is None:
            continue
        tag, m, common = best
        a = ref.loc[common].values
        b = g[g.tag == tag].set_index("seed").loc[common, "overall_accuracy"].values
        diff = a - b
        p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(a, b, zero_method="wilcox").pvalue
        out.append(dict(dataset=ds, reference=reference, strongest_baseline=tag, n_seeds=len(common),
                        mean_reference=float(a.mean()), mean_baseline=float(m), mean_diff=float(diff.mean()),
                        p=p, min_possible_p=2.0 / 2 ** len(common)))
    t = pd.DataFrame(out)
    if len(t):
        t["p_holm"] = holm(t["p"].values)
        t["significant_05"] = t["p_holm"] < 0.05
    return t


def friedman_nemenyi(df, outdir):
    piv = df.groupby(["dataset", "tag"])["overall_accuracy"].mean().unstack("tag").dropna()
    if piv.shape[0] < 3 or piv.shape[1] < 3:
        return None, None
    stat, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    ranks.to_csv(os.path.join(outdir, "mean_ranks.csv"), header=["mean_rank"])
    nem = None
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = nem.columns = piv.columns
        nem.to_csv(os.path.join(outdir, "friedman_nemenyi.csv"))
    except ImportError:
        print("pip install scikit-posthocs for the Nemenyi test")
    return (stat, p, piv.shape[0]), ranks


def drift_recovery(rows, pre_k=5, post_k=5, tol=0.95):
    """For every known drift. pre = mean accuracy of the pre_k windows before the drift window.
    drop = pre minus lowest accuracy in the next post_k windows.
    recovery = windows until accuracy is back to tol * pre (nan if it never is within 30 windows).
    first_k = mean accuracy in the first post_k windows after the drift (proxy for forgetting on recurrence)."""
    out = []
    for r in rows:
        W, acc = r["window_size"], np.array(r["window_accuracies"])
        starts = np.array(r["window_starts"]) if r.get("window_starts") else np.arange(len(acc)) * W
        for d, dp in enumerate(r.get("drift_indices", [])):
            w = int(np.searchsorted(starts, dp - 1, side="right") - 1)
            if w - pre_k < 0 or w + post_k >= len(acc):
                continue
            pre = acc[w - pre_k:w].mean()
            post = acc[w:w + post_k]
            rec = np.nan
            for j in range(0, 30):
                if w + j >= len(acc):
                    break
                if acc[w + j] >= tol * pre:
                    rec = j
                    break
            out.append(dict(dataset=r["dataset"], tag=r["tag"], seed=r["seed"], drift=d, pre=pre,
                            drop=pre - post.min(), recovery_windows=rec, first_k=post.mean()))
    return pd.DataFrame(out)


def early_stream(rows, ks=(5, 10, 20)):
    out = []
    for r in rows:
        acc = np.array(r["window_accuracies"])
        rec = dict(dataset=r["dataset"], tag=r["tag"], seed=r["seed"])
        for k in ks:
            rec[f"acc_first_{k}_windows"] = float(acc[:k].mean())
        out.append(rec)
    return pd.DataFrame(out)


def efficiency(rows):
    keep = ["single_event_latency_mean", "single_event_latency_p95", "predict_seconds_per_event_batched_mean",
            "predict_seconds_per_event_batched_p95", "update_seconds_per_window_mean",
            "update_seconds_per_window_p95", "peak_memory_mb", "trainable_parameters", "total_parameters"]
    df = pd.DataFrame([{**{k: r[k] for k in keep}, "dataset": r["dataset"], "tag": r["tag"]} for r in rows])
    return df.groupby("tag")[keep].mean().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/CONAP_FM/results")
    ap.add_argument("--reference", default="full")
    ap.add_argument("--out", default="runs/analysis")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="tags of competitor results you added to the folder. For each log the strongest one is tested against the reference and written to wilcoxon_main.csv")
    ap.add_argument("--tags", nargs="*", default=None, help="only use these configurations")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = load(a.results)
    if a.tags:
        rows = [r for r in rows if r["tag"] in a.tags]
    if not rows:
        raise SystemExit(f"No result files found in {a.results} for the requested tags. Run the experiments first.")
    df = pd.DataFrame([{k: r[k] for k in ("dataset", "tag", "seed", "overall_accuracy", "macro_f1")} for r in rows])

    accuracy_table(df).to_csv(os.path.join(a.out, "accuracy_table.csv"), index=False)
    wilcoxon_table(df, a.reference).to_csv(os.path.join(a.out, "wilcoxon.csv"), index=False)
    if a.baselines:
        strongest_baseline_table(df, a.reference, a.baselines).to_csv(os.path.join(a.out, "wilcoxon_main.csv"), index=False)
    fr, ranks = friedman_nemenyi(df, a.out)
    dr = drift_recovery(rows)
    if len(dr):
        dr.drop(columns=["seed"]).groupby(["dataset", "tag"]).mean(numeric_only=True).to_csv(os.path.join(a.out, "drift_recovery.csv"))
    else:
        print("No drift has enough windows before and after it, drift_recovery.csv not written.")
    early_stream(rows).drop(columns=["seed"]).groupby(["dataset", "tag"]).mean(numeric_only=True).to_csv(os.path.join(a.out, "early_stream.csv"))
    efficiency(rows).to_csv(os.path.join(a.out, "efficiency.csv"), index=False)

    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(f"Runs {len(rows)}, datasets {df.dataset.nunique()}, configurations {sorted(df.tag.unique())}\n")
        if fr:
            f.write(f"Friedman chi2 {fr[0]:.3f}, p {fr[1]:.4g}, over {fr[2]} datasets\n")
            f.write("Mean ranks\n" + ranks.to_string() + "\n")
    print(open(os.path.join(a.out, "report.txt")).read())


if __name__ == "__main__":
    main()
