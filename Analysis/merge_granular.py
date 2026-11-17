"""Merge the per-task granular accuracy files of slurm/17_granular.sh into one csv for the accuracy plots.
Columns of the result  Dataset, Method, Seed, Event_Index, Accuracy  (the format the plot script already reads) plus Backbone.
The DistilGPT2 method gets the suffix _distilgpt2 (CONAP_FM_full_distilgpt2) so that it is a separate curve next to the TinyLLM one.
  python Analysis/merge_granular.py [--dir OutputLogs/granular] [--out OutputLogs/granular_accuracy.csv]"""
import argparse, glob, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="OutputLogs/granular")
ap.add_argument("--out", default="OutputLogs/granular_accuracy.csv")
a = ap.parse_args()

parts = []
for f in sorted(glob.glob(os.path.join(a.dir, "*.csv"))):
    d = pd.read_csv(f)
    d["Backbone"] = os.path.basename(f).split("__")[0]
    parts.append(d)
if not parts:
    raise SystemExit(f"no files in {a.dir}")
d = pd.concat(parts, ignore_index=True)
d.loc[d.Backbone == "distilgpt2", "Method"] = d.loc[d.Backbone == "distilgpt2", "Method"] + "_distilgpt2"
n0 = len(d)
d = d.drop_duplicates(["Dataset", "Method", "Seed", "Event_Index"], keep="last")      # a run that was killed and repeated can leave duplicates
d = d.sort_values(["Dataset", "Method", "Seed", "Event_Index"])
d.to_csv(a.out, index=False)
print(f"{len(parts)} files, {n0} rows, {len(d)} after removing {n0 - len(d)} duplicates -> {a.out}")
print(d.groupby(["Method"]).Dataset.nunique().rename("logs").to_string())
