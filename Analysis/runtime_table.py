"""Full runtime table (end-to-end runtime of the whole stream, training + inference, in minutes).

  python -m Analysis.runtime_table --tinyllm ablation_results_tinyllm.csv --distilgpt2 ablation_results_distilgpt2.csv --out Results

Rows of competitor methods are only written if their runs are in the csv files, CONAP-FM rows always.
Writes  runtime_results.csv  (long format, one row per log and method)  and  runtime_results.md  (readable table, mean ± std).
"""
import argparse
import io
import os
import re

import pandas as pd

LOGS = [("IRO5000", "IRO5000"), ("ORI5000", "ORI5000"), ("ROI5000", "ROI5000"), ("OIR5000", "OIR5000"), ("RIO5000", "RIO5000"),
        ("BPIC2015_Recurrent", "BPI15-REC"), ("RequestForPayment", "BPI20-RFP"), ("InternationalDeclarations", "BPI20-ID"),
        ("DomesticDeclarations", "BPI20-DD"), ("BPIC2017", "BPI17")]
METHODS = [("CONAP-FM TinyLLM", "tiny", "full"), ("CONAP-FM DistilGPT2", "gpt2", "full"), ("TFCLPM", "tiny", "tfclpm"),
           ("CNAPwP", "tiny", "cnapwp"), ("DynaTrainCDD", "tiny", "dynatrain"), ("LastDrift", "tiny", "lastdrift")]


def load(path):
    t = open(path).read()
    t = re.sub(r"(?<=\d)(?=20\d\d-\d\d-\d\d \d\d:\d\d:\d\d,)", "\n", t)          # repair glued rows
    return pd.read_csv(io.StringIO(t)).drop_duplicates(["dataset", "tag", "seed"], keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinyllm", required=True)
    ap.add_argument("--distilgpt2", required=True)
    ap.add_argument("--out", default="Results")
    a = ap.parse_args()
    src = {"tiny": load(a.tinyllm), "gpt2": load(a.distilgpt2)}
    rows = []
    for ds, short in LOGS:
        for name, key, tag in METHODS:
            x = src[key]
            x = x[(x.dataset == ds) & (x.tag == tag)]
            if x.empty:
                continue   # method not in the csv (competitor runs are not part of this repository)
            m = x.runtime_hours * 60
            rows.append(dict(Dataset=short, Method=name, Mean_min=m.mean(), Std_min=m.std(), Median_min=m.median(),
                             Min_min=m.min(), Max_min=m.max(), Seeds=len(m)))
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    t.round(2).to_csv(os.path.join(a.out, "runtime_results.csv"), index=False)

    names = [m[0] for m in METHODS]
    lines = ["# Full runtime results", "",
             "End-to-end runtime of the whole evaluation stream (training + inference) in minutes, mean ± standard deviation over the seeds. "
             "The number of seeds is given in brackets. The median is in `runtime_results.csv` and is the better summary where a single run was slowed down by the cluster.", "",
             "| Log | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for _, short in LOGS:
        cells = []
        for n in names:
            r = t[(t.Dataset == short) & (t.Method == n)].iloc[0]
            cells.append(f"{r.Mean_min:.2f} ± {0 if pd.isna(r.Std_min) else r.Std_min:.2f} ({int(r.Seeds)})")
        lines.append(f"| {short} | " + " | ".join(cells) + " |")
    open(os.path.join(a.out, "runtime_results.md"), "w").write("\n".join(lines) + "\n")
    # LaTeX version (booktabs), fastest method per log in bold
    tex = ["% Requires \\usepackage{booktabs}", "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
           "\\caption{End-to-end runtime (minutes, mean $\\pm$ std over 10 seeds) of every method on every log. The fastest method per log is in bold.}",
           "\\label{tab:runtime_full}",
           "\\begin{tabular}{l" + "r" * len(names) + "}", "\\toprule",
           "Log & " + " & ".join(n.replace("CONAP-FM ", "CONAP-FM ") for n in names) + " \\\\", "\\midrule"]
    for _, short in LOGS:
        sub = t[t.Dataset == short].set_index("Method").loc[names]
        best = sub.Mean_min.idxmin()
        cells = []
        for n in names:
            c = f"{sub.loc[n, 'Mean_min']:.2f} $\\pm$ {sub.loc[n, 'Std_min']:.2f}"
            cells.append("\\textbf{" + c + "}" if n == best else c)
        tex.append(short.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(a.out, "runtime_results.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
