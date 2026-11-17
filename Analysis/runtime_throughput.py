"""Throughput in events per second for every method and log, as a LaTeX table in the style of the accuracy table.

  python -m Analysis.runtime_throughput --tinyllm ablation_results_tinyllm.csv --distilgpt2 ablation_results_distilgpt2.csv --out Results

Events per second = number of events in the evaluation stream (85% of |S|, Table 1) divided by the end-to-end runtime of a run.
Mean and std are taken over the seeds of the per-run values. Needs \\res{mean}{std} from the paper preamble.
"""
import argparse
import os

import pandas as pd

from Analysis.runtime_table import load

EVAL_FRAC = 0.85
LOGS = [("IRO5000", "IRO5000", 53647), ("ORI5000", "ORI5000", 54618), ("ROI5000", "ROI5000", 51995), ("OIR5000", "OIR5000", 55334),
        ("RIO5000", "RIO5000", 53426), ("BPIC2015_Recurrent", "BPI15-REC", 143118), ("RequestForPayment", "BPI20-RFP", 36796),
        ("InternationalDeclarations", "BPI20-ID", 72151), ("DomesticDeclarations", "BPI20-DD", 56437), ("BPIC2017", "BPI17", 200033)]
ROWS = [("DoNothing~\\cite{rizzi2022update}", "tiny", "donothing"), ("LastDrift~\\cite{EDBN}", "tiny", "lastdrift"),
        ("DynaTrainCDD~\\cite{DynaTrainCDD}", "tiny", "dynatrain"), ("TFCLPM~\\cite{TFCLPM}", "tiny", "tfclpm"),
        ("CNAPwP~\\cite{CNAPwP}", "tiny", "cnapwp"), None,
        ("\\textbf{\\textit{CONAP-FM}}{\\tiny~TinyLLM}", "tiny", "full"), ("\\textbf{\\textit{CONAP-FM}}{\\tiny~DistilGPT2}", "gpt2", "full")]


def fmt(x):
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinyllm", required=True)
    ap.add_argument("--distilgpt2", required=True)
    ap.add_argument("--out", default="Results")
    a = ap.parse_args()
    src = {"tiny": load(a.tinyllm), "gpt2": load(a.distilgpt2)}
    stats, rec = {}, []
    for r in ROWS:
        if r is None:
            continue
        name, key, tag = r
        for ds, short, n in LOGS:
            x = src[key]
            x = x[(x.dataset == ds) & (x.tag == tag)]
            if x.empty:
                continue   # method not in the csv (competitor runs are not part of this repository)
            ms = EVAL_FRAC * n / (x.runtime_hours * 3600)
            stats[(name, short)] = (ms.mean(), ms.std())
            rec.append(dict(Method=name, Dataset=short, Mean_eps=ms.mean(), Std_eps=ms.std(), Median_eps=ms.median(), Seeds=len(ms)))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rec).round(3).to_csv(os.path.join(a.out, "runtime_throughput.csv"), index=False)

    shorts = [s for _, s, _ in LOGS]
    rows = [r for r in ROWS if r is None or (r[0], shorts[0]) in stats]   # competitors only if their runs are in the csv
    best = {s: max((stats[(r[0], s)][0], r[0]) for r in rows if r)[1] for s in shorts}
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Processing speed in events per second (mean $\\pm$ standard deviation over the seeds), obtained by dividing the number of events in the evaluation stream by the end-to-end runtime. \\textbf{Bold} denotes the highest throughput per dataset. CONAP-FM is shown with the TinyLLM and the DistilGPT2 backbone.}",
         "\\vspace{-0.8em}", "\\label{tab:runtime_throughput}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.3pt}", "\\renewcommand{\\arraystretch}{1.05}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccccccc}", "\\toprule",
         " & \\multicolumn{6}{c}{\\textit{Synthetic}} & \\multicolumn{4}{c}{\\textit{Real-world}} \\\\",
         "\\cmidrule(lr){2-7} \\cmidrule(lr){8-11}", "Method & " + " & ".join(shorts) + " \\\\", "\\midrule"]
    for r in rows:
        if r is None:
            L.append("\\midrule")
            continue
        cells = []
        for s in shorts:
            m, sd = stats[(r[0], s)]
            m_s = f"\\textbf{{{fmt(m)}}}" if best[s] == r[0] else fmt(m)
            cells.append(f"\\res{{{m_s}}}{{{fmt(sd)}}}")
        L.append(r[0] + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}"]
    open(os.path.join(a.out, "runtime_throughput.tex"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
