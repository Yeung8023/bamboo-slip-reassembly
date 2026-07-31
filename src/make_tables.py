"""Emit the manuscript tables as LaTeX, straight from the result CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import style

FMT = "{:.3f}"


def _m(d, met):
    return d.groupby(["state", "method"])[met].mean()


def table_main(runs, out):
    d = pd.read_csv(runs)
    keep = ["top1", "mutual", "matching", "morph", "length", "full", "latent",
            "expert_top10", "expert_top50"]
    lbl = dict(style.METHOD_LABELS)
    lbl.update({"expert_top10": "Expert review, top-10 (oracle)",
                "expert_top50": "Expert review, top-50 (oracle)"})
    mets = ["f1", "ari", "exact_slip", "cross_slip_rate"]
    g = d[d.method.isin(keep)].groupby(["state", "method"])[mets].agg(["mean", "std"])

    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{\textbf{Reconstruction quality across the preservation "
             r"ladder.} Corpora of 1{,}151 fragments; mean over five seeds "
             r"(standard deviation in parentheses). Cross-slip rate is the "
             r"fraction of accepted joins that link fragments of two different "
             r"slips.}", r"\label{tab:main}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"State & Method & Join $F_1$ & Slip ARI & Exact slips & "
             r"Cross-slip \\", r"\midrule"]
    for st in ["P1", "P2", "P3", "P4", "P5"]:
        for i, mth in enumerate(keep):
            if (st, mth) not in g.index:
                continue
            r = g.loc[(st, mth)]
            cells = []
            for met in mets:
                mu, sd = r[(met, "mean")], r[(met, "std")]
                c = FMT.format(mu)
                if np.isfinite(sd):
                    c += r" \tiny{(" + f"{sd:.3f}" + ")}"
                if mth == "latent":
                    c = r"\textbf{" + FMT.format(mu) + r"} \tiny{(" + f"{sd:.3f}" + ")}"
                cells.append(c)
            name = style.tex_label(lbl.get(mth, mth))
            lines.append(f"{st if i == 0 else ''} & {name} & " +
                         " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    Path(out).write_text("\n".join(lines))


def table_exchange(path, out):
    d = pd.read_csv(path)
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{\textbf{What the constraint set is worth, in units of "
             r"matcher accuracy.} For each state, the Top-1 accuracy that "
             r"bipartite matching would require in order to reach the slip "
             r"partition the proposed model attains at the accuracy it "
             r"actually had.}", r"\label{tab:exchange}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"State & Matcher Top-1 & Proposed ARI & Equivalent Top-1 \\",
             r"\midrule"]
    for _, r in d.sort_values("top1").iterrows():
        eq = (r"\emph{unreachable}" if r.unreachable
              else f"{r.equivalent_top1:.3f} " +
                   r"\tiny{(+" + f"{r.accuracy_points:.3f}" + ")}")
        lines.append(f"{r.state} & {r.top1:.3f} & {r.proposed:.3f} & {eq}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out).write_text("\n".join(lines))


def table_ablation(path, out, state="P4"):
    d = pd.read_csv(path)
    d = d[d.state == state]
    order = ["latent", "latent_no_notch", "latent_no_hand", "full",
             "no_length", "no_notch", "no_width", "no_strat", "no_hand"]
    lbl = {"latent": "Latent-slip (proposed)",
           "latent_no_notch": r"\quad $-$ binding notches",
           "latent_no_hand": r"\quad $-$ scribal hand",
           "full": "Pairwise constraints",
           "no_length": r"\quad $-$ slip length",
           "no_notch": r"\quad $-$ binding notches",
           "no_width": r"\quad $-$ morphometry",
           "no_strat": r"\quad $-$ stratigraphy",
           "no_hand": r"\quad $-$ scribal hand"}
    mets = ["f1", "ari", "exact_slip", "cross_slip_rate"]
    g = d.groupby("method")[mets].mean()
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{\textbf{Leave-one-out over the constraint set at " +
             state + r".} Each row removes one constraint from the formulation "
             r"named above it.}", r"\label{tab:ablation}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Formulation & Join $F_1$ & Slip ARI & Exact slips & "
             r"Cross-slip \\", r"\midrule"]
    for mth in order:
        if mth not in g.index:
            continue
        r = g.loc[mth]
        cells = [FMT.format(r[m]) for m in mets]
        nm = lbl[mth]
        if mth in ("latent", "full"):
            nm = r"\textbf{" + nm + "}"
        lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out).write_text("\n".join(lines))


def table_methods(out):
    rows = [
        ("Top-1 candidate", "heuristic", "take the best candidate for each fragment",
         "the naive use of a ranker"),
        ("Mutual best", "heuristic", "accept $i\\!\\to\\!j$ only if $j$ also prefers $i$",
         "standard practical heuristic"),
        ("Bipartite matching", "optimisation", "uniqueness only",
         "strongest method that does not model the slip"),
        ("+ morphometry, context", "optimisation", "adds width gate and excavation unit",
         "isolates the cheap gates"),
        ("+ slip length", "optimisation", "adds order, acyclicity and length",
         "isolates the chain structure"),
        ("Pairwise constraints", "optimisation", "all constraints, pairwise proxies",
         "our first formulation"),
        ("Latent-slip (proposed)", "optimisation",
         "all constraints, latent $\\Lambda,\\Omega,\\eta$", "the proposed model"),
        ("Expert review (oracle)", "reference", "finds the partner if it is in top $k$",
         "upper bound on ranked-list workflows"),
    ]
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Compared methods. All optimisation based methods are "
             r"constraint subsets of the same model, solved by the same solver "
             r"on the same arc set.}", r"\label{tab:methods}",
             r"\begin{tabular}{llll}", r"\toprule",
             r"Method & Type & What it does & Why it is included \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out).write_text("\n".join(lines))


def table_metrics(out):
    rows = [
        ("Join $F_1$", "harmonic mean of precision and recall over true joins",
         "are the individual joins right"),
        ("Slip ARI", "adjusted Rand index of fragments grouped into slips",
         "are the slips right"),
        ("Exact slips", "fraction of true multi-fragment slips recovered end to end",
         "is the output usable without checking"),
        ("Cross-slip rate", "fraction of accepted joins linking two different slips",
         "how often two reconstructions are corrupted"),
        ("Review pairs", "candidate pairs a specialist must inspect",
         "the human cost of a ranked-list workflow"),
        ("Solve time", "wall clock time of the optimisation",
         "practical cost of the method"),
    ]
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Evaluation metrics and what each one answers.}",
             r"\label{tab:metrics}", r"\begin{tabular}{lll}", r"\toprule",
             r"Metric & Definition & Question it answers \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out).write_text("\n".join(lines))


def table_hyper(out):
    rows = [
        ("Encoder embedding dimension", "$d$", "128"),
        ("Descriptor size", "", "$2\\times32\\times48$"),
        ("Training pairs", "", "179952"),
        ("Training epochs, batch size", "", "40, 512"),
        ("Learning rate, schedule", "", "$3\\times10^{-4}$, one cycle"),
        ("Candidates kept per break end", "$k$", "50"),
        ("Width gate", "$\\tau_\\Omega$", "0.45 mm"),
        ("Notch window half width", "$\\tau_\\nu$", "16 mm"),
        ("Unplaced notch penalty", "$\\rho$", "1.0"),
        ("Maximum slip length", "$\\Lambda_{\\max}$", "285 mm"),
        ("Hand attribution accuracy", "$p$", "0.80"),
        ("Number of scribal hands", "$K$", "6"),
        ("Join threshold", "$\\theta$", "tuned per method on calibration corpora"),
        ("Solver time budget", "", "180 s (900 s for the scaling study)"),
    ]
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Hyperparameters. Only $\theta$ is tuned, and it is "
             r"tuned on calibration corpora that are disjoint from every "
             r"evaluation corpus.}", r"\label{tab:hyper}",
             r"\begin{tabular}{lll}", r"\toprule",
             r"Quantity & Symbol & Value \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out).write_text("\n".join(lines))


if __name__ == "__main__":
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    table_main("results/runs_main.csv", "paper/tables/table_main.tex")
    table_exchange("results/table_exchange.csv", "paper/tables/table_exchange.tex")
    table_ablation("results/runs_ablation.csv", "paper/tables/table_ablation.tex")
    table_methods("paper/tables/table_methods.tex")
    table_metrics("paper/tables/table_metrics.tex")
    table_hyper("paper/tables/table_hyper.tex")
    print("wrote paper/tables/*.tex")
