"""LaTeX tables for the revision, generated from the result files.

Written to revision/paper/tables/. Each builder skips itself if its result file
is missing, so tables can be produced as experiments finish.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import common

OUT = common.ROOT / "revision" / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def _ms(d, val, dec=3):
    """mean (sd) over seeds, as a LaTeX cell."""
    m, s = d[val].mean(), d[val].std()
    if not np.isfinite(m):
        return "--"
    if not np.isfinite(s):
        return f"{m:.{dec}f}"
    return f"{m:.{dec}f} \\tiny{{({s:.{dec}f})}}"


def table_dispersion():
    d = common.load_dispersion()
    rows = []
    ref = d[d.tag == "uniform"]
    for m, lab in (("matching", "Bipartite matching"),
                   ("context_none", "No excavation context")):
        k = ref[ref.method == m]
        if len(k):
            rows.append((lab, "--", "--", _ms(k, "arc_recall"), _ms(k, "ari"),
                         _ms(k, "cross_slip_rate")))

    blocks = [("uniform", None, "One unit per slip (as submitted)"),
              ("sigma", 0.0, "Deposition model, $\\sigma = 0$ mm"),
              ("sigma", 15.0, "\\quad $\\sigma = 15$ mm"),
              ("sigma", 30.0, "\\quad $\\sigma = 30$ mm"),
              ("sigma", 60.0, "\\quad $\\sigma = 60$ mm"),
              ("sigma", 120.0, "\\quad $\\sigma = 120$ mm"),
              ("sigma", 240.0, "\\quad $\\sigma = 240$ mm")]

    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Excavation context under the deposition model.}"
             r" Corpora of about 1150 fragments at preservation state P4, three"
             r" seeds, mean (standard deviation). Context coherence is the"
             r" fraction of true joins whose two fragments carry the same"
             r" recorded unit. Arc recall is the fraction of true joins"
             r" surviving into the candidate set, which is an upper bound on"
             r" the recall of any solver. The gate is the treatment used in the"
             r" submitted manuscript; the graded evidence is"
             r" Eq.~\eqref{eq:ctx}.}",
             r"\label{tab:dispersion}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.6cm}"
             r"lrrrr@{}}", r"\toprule",
             r"Setting & Context & Coherence & Arc recall & Partition index"
             r" & Cross-slip \\", r"\midrule"]
    for lab, _, _, ar, ari, cs in rows:
        lines.append(f"{lab} & -- & -- & {ar} & {ari} & {cs} \\\\")
    lines.append(r"\midrule")
    for tag, sig, lab in blocks:
        sub = d[d.tag == tag] if sig is None else \
            d[(d.tag == tag) & (d.disperse_sigma_mm == sig)]
        if not len(sub):
            continue
        first = True
        for m, mlab in (("context_hard", "gate"),
                        ("context_soft", "graded")):
            k = sub[sub.method == m]
            if not len(k):
                continue
            lines.append(
                f"{lab if first else ''} & {mlab} & "
                f"{k['coherence'].mean():.3f} & {_ms(k, 'arc_recall')} & "
                f"{_ms(k, 'ari')} & {_ms(k, 'cross_slip_rate')} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_dispersion.tex").write_text("\n".join(lines))
    print("table_dispersion.tex")


def table_record():
    d = common.load_dispersion()
    sub = d[d.tag == "record"]
    if not len(sub):
        return
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\caption{\textbf{Reassembly under an imperfect excavation"
             r" record.} Corpora of about 1150 fragments at preservation state"
             r" P4 with dispersion $\sigma = 30$ mm and a 250 mm grid, three"
             r" seeds. Missing denotes the fraction of fragments recovered with"
             r" no context record at all; misfiled denotes the fraction of the"
             r" remainder assigned to a neighbouring square.}",
             r"\label{tab:record}",
             r"\begin{tabular}{rrlrrr}", r"\toprule",
             r"Missing & Misfiled & Context & Coherence & Partition index &"
             r" Cross-slip \\", r"\midrule"]
    combos = sorted({(1 - k, e) for k, e in
                     zip(sub.unit_known_frac, sub.unit_error_frac)})
    for miss, err in combos:
        k0 = sub[(sub.unit_known_frac == 1 - miss) &
                 (sub.unit_error_frac == err)]
        first = True
        for m, mlab in (("context_hard", "gate"), ("context_soft", "graded")):
            k = k0[k0.method == m]
            if not len(k):
                continue
            lines.append(
                f"{miss:.0%} & {err:.0%} & {mlab} & "
                f"{k['coherence'].mean():.3f} & {_ms(k, 'ari')} & "
                f"{_ms(k, 'cross_slip_rate')} \\\\".replace("%", r"\%"))
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_record.tex").write_text("\n".join(lines))
    print("table_record.tex")


def table_notch(path=common.RESULTS / "notch.csv"):
    """Part A: is the tolerance, or the spread of slip length, the limit?"""
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Diagnosis of the binding-notch constraint.}"
             r" Corpora of about 1150 fragments at preservation state P4, three"
             r" seeds, mean (standard deviation). $\Delta$ is measured against"
             r" the same model with the notch constraint removed, within each"
             r" block. In the upper block slip lengths are drawn independently,"
             r" as in the submitted manuscript; in the lower block a roll"
             r" imposes one standard length, while the corpus still spans 231"
             r" to 278 mm. The change in partition index is not monotone in the"
             r" tolerance, although each individual change is consistent"
             r" across seeds, whereas every treatment reduces the rate of"
             r" joins that fuse two slips and raises the fraction of slips"
             r" recovered intact.}",
             r"\label{tab:notch}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.2cm}"
             r"lrrrr@{}}", r"\toprule",
             r"Corpus & Treatment & Partition index & $\Delta$ & Cross-slip"
             r" & Intact \\", r"\midrule"]
    cases = [("no_notch", "no notch constraint"),
             ("tol16", "notch, $\\tau_\\nu = 16$ mm"),
             ("tol8", "notch, $\\tau_\\nu = 8$ mm"),
             ("tol4", "notch, $\\tau_\\nu = 4$ mm"),
             ("tol2", "notch, $\\tau_\\nu = 2$ mm"),
             ("tol4_order", "notch, 4 mm, ordered")]
    for tag, blab in (("tol_iid", "Length drawn per slip"),
                      ("tol_roll", "One standard length per roll")):
        sub = d[d.tag == tag]
        if not len(sub):
            continue
        base = sub[sub.case == "no_notch"]["ari"].mean()
        first = True
        for case, clab in cases:
            k = sub[sub.case == case]
            if not len(k):
                continue
            lines.append(
                f"{blab if first else ''} & {clab} & {_ms(k, 'ari')} & "
                f"{k['ari'].mean() - base:+.3f} & "
                f"{_ms(k, 'cross_slip_rate')} & {_ms(k, 'exact_slip')} \\\\")
            first = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_notch.tex").write_text("\n".join(lines))
    print("table_notch.tex")


def table_roll(path=common.RESULTS / "notch_roll.csv",
               control=common.RESULTS / "notch_roll_control.csv"):
    """Part B: the operating curve, compared at matched grouping quality.

    Matched per seed and then averaged, not pooled and then matched. Pooling
    first mixes corpora whose baselines differ, and the quantity of interest is
    how much each variant improves on *its own* corpus.
    """
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    if Path(control).exists():
        d = pd.concat([d, pd.read_csv(control)], ignore_index=True)
    cases = [("no_notch", "no notch constraint"),
             ("notch", "notch"),
             ("roll_one", "notch, one length per square"),
             ("roll_two", "notch, two lengths per square"),
             ("roll_bundle", "notch, one length per bundle")]
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Slip length estimated jointly over a roll,"
             r" compared at matched grouping quality.} Corpora of about 1150"
             r" fragments at preservation state P4, three seeds, join threshold"
             r" swept over the whole grid. For each seed, every variant is read"
             r" at the threshold bringing it closest to the best slip partition"
             r" index that seed reaches without the notch constraint; the"
             r" columns are the mean over seeds and $\Delta$ is the mean of the"
             r" per-seed differences. The lower block is the control, in which"
             r" slip lengths are drawn independently so that a roll shares no"
             r" standard length for the pooling to find.}",
             r"\label{tab:roll}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.6cm}"
             r"rrrr@{}}", r"\toprule",
             r"Treatment & Partition index & Cross-slip & $\Delta$ &"
             r" Intact \\", r"\midrule"]
    for tag, blab in (("roll_typical", "One standard length per roll"),
                      ("roll_none", "Length drawn per slip (control)")):
        sub = d[d.tag == tag]
        if not len(sub):
            continue
        seeds = sorted(sub.seed.unique())
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{blab}}}}} \\\\")
        base = {}
        for case, clab in cases:
            ari, xs, ex = [], [], []
            for sd in seeds:
                k = sub[(sub.case == case) & (sub.seed == sd)].set_index("theta")
                ref = sub[(sub.case == "no_notch") & (sub.seed == sd)]["ari"].max()
                if not len(k) or not np.isfinite(ref):
                    continue
                m = (k["ari"] - ref).abs().idxmin()
                ari.append(k.loc[m, "ari"])
                xs.append(k.loc[m, "cross_slip_rate"])
                ex.append(k.loc[m, "exact_slip"])
            if not xs:
                continue
            if case == "no_notch":
                base[tag] = np.array(xs)
            dlt = np.mean(np.array(xs) - base[tag]) if tag in base else np.nan
            lines.append(
                f"\\quad {clab} & {np.mean(ari):.3f} & {np.mean(xs):.3f} & "
                f"{dlt:+.3f} & {np.mean(ex):.3f} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_roll.tex").write_text("\n".join(lines))
    print("table_roll.tex")


def table_baselines(path=common.RESULTS / "baselines.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    order = [("top1", "Top ranked candidate"),
             ("mutual", "Mutual best"),
             ("greedy", "Greedy chain assembly"),
             ("path", "Shortest path assembly (DeepZzle)"),
             ("bipartite", "Maximum weight bipartite matching"),
             ("loop", "Loop consistent filtering (loop constraint)"),
             ("constrained", "Constrained assembly (this work)")]
    states = sorted(d.state.unique())
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Comparison with global reassembly methods that"
             r" carry no model of the object.} Slip partition index, corpora of"
             r" about 1150 fragments, three seeds, mean (standard deviation)."
             r" Every method receives the same"
             r" calibrated weights, and each is given its own threshold tuned"
             r" on calibration corpora. The four global methods receive the"
             r" ungated candidate set, since none can represent the excavation"
             r" context or the morphometry that prunes the constrained model's"
             r" arcs; gating their candidates on evidence they cannot use"
             r" would credit them with it.}",
             r"\label{tab:baselines}",
             r"\begin{tabular}{l" + "r" * len(states) + "}", r"\toprule",
             "Method & " + " & ".join(states) + r" \\", r"\midrule"]
    for m, lab in order:
        sub = d[d.method == m]
        if not len(sub):
            continue
        cells = [_ms(sub[sub.state == s], "ari") for s in states]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(states) + 1) +
                 r"}{@{}l}{\emph{Multi fragment slips recovered exactly}} \\")
    for m, lab in order:
        sub = d[d.method == m]
        if not len(sub):
            continue
        cells = [_ms(sub[sub.state == s], "exact_slip") for s in states]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_baselines.tex").write_text("\n".join(lines))
    print("table_baselines.tex")


def table_decomposition(path=common.RESULTS / "decomposition.csv",
                        small=common.RESULTS / "decomposition_small.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    if Path(small).exists():
        # the rerun of the two smaller sizes also compared the selected join
        # sets, not only the objectives, and supersedes them here
        sm = pd.read_csv(small)
        d = pd.concat([d[~d.n_slips.isin(sm.n_slips.unique())], sm],
                      ignore_index=True)
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Cost and optimality of the two ways of solving"
             r" the same model.} Preservation state P4, three seeds, measured"
             r" on an otherwise idle machine with ten solver threads. Proven"
             r" is the fraction of runs returned as proven optimal within the"
             r" budget. Where both solvers prove optimality their objectives"
             r" and their selected joins are identical, which is asserted in"
             r" the released experiment script.}",
             r"\label{tab:decomp}",
             r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"Fragments & Solver & Time (s) & Proven & Components &"
             r" Largest & Partition index \\", r"\midrule"]
    for n in sorted(d.n_frag.unique()):
        first = True
        for solver, lab in (("monolithic", "one model"),
                            ("decomposed", "by component")):
            k = d[(d.n_frag == n) & (d.solver == solver)]
            if not len(k):
                continue
            proven = float(np.mean(k["status"] == "OPTIMAL"))
            comp = "--" if solver == "monolithic" else \
                f"{k['n_components'].mean():.0f}"
            big = "--" if solver == "monolithic" else \
                f"{k['max_component'].mean():.0f}"
            lines.append(
                f"{n if first else ''} & {lab} & {k['solve_s'].mean():.1f} & "
                f"{proven:.2f} & {comp} & {big} & {_ms(k, 'ari')} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_decomposition.tex").write_text("\n".join(lines))
    print("table_decomposition.tex")


def table_real(path=common.RESULTS / "real.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\caption{\textbf{Reassembly on photographs of excavated slips"
             r" against the rendered substrate.} Corpora built from infrared"
             r" photographs of real Qin and Han slips (DeepJiandu, DOI"
             r" 10.57760/sciencedb.08560), cut by"
             r" the same fracture model, described by the same code and scored"
             r" by the same matcher, which was trained on synthetic corpora"
             r" only and is not retrained here. The control column is the"
             r" identical experiment on rendered bamboo at the same seeds."
             r" Three seeds, mean (standard deviation).}",
             r"\label{tab:real}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Substrate & Method & Top-1 & Top-50 & Partition index &"
             r" Cross-slip \\", r"\midrule"]
    for sub_tag, lab in (("synthetic", "Rendered bamboo"),
                         ("real", "Photographed slips")):
        s = d[d.substrate == sub_tag]
        if not len(s):
            continue
        first = True
        for m, mlab in (("bipartite", "bipartite matching"),
                        ("constrained", "constrained (this work)")):
            k = s[s.method == m]
            if not len(k):
                continue
            lines.append(
                f"{lab if first else ''} & {mlab} & "
                f"{k['top1'].mean():.3f} & {k['top50'].mean():.3f} & "
                f"{_ms(k, 'ari')} & {_ms(k, 'cross_slip_rate')} \\\\")
            first = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_real.tex").write_text("\n".join(lines))
    print("table_real.tex")


# ---------------------------------------------------------------------------
# The headline tables, regenerated under the deposition model
# ---------------------------------------------------------------------------

MAIN_LABEL = {"top1": "Top ranked candidate", "mutual": "Mutual best",
              "matching": "Maximum weight bipartite matching",
              "morph": r"\quad $+$ morphometry and context",
              "length": r"\quad $+$ slip length",
              "full": "Pairwise constraints (all evidence)",
              "latent": "Latent slip properties (this work)"}
MAIN_ORDER = ["top1", "mutual", "matching", "morph", "length", "full",
              "latent"]


def table_main_rev(path=common.RESULTS / "rerun_main.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    states = ["P1", "P2", "P3", "P4", "P5"]
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Reassembly quality across the preservation"
             r" ladder.} Corpora of about 1150 fragments, five seeds, mean"
             r" (standard deviation). Fragments are deposited, dispersed and"
             r" recorded as Section~\ref{sec:deposition} describes, and the"
             r" excavation record enters as graded evidence"
             r" (Eq.~\eqref{eq:ctx}); the context coherence of these corpora"
             r" is 0.63. Every method is a constraint subset of the same"
             r" model, solved by the same solver on the same candidate set,"
             r" and each is evaluated at its own threshold tuned on"
             r" calibration corpora on the measure reported here.}",
             r"\label{tab:main}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm}"
             + "r" * len(states) + r"@{}}", r"\toprule",
             r"\multicolumn{" + str(len(states) + 1) +
             r"}{@{}l}{\emph{Slip partition index}} \\"]
    for val, head in (("ari", None),
                      ("exact_slip", r"\emph{Multi fragment slips recovered exactly}"),
                      ("cross_slip_rate", r"\emph{Joins fusing two different slips}")):
        if head:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(len(states) + 1) +
                         r"}{@{}l}{" + head + r"} \\")
        lines.append(r"\cmidrule(l){2-" + str(len(states) + 1) + "}")
        lines.append("Method & " + " & ".join(states) + r" \\")
        for m in MAIN_ORDER:
            sub = d[d.method == m]
            if not len(sub):
                continue
            cells = [_ms(sub[sub.state == s], val) for s in states]
            lines.append(f"{MAIN_LABEL[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_main_rev.tex").write_text("\n".join(lines))
    print("table_main_rev.tex")


ABL_LABEL = {"latent": "Latent slip properties (this work)",
             "no_strat": r"\quad $-$ excavation context",
             "no_length": r"\quad $-$ slip length",
             "no_hand": r"\quad $-$ scribal hand",
             "no_width": r"\quad $-$ slip width",
             "no_notch": r"\quad $-$ binding notches",
             "full": "Pairwise constraints",
             "full_no_notch": r"\quad $-$ binding notches",
             "full_no_hand": r"\quad $-$ scribal hand"}


def table_ablation_rev(path=common.RESULTS / "rerun_ablation.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    states = ["P3", "P4"]
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{What each source of evidence is worth,"
             r" measured by removing it.} Corpora of about 1150 fragments,"
             r" five seeds, mean (standard deviation), under the deposition"
             r" model with the excavation record read as graded evidence."
             r" $\Delta$ is the change in slip partition index against the"
             r" complete model of the same block. Excavation context is the"
             r" most valuable single source at both states. The binding"
             r" notches are the clearest case of a source whose value does"
             r" not appear in the partition index: removing them costs 0.014"
             r" of index at P4 but raises the rate of joins that fuse two"
             r" slips by 39\% and lowers exact recovery by 0.044.}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.0cm}"
             r"rrrrrr@{}}", r"\toprule",
             r"& \multicolumn{3}{c}{P3} & \multicolumn{3}{c}{P4} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Model & Index & $\Delta$ & Cross-slip & Index & $\Delta$"
             r" & Cross-slip \\", r"\midrule"]
    for block in (["latent", "no_strat", "no_length", "no_hand", "no_width",
                   "no_notch"],
                  ["full", "full_no_notch", "full_no_hand"]):
        base = {s: d[(d.state == s) & (d.method == block[0])]["ari"].mean()
                for s in states}
        for m in block:
            cells = []
            for s in states:
                k = d[(d.state == s) & (d.method == m)]
                if not len(k):
                    cells += ["--", "--", "--"]
                    continue
                cells += [_ms(k, "ari"), f"{k['ari'].mean() - base[s]:+.3f}",
                          f"{k['cross_slip_rate'].mean():.3f}"]
            lines.append(f"{ABL_LABEL[m]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_ablation_rev.tex").write_text("\n".join(lines))
    print("table_ablation_rev.tex")


def table_exchange_rev(path=common.RESULTS / "rerun_main.csv"):
    """What the constraint set is worth in units of matcher accuracy.

    Recomputed from the reran corpora, so that this table and the main table
    describe the same experiment.
    """
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    g = d.groupby(["state", "method"]).agg(top1=("top1", "mean"),
                                           ari=("ari", "mean")).reset_index()
    bip = g[g.method == "matching"].set_index("state")
    lat = g[g.method == "latent"].set_index("state")
    states = ["P5", "P4", "P3", "P2", "P1"]
    x = np.array([bip.loc[s, "top1"] for s in states])
    y = np.array([bip.loc[s, "ari"] for s in states])
    o = np.argsort(x)
    xs, ys = x[o], y[o]

    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{\textbf{Constraint set expressed as an equivalent gain"
             r" in pairwise matching accuracy.} For each preservation state:"
             r" the measured Top-1 accuracy of the matcher; the slip partition"
             r" index attained by the proposed model at that accuracy; and the"
             r" Top-1 accuracy that maximum weight bipartite matching would"
             r" require to attain the same partition index, obtained by"
             r" interpolating its performance along the preservation ladder."
             r" \emph{Unreachable} denotes states at which bipartite matching"
             r" does not attain that partition index at any accuracy observed"
             r" on the ladder, including that of the best preserved corpus.}",
             r"\label{tab:exchange}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"State & Matcher Top-1 & Proposed index & Equivalent Top-1 \\",
             r"\midrule"]
    for s in states:
        t, la = bip.loc[s, "top1"], lat.loc[s, "ari"]
        if la > ys.max():
            eq = r"\emph{unreachable}"
        else:
            v = float(np.interp(la, ys, xs))
            eq = f"{v:.3f} " + r"\tiny{(+" + f"{v - t:.3f}" + ")}"
        lines.append(f"{s} & {t:.3f} & {la:.3f} & {eq}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_exchange_rev.tex").write_text("\n".join(lines))
    print("table_exchange_rev.tex")


def table_scaling_rev(path=common.RESULTS / "rerun_scaling.csv"):
    """Behaviour with corpus size, under the deposition model."""
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    d["grp"] = pd.cut(d.n_frag, [0, 600, 1500, 3000, 6000],
                      labels=["425", "1151", "2540", "5060"])
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Behaviour with corpus size.} Preservation"
             r" state P4 under the deposition model, three seeds per size,"
             r" mean over seeds. Each method is evaluated at its own threshold,"
             r" tuned on calibration corpora. The advantage of the constrained"
             r" model over bipartite matching grows with the corpus on every"
             r" measure.}",
             r"\label{stab:scaling}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Fragments & 425 & 1151 & 2540 & 5060 \\", r"\midrule"]
    for val, head in (("ari", "Slip partition index"),
                      ("exact_slip", "Slips recovered exactly"),
                      ("cross_slip_rate", "Joins fusing two slips")):
        lines.append(r"\multicolumn{5}{@{}l}{\emph{" + head + r"}} \\")
        for m, lab in (("matching", r"\quad bipartite matching"),
                       ("full", r"\quad pairwise constraints"),
                       ("latent", r"\quad latent properties")):
            k = d[d.method == m]
            cells = [f"{k[k.grp == g][val].mean():.3f}"
                     for g in ["425", "1151", "2540", "5060"]]
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_scaling_rev.tex").write_text("\n".join(lines))
    print("table_scaling_rev.tex")


if __name__ == "__main__":
    table_dispersion()
    table_record()
    table_notch()
    table_roll()
    table_baselines()
    table_decomposition()
    table_real()
    table_main_rev()
    table_ablation_rev()
    table_exchange_rev()
    table_scaling_rev()
