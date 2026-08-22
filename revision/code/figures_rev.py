"""Figures for the revision, in the style of the submitted manuscript.

Each function is independent and skips itself when its result file is missing,
so figures can be produced as the experiments finish rather than only at the
end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import FIGS, RESULTS

sys.path.insert(0, str(common.ROOT / "src"))
import style  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

style.apply()
FIGS.mkdir(parents=True, exist_ok=True)

C_HARD = style.RAMP[1]
C_SOFT = style.ACCENT

def _legend_below(fig, ax, ncol=4, bottom=0.20):
    """One shared legend under the panels.

    A legend inside an axes has to sit somewhere, and on these figures every
    corner already holds a curve. Putting it below the panels is the only
    placement that cannot overlap the data.
    """
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom)
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.5,
               columnspacing=1.4)



def _new(ncols, w, h, key=None, **kw):
    """A figure whose typography is authored for the width it prints at.

    The manuscript places figures at 0.95\\textwidth. A canvas wider than that
    is reduced before a reader sees it, and its type is reduced with it, so the
    point sizes are set from the canvas width rather than left at their
    defaults. Returns the scale factor as well, for sizes given directly to a
    plotting call.
    """
    s = style.scale_for(w, key=key)
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), **kw)
    return fig, axes, s


C_NONE = style.GREY_1
C_MATCH = style.GREY_2

LABEL = {"context_hard": "context as a gate",
         "context_soft": "context as graded evidence",
         "context_none": "no context",
         "matching": "bipartite matching"}
COLOR = {"context_hard": C_HARD, "context_soft": C_SOFT,
         "context_none": C_NONE, "matching": C_MATCH}
MARK = {"context_hard": "s", "context_soft": "o", "context_none": "^",
        "matching": "v"}


def _agg(d, keys, val):
    g = d.groupby(keys)[val]
    return g.mean(), g.std()


def fig_dispersion():
    d = common.load_dispersion()
    ref = d[d.tag == "uniform"]
    refs = {m: ref[ref.method == m] for m in ("context_none", "matching")}

    fig, axes, s = _new(3, 11.0, 3.9, key="figR1_dispersion",
                        gridspec_kw=dict(wspace=0.42))

    for ax, (tag, xcol, xlabel) in zip(
            axes[:2],
            [("sigma", "disperse_sigma_mm",
              "dispersion $\\sigma$ (mm)"),
             ("grid", "block_mm", "grid square (mm)")]):
        sub = d[d.tag == tag]
        if not len(sub):
            ax.set_xlabel(xlabel)
            ax.set_ylabel("slip partition index")
            style.finish(ax)
            continue
        for m in ("context_hard", "context_soft"):
            k = sub[sub.method == m]
            mu, sd = _agg(k, xcol, "ari")
            ax.errorbar(mu.index, mu.values, yerr=sd.values, marker=MARK[m],
                        color=COLOR[m], label=LABEL[m], capsize=1.2 * s)
        for m, ls in (("context_none", "--"), ("matching", ":")):
            if len(refs[m]):
                ax.axhline(refs[m]["ari"].mean(), ls=ls, lw=0.45 * s * s,
                           color=COLOR[m], label=LABEL[m])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("slip partition index")
        ax.margins(x=0.08)
        if tag == "grid":
            ax.set_xscale("log")
            ax.set_xticks([125, 250, 500, 1000])
            ax.set_xticklabels(["125", "250", "500", "1000"])
            ax.set_xticks([], minor=True)
        style.finish(ax)

    ax = axes[2]
    sub = d[d.tag.isin(("sigma", "grid", "record"))]
    for m in ("context_hard", "context_soft"):
        k = sub[sub.method == m]
        ax.scatter(k["coherence"], k["ari"], s=18, marker=MARK[m],
                   color=COLOR[m], label=LABEL[m], alpha=0.85,
                   edgecolors="none")
    for m, ls in (("context_none", "--"), ("matching", ":")):
        if len(refs[m]):
            ax.axhline(refs[m]["ari"].mean(), ls=ls, lw=0.45 * s * s, color=COLOR[m],
                       label=LABEL[m])
    ax.set_xlabel("context coherence")
    ax.set_ylabel("slip partition index")
    style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[2], ncol=4, bottom=0.36)
    fig.savefig(FIGS / "figR1_dispersion.png")
    plt.close(fig)
    print("figR1_dispersion.png")


def fig_arc_recall():
    """Why the gate fails: it deletes the joins it is meant to protect."""
    d = common.load_dispersion()
    sub = d[d.tag == "sigma"]
    fig, axes, s = _new(2, 7.6, 3.2, key="figR2_arc_recall")
    for m in ("context_hard", "context_soft", "context_none"):
        k = sub[sub.method == m]
        if not len(k):
            continue
        mu, sd = _agg(k, "disperse_sigma_mm", "arc_recall")
        axes[0].errorbar(mu.index, mu.values, yerr=sd.values, marker=MARK[m],
                         color=COLOR[m], label=LABEL[m], capsize=1.2 * s)
        mu, sd = _agg(k, "disperse_sigma_mm", "cross_slip_rate")
        axes[1].errorbar(mu.index, mu.values, yerr=sd.values, marker=MARK[m],
                         color=COLOR[m], label=LABEL[m], capsize=1.2 * s)
    axes[0].set_ylabel("true joins in candidate set")
    axes[1].set_ylabel("joins fusing two slips")
    for ax in axes:
        ax.set_xlabel("dispersion $\\sigma$ (mm)")
        style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=3, bottom=0.32)
    fig.savefig(FIGS / "figR2_arc_recall.png")
    plt.close(fig)
    print("figR2_arc_recall.png")


def fig_notch(path=RESULTS / "notch.csv",
              frontier=RESULTS / "notch_roll.csv"):
    """Why the notch constraint was weak, and what the joint length estimate does.

    Panel a asks whether the tolerance was the limit. Panels b and c ask the
    question the single-threshold comparison could not answer: the notch
    evidence halves the damaging error while lowering the partition index,
    which is what moving along an operating curve looks like, so the curves
    themselves are plotted and compared at matched operating points.
    """
    have_a = Path(path).exists()
    have_f = Path(frontier).exists()
    if not (have_a or have_f):
        return
    fig, axes, s = _new(3, 11.0, 3.9, key="figR3_notch",
                        gridspec_kw=dict(wspace=0.42))

    # a) is it the tolerance?
    ax = axes[0]
    if have_a:
        d = pd.read_csv(path)
        tol_order = ["tol16", "tol8", "tol4", "tol2"]
        xs = [16, 8, 4, 2]
        for tag, col, mk, lab in (
                ("tol_iid", C_NONE, "^", "length drawn per slip"),
                ("tol_roll", C_SOFT, "o", "one standard length per roll")):
            sub = d[d.tag == tag]
            if not len(sub):
                continue
            base = sub[sub.case == "no_notch"]["ari"].mean()
            mu = [sub[sub.case == c]["ari"].mean() - base for c in tol_order]
            sd = [sub[sub.case == c]["ari"].std() for c in tol_order]
            ax.errorbar(xs, mu, yerr=sd, marker=mk, color=col, label=lab,
                        capsize=1.2 * s)
        ax.axhline(0, color=style.AXIS, lw=0.5 * s)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xticks([], minor=True)
    ax.set_xlabel("notch tolerance $\\tau_\\nu$ (mm)")
    ax.set_ylabel("change in partition index")
    style.finish(ax)

    # b), c) the operating curve, with and without a roll standard
    order = [("no_notch", C_NONE, "^", "no notch constraint"),
             ("notch", style.RAMP[1], "s", "notch"),
             ("roll_one", style.RAMP[3], "D", "notch, length pooled by square"),
             ("roll_two", style.ACCENT_SOFT, "v",
              "notch, two lengths per square"),
             ("roll_bundle", C_SOFT, "o", "notch, length pooled by bundle")]
    if have_f:
        f = pd.read_csv(frontier)
        ctl = RESULTS / "notch_roll_control.csv"
        if ctl.exists():
            f = pd.concat([f, pd.read_csv(ctl)], ignore_index=True)
        for ax, tag in ((axes[1], "roll_typical"), (axes[2], "roll_none")):
            sub = f[f.tag == tag]
            for case, col, mk, lab in order:
                k = sub[sub.case == case]
                if not len(k):
                    continue
                g = k.groupby("theta")[["ari", "cross_slip_rate"]].mean()
                g = g.sort_values("cross_slip_rate")
                ax.plot(g["cross_slip_rate"], g["ari"], marker=mk, color=col,
                        label=lab, ms=2.2 * s, lw=0.75 * s)
    for ax in axes[1:]:
        ax.set_xlabel("joins fusing two slips")
        ax.set_ylabel("slip partition index")
        style.finish(ax)
    style.panel_titles(list(axes))
    if have_f:
        _legend_below(fig, axes[1], ncol=3, bottom=0.36)
    else:
        fig.tight_layout()
    fig.savefig(FIGS / "figR3_notch.png")
    plt.close(fig)
    print("figR3_notch.png")


def fig_baselines(path=RESULTS / "baselines.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    order = ["top1", "mutual", "greedy", "path", "bipartite", "loop",
             "constrained"]
    labs = ["top ranked", "mutual best", "greedy", "shortest path",
            "bipartite", "loop consistent", "constrained (this work)"]
    cols = [style.GREY_1, style.GREY_2, style.RAMP[0], style.RAMP[1],
            style.RAMP[2], style.RAMP[3], style.ACCENT]
    states = sorted(d.state.unique())
    fig, axes, s = _new(2, 9.4, 3.4, key="figR4_baselines")
    for ax, val, ylab in ((axes[0], "ari", "slip partition index"),
                          (axes[1], "cross_slip_rate",
                           "joins fusing two slips")):
        x = np.arange(len(states))
        w = 0.8 / len(order)
        for k, (m, lab, c) in enumerate(zip(order, labs, cols)):
            sub = d[d.method == m]
            mu = [sub[sub.state == s][val].mean() for s in states]
            sd = [sub[sub.state == s][val].std() for s in states]
            ax.bar(x + (k - len(order) / 2 + 0.5) * w, mu, width=w, yerr=sd,
                   capsize=1.0 * s, color=c, label=lab, error_kw=dict(lw=0.45 * s))
        ax.set_xticks(x)
        ax.set_xticklabels(states)
        ax.set_xlabel("preservation state")
        ax.set_ylabel(ylab)
        style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=3, bottom=0.40)
    fig.savefig(FIGS / "figR4_baselines.png")
    plt.close(fig)
    print("figR4_baselines.png")


def fig_decomposition(path=RESULTS / "decomposition.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    fig, axes, s = _new(3, 11.0, 3.9, key="figR5_decomposition",
                        gridspec_kw=dict(wspace=0.42))
    for solver, col, mk, lab in (("monolithic", style.GREY_2, "s",
                                  "one model for the corpus"),
                                 ("decomposed", style.ACCENT, "o",
                                  "component by component")):
        sub = d[d.solver == solver]
        if not len(sub):
            continue
        # A symmetric standard deviation is the wrong interval on a
        # logarithmic axis: at the largest sizes the spread exceeds the mean,
        # the lower end falls below zero and the bar is drawn the full height
        # of the panel. The whiskers run from the fastest run to the slowest,
        # which is what "not reproducible at this scale" actually looks like.
        g = sub.groupby("n_frag")["solve_s"]
        mu, lo, hi = g.mean(), g.min(), g.max()
        axes[0].errorbar(mu.index, mu.values,
                         yerr=[mu.values - lo.values, hi.values - mu.values],
                         marker=mk, color=col, label=lab, capsize=1.2 * s)
        opt = sub.groupby("n_frag")["status"].apply(
            lambda x: float(np.mean(x == "OPTIMAL")))
        axes[1].plot(opt.index, opt.values, marker=mk, color=col, label=lab)
    if (d["solve_s"] > 0).any():
        axes[0].set_yscale("log")
    axes[0].set_ylabel("solve time (s)")
    axes[1].set_ylabel("proven optimal")
    axes[1].set_ylim(-0.05, 1.05)
    sub = d[d.solver == "decomposed"]
    if len(sub) and sub["max_component"].notna().any():
        mu, sd = _agg(sub, "n_frag", "max_component")
        axes[2].errorbar(mu.index, mu.values, yerr=sd.values, marker="o",
                         color=style.ACCENT, capsize=1.2 * s)
        xs = sorted(d.n_frag.unique())
        axes[2].plot(xs, xs, ls=":", color=style.AXIS)
        axes[2].annotate("whole corpus", (xs[-1], xs[-1]), ha="right",
                         va="bottom", color=style.AXIS,
                         fontsize=style.PRINTED["tick"] * s)
        axes[2].set_yscale("log")
    axes[2].set_ylabel("largest component")
    for ax in axes:
        ax.set_xlabel("fragments in the corpus")
        style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=3, bottom=0.32)
    fig.savefig(FIGS / "figR5_decomposition.png")
    plt.close(fig)
    print("figR5_decomposition.png")


def fig_realism(path=RESULTS / "realism.csv", real=RESULTS / "real.csv",
                uneven=RESULTS / "realism_uneven.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    if Path(uneven).exists():
        # the corrected rerun of the uneven-preservation part supersedes
        # whatever the first pass wrote
        d = pd.concat([d[d.part != "uneven"], pd.read_csv(uneven)],
                      ignore_index=True)
    fig, axes, s = _new(3, 11.0, 3.9, key="figR6_realism",
                        gridspec_kw=dict(wspace=0.42))

    ax = axes[0]
    sub = d[d.part == "misspecification"]
    for m, col, mk, lab in (("constrained", style.ACCENT, "o",
                             "constrained (this work)"),
                            ("bipartite", style.GREY_2, "s",
                             "bipartite matching")):
        k = sub[sub.method == m]
        mu, sd = _agg(k, "level", "ari")
        ax.errorbar(mu.index, mu.values, yerr=sd.values, marker=mk, color=col,
                    label=lab, capsize=1.2 * s)
    ax.set_xlabel("taphonomy misspecified")
    ax.set_ylabel("slip partition index")
    style.finish(ax)

    ax = axes[1]
    sub = d[d.part == "uneven"] if "mix" in d.columns else d.iloc[0:0]
    mixes = ["P4_uniform", "P3_to_P5", "P2_to_P5"]
    labs = ["P4 only", "P3-P5", "P2-P5"]
    x = np.arange(len(mixes))
    for k, (m, col, lab) in enumerate((("constrained", style.ACCENT,
                                        "constrained"),
                                       ("bipartite", style.GREY_2,
                                        "bipartite"))):
        kk = sub[sub.method == m] if len(sub) else sub
        mu = [kk[kk["mix"] == g]["ari"].mean() if len(kk) else np.nan
              for g in mixes]
        sd = [kk[kk["mix"] == g]["ari"].std() if len(kk) else np.nan
              for g in mixes]
        ax.bar(x + (k - 0.5) * 0.36, mu, width=0.36, yerr=sd, capsize=1.2 * s,
               color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_xlabel("preservation spread")
    ax.set_ylabel("slip partition index")
    style.finish(ax)

    ax = axes[2]
    if Path(real).exists():
        r = pd.read_csv(real)
        subs = ["synthetic", "real"]
        x = np.arange(len(subs))
        for k, (m, col, lab) in enumerate((("constrained", style.ACCENT,
                                            "constrained"),
                                           ("bipartite", style.GREY_2,
                                            "bipartite"))):
            kk = r[r.method == m]
            mu = [kk[kk.substrate == s]["ari"].mean() for s in subs]
            sd = [kk[kk.substrate == s]["ari"].std() for s in subs]
            ax.bar(x + (k - 0.5) * 0.36, mu, width=0.36, yerr=sd, capsize=1.2 * s,
                   color=col, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(["rendered", "photographed"])
        ax.set_xlabel("substrate")
        ax.set_ylabel("slip partition index")
    style.finish(ax)

    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=2, bottom=0.34)
    fig.savefig(FIGS / "figR6_realism.png")
    plt.close(fig)
    print("figR6_realism.png")


def _main():
    fig_dispersion()
    fig_arc_recall()
    fig_notch()
    fig_baselines()
    fig_decomposition()
    fig_realism()
    fig_real_examples()
    fig_main_rev()
    fig_ablation_rev()
    fig_scaling_rev()
    fig_substitution_rev()
    fig_frontier_rev()
    fig_pr_spread_rev()


def fig_real_examples(screened=common.ROOT / "revision" / "data_real" /
                      "screened.json", n=6, seed=3):
    """What the corpora built on photographed slips actually look like.

    Real material, generated fracture: the panel shows the photographs after
    segmentation and rectification, and the fragments they were cut into, so a
    reader can see what the matcher is being asked to work from.
    """
    import json
    if not Path(screened).exists():
        return
    import real_corpus
    from common import synth
    import preservation as pres

    paths = json.loads(Path(screened).read_text())
    sub = real_corpus.RealSubstrate(paths, seed=seed)
    spec = common.make_spec(11, n, "P4",
                            dep=synth.DepositionSpec(model="spatial"))
    inst = synth.generate_instance(spec, substrate=sub, render_gallery=40)

    by_slip = {}
    for g in inst["gallery"]:
        f = inst["frags"][g["frag_id"]]
        by_slip.setdefault(f["slip_id"], []).append((f["order"], g))
    slips = [s for s in sorted(by_slip) if len(by_slip[s]) > 1][:n]
    if not slips:
        return

    cmap = matplotlib.colormaps["gray"].with_extremes(bad="white")
    fig, axes, s = _new(len(slips), 1.35 * len(slips), 6.2, key="figR7_real_examples")
    axes = np.atleast_1d(axes)
    wmax = max(p["img"].shape[1] for sid in slips for p in
               (g for _, g in by_slip[sid]))
    for ax, sid in zip(axes, slips):
        pieces = [g for _, g in sorted(by_slip[sid])]
        gap = 16
        canvas = []
        for p in pieces:
            im = np.where(p["mask"] > 0.5, p["img"], np.nan)
            pad = wmax - im.shape[1]
            im = np.pad(im, ((0, 0), (pad // 2, pad - pad // 2)),
                        constant_values=np.nan)
            canvas.append(im.astype(np.float32))
            canvas.append(np.full((gap, wmax), np.nan, np.float32))
        img = np.vstack(canvas[:-1])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                  aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"slip {sid}", fontsize=style.PRINTED["tick"] * s, fontweight="normal")
    fig.tight_layout()
    fig.savefig(FIGS / "figR7_real_examples.png")
    plt.close(fig)
    print("figR7_real_examples.png")


def fig_main_rev(path=RESULTS / "rerun_main.csv"):
    """The main comparison and the ablation, under the deposition model."""
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    states = ["P1", "P2", "P3", "P4", "P5"]
    order = [("top1", style.GREY_1, "^", "top ranked"),
             ("mutual", style.GREY_2, "v", "mutual best"),
             ("matching", style.RAMP[0], "s", "bipartite matching"),
             ("morph", style.RAMP[1], "D", "$+$ morphometry, context"),
             ("length", style.RAMP[2], "P", "$+$ slip length"),
             ("full", style.RAMP[3], "X", "pairwise constraints"),
             ("latent", style.ACCENT, "o", "latent properties (this work)")]
    fig, axes, s = _new(3, 11.0, 3.9, key="figR8_main",
                        gridspec_kw=dict(wspace=0.42))
    for ax, val, ylab in ((axes[0], "ari", "slip partition index"),
                          (axes[1], "exact_slip",
                           "slips recovered exactly"),
                          (axes[2], "cross_slip_rate",
                           "joins fusing two slips")):
        for m, col, mk, lab in order:
            sub = d[d.method == m]
            mu = [sub[sub.state == s][val].mean() for s in states]
            sd = [sub[sub.state == s][val].std() for s in states]
            ax.errorbar(range(len(states)), mu, yerr=sd, marker=mk, color=col,
                        label=lab, capsize=1.2 * s, ms=2.4 * s)
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states)
        ax.set_xlabel("preservation state")
        ax.set_ylabel(ylab)
        style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=4, bottom=0.34)
    fig.savefig(FIGS / "figR8_main.png")
    plt.close(fig)
    print("figR8_main.png")


def fig_ablation_rev(path=RESULTS / "rerun_ablation.csv"):
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    lat = [("no_strat", "excavation context"), ("no_length", "slip length"),
           ("no_hand", "scribal hand"), ("no_width", "slip width"),
           ("no_notch", "binding notches")]
    fig, axes, s = _new(2, 8.6, 3.6, key="figR9_ablation")
    for ax, st in zip(axes, ["P3", "P4"]):
        k = d[d.state == st]
        base = k[k.method == "latent"]["ari"].mean()
        basex = k[k.method == "latent"]["cross_slip_rate"].mean()
        y = np.arange(len(lat))
        dari = [k[k.method == m]["ari"].mean() - base for m, _ in lat]
        dx = [k[k.method == m]["cross_slip_rate"].mean() - basex
              for m, _ in lat]
        ax.barh(y + 0.19, dari, height=0.36, color=style.RAMP[2],
                label="change in partition index")
        ax.barh(y - 0.19, dx, height=0.36, color=style.ACCENT,
                label="change in joins fusing two slips")
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in lat], fontsize=style.PRINTED["tick"] * s)
        ax.axvline(0, color=style.AXIS, lw=0.5 * s)
        ax.set_xlabel("effect of removing the source")
        ax.invert_yaxis()
        style.finish(ax, grid_axis="x")
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=2, bottom=0.30)
    fig.savefig(FIGS / "figR9_ablation.png")
    plt.close(fig)
    print("figR9_ablation.png")


def fig_scaling_rev(path=RESULTS / "rerun_scaling.csv"):
    """Behaviour with corpus size, under the deposition model."""
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    order = [("matching", style.RAMP[0], "s", "bipartite matching"),
             ("full", style.RAMP[3], "X", "pairwise constraints"),
             ("latent", style.ACCENT, "o", "latent properties (this work)")]
    fig, axes, s = _new(3, 11.0, 3.9, key="figR10_scaling",
                        gridspec_kw=dict(wspace=0.42))
    for ax, val, ylab in ((axes[0], "ari", "slip partition index"),
                          (axes[1], "exact_slip",
                           "slips recovered exactly"),
                          (axes[2], "cross_slip_rate",
                           "joins fusing two slips")):
        for m, col, mk, lab in order:
            k = d[d.method == m]
            k = k.assign(grp=pd.cut(k.n_frag, [0, 600, 1500, 3000, 6000]))
            g = k.groupby("grp", observed=True).agg(
                x=("n_frag", "mean"), mu=(val, "mean"), sd=(val, "std"))
            ax.errorbar(g["x"], g["mu"], yerr=g["sd"], marker=mk, color=col,
                        label=lab, capsize=1.2 * s, ms=2.4 * s)
        # Four corpus sizes on a log axis: the automatic decade and minor
        # labels collide, so the sizes themselves are the ticks.
        ax.set_xscale("log")
        xs = sorted(d.groupby(pd.cut(d.n_frag, [0, 600, 1500, 3000, 6000]),
                              observed=True)["n_frag"].mean())
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(round(v, -1)):d}" for v in xs])
        ax.set_xticks([], minor=True)
        ax.set_xlabel("fragments in the corpus")
        ax.set_ylabel(ylab)
        style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=3, bottom=0.32)
    fig.savefig(FIGS / "figR10_scaling.png")
    plt.close(fig)
    print("figR10_scaling.png")


def fig_substitution_rev(path=RESULTS / "rerun_main.csv"):
    """What the constraints are worth in units of matcher accuracy.

    Drawn from the reran corpora, so that the curve and Table 6 describe the
    same experiment. Each point is one preservation state, placed at the Top-1
    accuracy the matcher reached on it.
    """
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    order = [("top1", style.GREY_1, "^", "top ranked"),
             ("mutual", style.GREY_2, "v", "mutual best"),
             ("matching", style.RAMP[0], "s", "bipartite matching"),
             ("morph", style.RAMP[1], "D", "$+$ morphometry, context"),
             ("length", style.RAMP[2], "P", "$+$ slip length"),
             ("full", style.RAMP[3], "X", "pairwise constraints"),
             ("latent", style.ACCENT, "o", "latent properties (this work)")]
    fig, axes, s = _new(2, 8.0, 3.6, key="figR12_substitution", gridspec_kw=dict(wspace=0.34))
    for ax, val, ylab in ((axes[0], "ari", "slip partition index"),
                          (axes[1], "exact_slip", "slips recovered exactly")):
        for m, col, mk, lab in order:
            k = d[d.method == m]
            g = k.groupby("state").agg(x=("top1", "mean"), mu=(val, "mean"),
                                       sd=(val, "std")).sort_values("x")
            ax.errorbar(g["x"], g["mu"], yerr=g["sd"], marker=mk, color=col,
                        label=lab, capsize=1.2 * s)
        ax.set_xlabel("matcher Top-1 accuracy")
        ax.set_ylabel(ylab)
        style.finish(ax)
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=3, bottom=0.36)
    fig.savefig(FIGS / "figR12_substitution.png")
    plt.close(fig)
    print("figR12_substitution.png")


def fig_frontier_rev(path=RESULTS / "frontier_rev.csv"):
    """The operating curve of the two formulations, under the deposition model.

    Every point is one value of the join threshold, so the comparison is
    between curves and not between two tuned points.
    """
    if not Path(path).exists():
        return
    d = pd.read_csv(path)
    states = [x for x in ("P2", "P3", "P4", "P5") if x in set(d.state)]
    order = [("full", style.RAMP[3], "X", "pairwise constraints"),
             ("latent", style.ACCENT, "o", "latent properties (this work)")]
    fig, axes, s = _new(len(states), 3.0 * len(states), 3.4, key="figR13_frontier",
                        gridspec_kw=dict(wspace=0.40))
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, states):
        for m, col, mk, lab in order:
            g = d[(d.state == st) & (d.method == m)].sort_values("cross_slip_rate")
            ax.plot(g["cross_slip_rate"], g["ari"], marker=mk, color=col,
                    label=lab, ms=2.2 * s, lw=0.75 * s)
        ax.set_xlabel("joins fusing two slips")
        style.finish(ax)
    axes[0].set_ylabel("slip partition index")
    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=2, bottom=0.34)
    fig.savefig(FIGS / "figR13_frontier.png")
    plt.close(fig)
    print("figR13_frontier.png")


def fig_pr_spread_rev(front=RESULTS / "frontier_rev.csv",
                      runs=RESULTS / "rerun_main.csv"):
    """Operating curves and the spread over seeds, both from the rerun.

    The submitted version of this figure drew its curves and its box plot from
    corpora that predate the deposition model, while the caption quoted the
    reran significance values. Both halves now come from the same experiment
    as the main table.
    """
    if not (Path(front).exists() and Path(runs).exists()):
        return
    d = pd.read_csv(front)
    r = pd.read_csv(runs)
    states = [x for x in ("P2", "P3", "P4", "P5") if x in set(d.state)]
    order = [("full", style.RAMP[3], "X", "pairwise constraints"),
             ("latent", style.ACCENT, "o", "latent properties (this work)")]
    fig, axes, s = _new(len(states) + 1, 3.0 * (len(states) + 1), 3.4,
                        key="figR14_pr_spread",
                        gridspec_kw=dict(wspace=0.62))
    for ax, st in zip(axes, states):
        for m, col, mk, lab in order:
            g = d[(d.state == st) & (d.method == m)].sort_values("recall")
            ax.plot(g["recall"], g["precision"], marker=mk, color=col,
                    label=lab, ms=2.2 * s, lw=0.75 * s)
        ax.set_xlabel("recall")
        # three panels across the page leave each one about 1.2 in wide, which
        # the default tick count runs together
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        style.finish(ax, grid_axis="both")
    axes[0].set_ylabel("precision")

    keep = ["matching", "morph", "length", "full", "latent"]
    ref = r[r.state == "P4"]
    ax = axes[-1]
    data = [ref[ref.method == m].ari.values for m in keep]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color=style.INK, lw=0.7 * s))
    for patch, m in zip(bp["boxes"], keep):
        patch.set_facecolor(style.METHOD_COLORS.get(m, style.MUTED))
        patch.set_alpha(0.85)
        patch.set_edgecolor(style.INK)
    for k, v in enumerate(data):
        ax.scatter(np.full(len(v), k + 1) + np.linspace(-0.12, 0.12, len(v)),
                   v, s=2.0 * s, color=style.INK, zorder=5, alpha=0.7)
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels(["match", "morph", "length", "pairwise", "latent"],
                       rotation=40, ha="right")
    ax.set_ylabel("partition index")
    style.finish(ax)

    style.panel_titles(list(axes))
    _legend_below(fig, axes[0], ncol=2, bottom=0.42)
    fig.savefig(FIGS / "figR14_pr_spread.png")
    plt.close(fig)
    print("figR14_pr_spread.png")


if __name__ == "__main__":
    _main()
