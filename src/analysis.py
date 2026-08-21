"""Tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

METRICS = ["precision", "recall", "f1", "ari", "exact_slip",
           "clean_fragments", "same_slip_precision", "cross_slip_rate"]


def _agg(d, by=("state", "method")):
    g = d.groupby(list(by))
    m = g[METRICS + ["solve_s", "top1", "top50"]].mean()
    s = g[METRICS].std()
    n = g.size().rename("n_seeds")
    out = m.join(s, rsuffix="_sd").join(n)
    return out.reset_index()


def table_main(d, out):
    t = _agg(d)
    t = t[t.method.isin(style.METHOD_ORDER + ["expert_top50", "expert_top10"])]
    t["method"] = pd.Categorical(
        t.method, style.METHOD_ORDER + ["expert_top10", "expert_top50"], True)
    t = t.sort_values(["state", "method"])
    t.to_csv(out, index=False)
    return t


def fig_main(d, out):
    """Headline: what each constraint layer buys, across the ladder."""
    style.apply()
    states = [s for s in ["P1", "P2", "P3", "P4", "P5"] if s in set(d.state)]
    methods = [m for m in style.METHOD_ORDER if m in set(d.method)]
    panels = [("f1", "Join F1"), ("ari", "Slip partition (ARI)"),
              ("exact_slip", "Slips recovered exactly"),
              ("cross_slip_rate", "Cross-slip joins (error rate)")]

    s = style.scale_for(13.6, key=Path(out).stem)

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.3))
    a = _agg(d)
    for ax, (met, lab) in zip(axes, panels):
        for k, mth in enumerate(methods):
            sub = a[a.method == mth].set_index("state").reindex(states)
            y = sub[met].values
            e = sub.get(met + "_sd", pd.Series(np.zeros(len(states)))).values
            ax.errorbar(range(len(states)), y, yerr=e,
                        color=style.METHOD_COLORS.get(mth, style.MUTED),
                        marker=style.MARKERS[k % len(style.MARKERS)],
                        capsize=2, lw=style.METHOD_LW.get(mth, 1.5),
                        ms=4.5 if mth == "latent" else 3.8, zorder=6 if mth == "latent" else 4,
                        label=style.METHOD_LABELS.get(mth, mth))
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states)
        ax.set_xlabel("Preservation state")
        ax.set_title(lab)
        style.finish(ax)
    style.panel_titles(axes)
    axes[0].set_ylim(0, 1)
    axes[-1].legend(loc="upper left", ncol=1)
    fig.suptitle("Global assembly versus ranking-based rejoining, "
                 "across corpus preservation", y=1.04, fontsize=11 * s,
                 fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_substitution(d, out):
    """Reconstruction quality against the matcher accuracy that produced it.

    The x-axis is the matcher's own measured Top-1, so the curves can be read
    against any published pairwise matcher rather than against our generator's
    parameters.
    """
    style.apply()
    methods = [m for m in style.METHOD_ORDER if m in set(d.method)]
    s = style.scale_for(8.4, key=Path(out).stem)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, met, lab in [(axes[0], "ari", "Slip partition (ARI)"),
                         (axes[1], "exact_slip", "Slips recovered exactly")]:
        for k, mth in enumerate(methods):
            sub = d[d.method == mth].groupby("state").agg(
                x=("top1", "mean"), y=(met, "mean"), e=(met, "std")).sort_values("x")
            ax.errorbar(sub.x, sub.y, yerr=sub.e,
                        color=style.METHOD_COLORS.get(mth, style.MUTED),
                        marker=style.MARKERS[k % len(style.MARKERS)],
                        capsize=2, lw=style.METHOD_LW.get(mth, 1.5),
                        ms=4.5 if mth == "latent" else 3.8, zorder=6 if mth == "latent" else 4,
                        label=style.METHOD_LABELS.get(mth, mth))
        ax.set_xlabel("Pairwise matcher Top-1 accuracy")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        style.finish(ax)
    style.panel_titles(axes)
    style.legend_below(fig, axes[1], ncol=3, bottom=0.34)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(d, out, state=None):
    """Leave-one-out over the constraint set at the reference operating point."""
    style.apply()
    abl = ["latent", "latent_no_notch", "latent_no_hand", "full",
           "no_length", "no_notch", "no_width", "no_strat", "no_hand"]
    sub = d[d.method.isin(abl)]
    if sub.empty:
        return False
    if state is None:
        state = sorted(set(sub.state))[-1]
    sub = sub[sub.state == state]
    labels = {"latent": "Latent-slip (proposed)",
              "latent_no_notch": "  − binding notches",
              "latent_no_hand": "  − scribal hand",
              "full": "Pairwise constraints",
              "no_length": "  − slip length", "no_notch": "  − binding notches",
              "no_width": "  − morphometry", "no_strat": "  − stratigraphy",
              "no_hand": "  − scribal hand"}
    a = sub.groupby("method")[["f1", "ari", "exact_slip",
                               "cross_slip_rate"]].agg(["mean", "std"])
    order = [m for m in abl if m in a.index]
    heads = {"latent", "full"}
    s = style.scale_for(13.6, key=Path(out).stem)
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4))
    for ax, met, lab in zip(axes, ["f1", "ari", "exact_slip", "cross_slip_rate"],
                            ["Join F1", "Slip partition (ARI)",
                             "Slips recovered exactly",
                             "Cross-slip joins (error rate)"]):
        vals = [a.loc[m, (met, "mean")] for m in order]
        errs = [a.loc[m, (met, "std")] for m in order]
        cols = [style.ACCENT if m == "latent" else
                style.RAMP[3] if m == "full" else
                ("#e3a894" if m.startswith("latent") else style.RAMP[1])
                for m in order]
        ax.barh(range(len(order)), vals, xerr=errs, color=cols, height=0.66,
                error_kw=dict(lw=0.8))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[m] for m in order],
                           fontweight=["bold" if m in heads else "normal"
                                       for m in order][0] if False else None)
        for t, m in zip(ax.get_yticklabels(), order):
            if m in heads:
                t.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        style.finish(ax, grid_axis="x")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    style.panel_titles(axes)
    fig.suptitle(f"Contribution of each constraint ({state})", y=1.04,
                 fontsize=11 * s, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_scaling(d, out):
    style.apply()
    sub = d[d.method.isin(["matching", "latent"])]
    if sub.n_frag.nunique() < 2:
        return False
    s = style.scale_for(11.0, key=Path(out).stem)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for k, mth in enumerate(["matching", "latent"]):
        g = sub[sub.method == mth].groupby("n_slips")
        x = sorted(g.groups)
        xf = [g["n_frag"].mean()[k] for k in x]
        for ax, met, lab in zip(axes, ["ari", "exact_slip", "solve_s"],
                                ["Slip partition (ARI)",
                                 "Slips recovered exactly",
                                 "Solve time (s)"]):
            m = g[met].mean().reindex(x)
            e = g[met].std().reindex(x)
            ax.errorbar(xf, m, yerr=e, color=style.METHOD_COLORS[mth],
                        marker=style.MARKERS[k], capsize=2,
                        lw=style.METHOD_LW.get(mth, 1.6),
                        ms=4.6 if mth == "latent" else 4.0,
                        label=style.METHOD_LABELS[mth])
            ax.set_xlabel("Fragments in corpus")
            ax.set_ylabel(lab)
            style.finish(ax)
    axes[2].set_yscale("log")
    style.panel_titles(axes)
    axes[0].legend(loc="lower left")
    fig.suptitle("Scaling with corpus size", y=1.04, fontsize=11 * s,
                 fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_effort(d, out, state=None):
    """Quality against the human effort each workflow costs.

    Ranking workflows are charged the candidate pairs a specialist must
    inspect; the optimisation is charged nothing, because it returns a
    reconstruction rather than a list.
    """
    style.apply()
    if state is None:
        state = sorted(set(d.state))[-1]
    sub = d[d.state == state]
    s = style.scale_for(6.4, key=Path(out).stem)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    exp = sub[sub.method.str.startswith("expert_top")]
    xmax = 1.0
    if not exp.empty:
        g = exp.groupby("method").agg(x=("review_pairs", "mean"),
                                      y=("ari", "mean"), e=("ari", "std"))
        g["k"] = [int(m.split("top")[1]) for m in g.index]
        g = g.sort_values("k")
        xmax = g.x.max()
        ax.errorbar(g.x, g.y, yerr=g.e, color=style.ORANGE, marker="s",
                    capsize=2, label="Expert review of ranked list (oracle)")
        for _, r in g.iterrows():
            # Labels sit to the left of their point so they stay clear of the
            # reference labels on the right-hand side.
            ax.annotate(f"top-{int(r.k)}", (r.x, r.y), textcoords="offset points",
                        xytext=(-6, -11), ha="right", fontsize=7 * s,
                        color=style.INK_SECONDARY)
    ax.set_xscale("log")
    # Room to the right of the last oracle point for the reference labels, so
    # that they cannot collide with the top-k annotations.
    ax.set_xlim(right=xmax * 26)
    for mth, col in [("latent", style.ACCENT), ("full", style.BLUE),
                     ("matching", style.MUTED)]:
        s2 = sub[sub.method == mth]
        if s2.empty:
            continue
        v = s2.ari.mean()
        ax.axhline(v, color=col, ls="--", lw=1.3)
        ax.annotate(style.METHOD_LABELS[mth] + ", no review",
                    (xmax * 22, v), fontsize=style.PRINTED["tick"] * s,
                    color=col, ha="right", va="top")
    ax.set_xlabel("candidate pairs a specialist must inspect")
    ax.set_ylabel("slip partition index")
    style.finish(ax)
    ax.legend(loc="upper left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_frontier(d, out):
    """Does one formulation dominate the other's whole operating frontier?

    Each point is one threshold; the curve a formulation traces as the
    threshold varies is its frontier.  Up and to the left is better: more
    reconstruction, fewer of the joins that fuse two slips.
    """
    style.apply()
    states = sorted(set(d.state))
    s = style.scale_for(4.1 * len(states, key=Path(out).stem))
    fig, axes = plt.subplots(1, len(states), figsize=(4.1 * len(states), 3.5),
                             squeeze=False)
    for ax, st in zip(axes[0], states):
        sub = d[d.state == st]
        for k, mth in enumerate(["full", "latent"]):
            s2 = sub[sub.method == mth]
            if s2.empty:
                continue
            g = s2.groupby("theta").agg(x=("cross_slip_rate", "mean"),
                                        y=("ari", "mean")).sort_values("x")
            ax.plot(g.x, g.y, color=style.METHOD_COLORS[mth],
                    marker=style.MARKERS[k], lw=style.METHOD_LW.get(mth, 1.6),
                    ms=4.6 if mth == "latent" else 4.0,
                    zorder=6 if mth == "latent" else 4,
                    label=style.METHOD_LABELS[mth])
        ax.set_xlabel("Cross-slip join rate")
        if ax is axes[0][0]:
            ax.set_ylabel("slip partition index")
        style.finish(ax, grid_axis="both")
    style.panel_titles(axes[0])
    style.legend_below(fig, axes[0][0], ncol=2, bottom=0.40)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def table_exchange(d, out, ref="matching", met="ari", proposed="latent"):
    """How much matcher accuracy the constraint set is worth.

    For each preservation state we ask what Top-1 accuracy the reference method
    would need in order to reach the quality the constrained model attains at
    the accuracy it actually had.  Where the reference cannot reach it at any
    accuracy on the ladder, the constraints buy something no pairwise
    improvement can.
    """
    g = d.groupby(["method", "state"]).agg(
        x=("top1", "mean"), y=(met, "mean")).reset_index()
    b = g[g.method == ref].sort_values("x")
    rows = []
    for _, r in g[g.method == proposed].sort_values("x").iterrows():
        if r.y <= b.y.max():
            need = float(np.interp(r.y, b.y.values, b.x.values))
            rows.append(dict(state=r.state, top1=r.x, proposed=r.y,
                             equivalent_top1=need, accuracy_points=need - r.x,
                             unreachable=False))
        else:
            rows.append(dict(state=r.state, top1=r.x, proposed=r.y,
                             equivalent_top1=np.nan, accuracy_points=np.nan,
                             unreachable=True))
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs_main.csv")
    ap.add_argument("--ablation", default="results/runs_ablation.csv")
    ap.add_argument("--scaling", default="results/runs_scaling.csv")
    ap.add_argument("--frontier", default="results/frontier.csv")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--tables", default="results")
    a = ap.parse_args()

    Path(a.figs).mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.runs)
    t = table_main(d, f"{a.tables}/table_main.csv")
    print(t[["state", "method", "f1", "ari", "exact_slip",
             "same_slip_precision", "cross_slip_rate"]].round(3).to_string(index=False))

    ex = table_exchange(d, f"{a.tables}/table_exchange.csv")
    print("\naccuracy-equivalent of the constraint set:")
    print(ex.round(3).to_string(index=False))

    fig_main(d, f"{a.figs}/fig2_main.png")
    fig_substitution(d, f"{a.figs}/fig3_substitution.png")
    fig_effort(d, f"{a.figs}/fig6_effort.png")
    print("wrote fig2, fig3, fig6")

    if Path(a.frontier).exists():
        df = pd.read_csv(a.frontier)
        if fig_frontier(df, f"{a.figs}/fig5_frontier.png"):
            print("wrote fig5_frontier")

    if Path(a.ablation).exists():
        da = pd.read_csv(a.ablation)
        if fig_ablation(da, f"{a.figs}/fig4_ablation.png"):
            _agg(da).to_csv(f"{a.tables}/table_ablation.csv", index=False)
            print("wrote fig4 + table_ablation")
    if Path(a.scaling).exists():
        ds = pd.read_csv(a.scaling)
        if fig_scaling(ds, f"{a.figs}/fig8_scaling.png"):
            _agg(ds, by=("n_frag", "method")).to_csv(
                f"{a.tables}/table_scaling.csv", index=False)
            print("wrote fig5 + table_scaling")


if __name__ == "__main__":
    main()


def fig_sobol(path, out, metric="ari"):
    """Total-order Sobol indices: what actually drives the reconstruction."""
    style.apply()
    d = pd.read_csv(path)
    d = d[d.metric == metric].sort_values("ST")
    nice = {"face_texture_noise": "face staining", "face_loss_mm": "face material loss",
            "pullout_mean_mm": "break relief", "p_gap": "material lost at break",
            "p_fragment_lost": "fragments never recovered",
            "breaks_per_slip": "breaks per slip", "hand_accuracy": "hand attribution accuracy",
            "notch_sigma_mm": "notch measurement noise",
            "notch_tol_mm": "notch tolerance (method)", "width_tol_mm": "width tolerance (method)"}
    cols = [style.ORANGE if p.endswith("_tol_mm") else style.BLUE for p in d.param]
    s = style.scale_for(6.0, key=Path(out).stem)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.barh(range(len(d)), d.ST, xerr=d.ST_conf, color=cols, height=0.66,
            error_kw=dict(lw=0.8))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([nice.get(p, p) for p in d.param])
    ax.set_xlabel("Total-order Sobol index $S_T$ (slip partition ARI)")
    ax.set_title("Corpus condition drives the result, not the method's tolerances")
    style.finish(ax, grid_axis="x")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True
