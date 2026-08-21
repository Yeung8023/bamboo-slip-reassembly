"""Additional result figures.

These cover what a method paper has to show beyond a bar chart of means: that
the scores are calibrated, that the solver converges, how the methods trade
precision against recall, how the results are distributed over seeds, and what
the reconstructions actually look like when they succeed and when they fail.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import assemble
import baselines
import matcher as M
import metrics
import preservation as pres
import run_experiments as RX
import style
import synth


# --------------------------------------------------------------------------
# matcher validity: separation, calibration
# --------------------------------------------------------------------------

def fig_calibration(model, out, state="P4", n_slips=400, seed=0, n_neg=200000):
    """Is the score informative, and does Platt scaling make it a probability?

    A weight that is not calibrated makes the threshold meaningless, so this is
    a precondition for everything the optimisation does with the weights.
    """
    style.apply()
    D = pres.damage(state)
    ci, _, cS, _ = RX.prep(model, RX.CAL_SEED0, 150, D)
    cal = assemble.fit_calibration(cS, ci["joins"])
    inst, meta, S, _ = RX.prep(model, seed, n_slips, D)

    joins = np.asarray(inst["joins"]).reshape(-1, 2)
    pos = S[joins[:, 0], joins[:, 1]]
    rng = np.random.default_rng(0)
    n = S.shape[0]
    ii, jj = rng.integers(0, n, n_neg), rng.integers(0, n, n_neg)
    keep = ii != jj
    jset = set(map(tuple, joins.tolist()))
    m = np.array([(int(a), int(b)) not in jset
                  for a, b in zip(ii[keep], jj[keep])])
    neg = S[ii[keep][m], jj[keep][m]]

    s = style.scale_for(12.4)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    bins = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 60)
    ax.hist(neg, bins=bins, density=True, color=style.GREY_2, alpha=0.8,
            label="non-joining pairs")
    ax.hist(pos, bins=bins, density=True, color=style.RAMP[2], alpha=0.85,
            label="true joins")
    ax.set_xlabel("matcher score $s_{ij}$")
    ax.set_ylabel("density")
    ax.legend()
    style.finish(ax)

    ax = axes[1]
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    sc = np.concatenate([pos, neg])
    pr, rc, _ = precision_recall_curve(y, sc)
    fpr, tpr, _ = roc_curve(y, sc)
    ax.plot(rc, pr, color=style.RAMP[2], lw=2.0)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    style.finish(ax, grid_axis="both")
    ax.text(0.94, 0.90, f"ROC AUC {auc(fpr, tpr):.3f}", transform=ax.transAxes,
            ha="right", fontsize=style.PRINTED["tick"] * s, color=style.INK,
            fontweight="bold")

    ax = axes[2]
    # The Platt fit is made at balanced class ratio on purpose, so that it
    # returns a likelihood ratio and carries no corpus prior. Testing it must
    # therefore also be done at balanced ratio, which is what it claims to
    # model; the prior enters the method through the structure instead.
    rs = np.random.default_rng(1)
    negb = rs.choice(neg, size=min(len(neg), len(pos)), replace=False)
    scb = np.concatenate([pos, negb])
    yb = np.concatenate([np.ones(len(pos)), np.zeros(len(negb))])
    pb = 1.0 / (1.0 + np.exp(-(cal[0] * scb + cal[1])))
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pb, edges) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() < 15:
            continue
        xs.append(pb[sel].mean())
        ys.append(yb[sel].mean())
        ns.append(int(sel.sum()))
    ax.plot([0, 1], [0, 1], color=style.MUTED, ls="--", lw=1.1,
            label="perfect calibration")
    ax.plot(xs, ys, color=style.ACCENT, marker="o", lw=1.8, ms=5,
            label="observed")
    ece = float(np.sum(np.array(ns) * np.abs(np.array(xs) - np.array(ys)))
                / max(np.sum(ns), 1))
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.04, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=style.PRINTED["tick"] * s, color=style.INK,
            fontweight="bold")
    ax.legend(loc="lower right")
    style.finish(ax, grid_axis="both")

    style.panel_titles(axes)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# solver behaviour: convergence and cost
# --------------------------------------------------------------------------

def fig_convergence(model, out, state="P4", sizes=(150, 400, 900),
                    time_limit=240.0, seed=0):
    """How the incumbent and the bound meet, and how cost grows with size."""
    style.apply()
    D = pres.damage(state)
    ci, _, cS, _ = RX.prep(model, RX.CAL_SEED0, 150, D)
    cal = assemble.fit_calibration(cS, ci["joins"])
    base = assemble.AssemblyConfig(time_limit=time_limit, workers=8,
                                   theta=1.73, use_latent=True)

    traces, stats = [], []
    for ns in sizes:
        inst, meta, S, _ = RX.prep(model, seed, ns, D)
        W = assemble.log_odds(S, cal)
        prob = assemble.build_problem(meta, W, base)
        r = assemble.solve_cpsat(prob, trace=True)
        traces.append((len(meta), r))
        stats.append(dict(n=len(meta), arcs=len(prob["arcs"]),
                          wall=r["wall"], status=r["status"]))
        print(f"  n={len(meta)} arcs={len(prob['arcs'])} "
              f"{r['status']} {r['wall']:.1f}s")

    s = style.scale_for(12.6)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))

    ax = axes[0]
    for k, (n, r) in enumerate(traces):
        tr = np.array(r.get("trace", []))
        if len(tr) == 0:
            continue
        c = style.SEQ_BLUE[-(k + 2)]
        ax.step(tr[:, 0], tr[:, 1], where="post", color=c, lw=1.7,
                label=f"{n} fragments")
        ax.step(tr[:, 0], tr[:, 2], where="post", color=c, lw=1.0, ls="--",
                alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("wall clock time (s)")
    ax.set_ylabel("objective")
    style.finish(ax, grid_axis="both")

    ax = axes[1]
    for k, (n, r) in enumerate(traces):
        tr = np.array(r.get("trace", []))
        if len(tr) == 0:
            continue
        gap = (tr[:, 2] - tr[:, 1]) / np.maximum(np.abs(tr[:, 1]), 1e-9)
        ax.step(tr[:, 0], 100 * np.clip(gap, 0, None), where="post",
                color=style.SEQ_BLUE[-(k + 2)], lw=1.7, label=f"{n}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall clock time (s)")
    ax.set_ylabel("gap (%)")
    style.finish(ax, grid_axis="both")

    ax = axes[2]
    st = pd.DataFrame(stats)
    ax.plot(st.n, st.arcs, color=style.ORANGE, marker="s", lw=1.6,
            label="candidate arcs")
    ax.set_xlabel("fragments in corpus")
    ax.set_ylabel("candidate arcs", color=style.ORANGE)
    ax.set_xscale("log"); ax.set_yscale("log")
    lo = np.polyfit(np.log(st.n), np.log(st.arcs), 1)[0]
    ax.text(0.05, 0.92, f"arcs $\\propto n^{{{lo:.2f}}}$",
            transform=ax.transAxes, fontsize=style.PRINTED["tick"] * s,
            color=style.ORANGE)
    ax2 = ax.twinx()
    ax2.plot(st.n, st.wall, color=style.BLUE, marker="o", lw=1.6)
    ax2.set_ylabel("solve time (s)", color=style.BLUE)
    ax2.set_yscale("log")
    style.finish(ax, grid_axis="both")

    style.panel_titles(axes)
    style.legend_below(fig, axes[0], ncol=3, bottom=0.30)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# operating curves and distributions
# --------------------------------------------------------------------------

def fig_pr_and_spread(frontier, runs, out):
    """Precision against recall along the threshold, and the spread over seeds.

    Means alone hide whether a difference is reliable, so the right panel gives
    the distribution and a paired test rather than an error bar.
    """
    style.apply()
    d = pd.read_csv(frontier)
    r = pd.read_csv(runs)
    states = sorted(set(d.state))

    s = style.scale_for(3.5 * (len(states) + 1))
    fig, axes = plt.subplots(1, len(states) + 1,
                             figsize=(3.5 * (len(states) + 1), 3.4))
    for ax, st in zip(axes, states):
        sub = d[d.state == st]
        xs, ys = [], []
        for k, mth in enumerate(["full", "latent"]):
            g = sub[sub.method == mth].groupby("theta").agg(
                p=("precision", "mean"), rc=("recall", "mean")).sort_values("rc")
            ax.plot(g.rc, g.p, color=style.METHOD_COLORS[mth],
                    marker=style.MARKERS[k], ms=4.6,
                    lw=style.METHOD_LW.get(mth, 1.6),
                    zorder=6 if mth == "latent" else 4,
                    label=style.METHOD_LABELS[mth])
            xs += list(g.rc); ys += list(g.p)
        # zoom to the region the curves occupy, otherwise the difference that
        # the figure exists to show is a few pixels wide
        mx, Mx = min(xs), max(xs)
        my, My = min(ys), max(ys)
        px, py = 0.08 * (Mx - mx) + 0.01, 0.12 * (My - my) + 0.01
        ax.set_xlim(mx - px, Mx + px)
        ax.set_ylim(my - py, My + py)
        ax.set_xlabel("recall")
        if ax is axes[0]:
            ax.set_ylabel("precision")
        style.finish(ax, grid_axis="both")

    ax = axes[-1]
    keep = ["matching", "morph", "length", "full", "latent"]
    ref = r[r.state == "P4"]
    data = [ref[ref.method == m].ari.values for m in keep]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color=style.INK, lw=1.2))
    for patch, m in zip(bp["boxes"], keep):
        patch.set_facecolor(style.METHOD_COLORS.get(m, style.MUTED))
        patch.set_alpha(0.85)
        patch.set_edgecolor(style.INK)
    for k, m in enumerate(keep):
        v = ref[ref.method == m].ari.values
        ax.scatter(np.full(len(v), k + 1) + np.linspace(-0.12, 0.12, len(v)),
                   v, s=9, color=style.INK, zorder=5, alpha=0.7)
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels(["match", "+morph", "+len", "pairwise", "latent"],
                       fontsize=8)
    ax.set_ylabel("partition index")
    style.finish(ax)

    from scipy.stats import ttest_rel, wilcoxon
    a = ref[ref.method == "latent"].sort_values("seed").ari.values
    b = ref[ref.method == "full"].sort_values("seed").ari.values
    if len(a) == len(b) and len(a) > 2:
        t, pv = ttest_rel(a, b)
        try:
            _, pw = wilcoxon(a, b)
        except ValueError:
            pw = float("nan")
        ax.text(0.5, -0.20, f"latent vs pairwise over {len(a)} seeds\n"
                f"paired $t$: $p$={pv:.4f}   Wilcoxon: $p$={pw:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=style.PRINTED["tick"] * s, color=style.INK,
                linespacing=1.3)

    style.panel_titles(axes)
    style.legend_below(fig, axes[0], ncol=2, bottom=0.34)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# qualitative: what a success and a failure look like
# --------------------------------------------------------------------------

def fig_qualitative(model, out, state="P4", n_slips=60, seed=23):
    """Reconstructions next to ground truth, including the failure cases.

    Aggregate metrics say how often the method is right. They do not say what
    being wrong looks like, which is what a conservator needs to judge.
    """
    style.apply()
    D = pres.damage(state)
    ci, _, cS, _ = RX.prep(model, RX.CAL_SEED0, 150, D)
    cal = assemble.fit_calibration(cS, ci["joins"])

    inst = synth.generate_instance(
        synth.InstanceSpec(n_slips=n_slips, seed=seed, damage=D),
        render_gallery=10000)
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = M.score_matrix(model, RX.descs(inst))
    W = assemble.log_odds(S, cal)
    cfg = assemble.AssemblyConfig(time_limit=120, workers=8, theta=1.73,
                                  use_latent=True)
    r = assemble.solve_cpsat(assemble.build_problem(meta, W, cfg))
    n = len(meta)
    pred = assemble.chains_from_joins(n, r["joins"])
    true = metrics.ground_truth_chains(n, inst["joins"])
    tset = {tuple(c) for c in true if len(c) > 1}
    gal = {g["frag_id"]: g for g in inst["gallery"]}
    slip_of = {f["frag_id"]: f["slip_id"] for f in inst["frags"]}

    ok = [c for c in pred if len(c) >= 3 and tuple(c) in tset]
    partial = [c for c in pred if len(c) >= 2 and tuple(c) not in tset
               and len({slip_of[i] for i in c}) == 1]
    bad = [c for c in pred if len(c) >= 2 and len({slip_of[i] for i in c}) > 1]
    picks = ([(c, "correct") for c in ok[:3]] +
             [(c, "wrong order or gap") for c in partial[:2]] +
             [(c, "two slips fused") for c in bad[:2]])
    if not picks:
        print("no chains to show")
        return

    s = style.scale_for(1.5 * len(picks) + 1.2)

    fig, ax = plt.subplots(figsize=(1.5 * len(picks) + 1.2, 5.0))
    ax.set_xlim(-0.34, len(picks) + 0.10); ax.set_ylim(-0.06, 1.06)
    ax.axis("off")
    cols = {"correct": style.RAMP[3],
            "wrong order or gap": style.ACCENT_SOFT,
            "two slips fused": "#b3402c"}
    gap = 0.014
    for j, (c, kind) in enumerate(picks):
        hs = np.array([gal[i]["img"].shape[0] / synth.PX_PER_MM for i in c],
                      float)
        # every chain is drawn to the same total height, so the eye compares
        # composition and not the accident of how long each chain happens to be
        usable = 1.0 - gap * (len(c) - 1)
        hs = hs / hs.sum() * usable
        y = 1.0
        x0, x1 = j + 0.28, j + 0.70
        for fid, hh in zip(c, hs):
            g = gal[fid]
            ax.imshow(np.dstack([g["img"]] * 3 + [g["mask"]]),
                      extent=(x0, x1, y - hh, y), aspect="auto",
                      interpolation="bilinear", zorder=3)
            ax.text(x1 + 0.04, y - hh / 2, f"{slip_of[fid]}", fontsize=6.0 * s,
                    va="center", color=style.INK_SECONDARY)
            y -= hh + gap
        ax.plot([x0 - 0.09, x0 - 0.09], [0.0, 1.0], color=cols[kind], lw=3.4,
                solid_capstyle="butt", zorder=4)
        ax.text(x0 - 0.16, 0.5, kind, rotation=90, ha="right", va="center",
                fontsize=6.8 * s, color=cols[kind], fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--which", default="calib,conv,pr,qual")
    ap.add_argument("--figs", default="figs")
    a = ap.parse_args()
    Path(a.figs).mkdir(exist_ok=True)
    model = M.load_model(a.model)
    w = a.which.split(",")
    if "calib" in w:
        fig_calibration(model, f"{a.figs}/fig10_calibration.png")
    if "conv" in w:
        fig_convergence(model, f"{a.figs}/fig11_convergence.png")
    if "pr" in w:
        fig_pr_and_spread("results/frontier.csv", "results/runs_main.csv",
                          f"{a.figs}/fig12_pr_spread.png")
    if "qual" in w:
        fig_qualitative(model, f"{a.figs}/fig13_qualitative.png")
