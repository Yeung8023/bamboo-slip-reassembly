"""Qualitative figures: the benchmark, and the Yunmeng case demonstration."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import assemble
import matcher as M
import metrics
import preservation as pres
import style
import synth

# WisePanda (Nature Communications 2026) reports Top-50 accuracy rising from
# 36% to 52% among more than one thousand candidate fragments.
WISEPANDA_TOP50 = 0.52
WISEPANDA_BASELINE_TOP50 = 0.36


def _descs(inst):
    F = inst["frags"]
    return dict(bot_prof=np.array([f["bot_prof"] for f in F]),
                bot_patch=np.array([f["bot_patch"] for f in F]),
                top_prof=np.array([f["top_prof"] for f in F]),
                top_patch=np.array([f["top_patch"] for f in F]))


def fig_benchmark(model, out, seed=11):
    """What the benchmark looks like, and why ranking alone cannot finish."""
    style.apply()
    fig = plt.figure(figsize=(14.4, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1.0, 1.15, 1.15],
                          wspace=0.42, top=0.84)

    # (a) an intact slip, torn into fragments
    inst = synth.generate_instance(
        synth.InstanceSpec(n_slips=10, seed=seed, damage=pres.damage("P3")),
        render_gallery=80)
    gal = {g["frag_id"]: g for g in inst["gallery"]}
    bys = defaultdict(list)
    for f in inst["frags"]:
        bys[f["slip_id"]].append(f)
    sid = max(bys, key=lambda s: len(bys[s]))
    fl = sorted(bys[sid], key=lambda f: f["order"])[:4]

    sub = gs[0].subgridspec(1, len(fl), wspace=0.12)
    for i, f in enumerate(fl):
        ax = fig.add_subplot(sub[0, i])
        g = gal[f["frag_id"]]
        ax.imshow(np.dstack([g["img"]] * 3 + [g["mask"]]), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        tag = f"{f['height_mm']:.0f} mm"
        if f["bot_gap"] > 0.5:
            tag += f"\n{f['bot_gap']:.0f} mm lost"
        ax.set_xlabel(tag, fontsize=6.5, color=style.INK_SECONDARY,
                      fontweight="normal", labelpad=2)
    fig.text(0.075, 0.90, "a)  One slip, fractured and degraded", fontsize=9,
             fontweight="bold")

    # (b) the fracture signature: true partner against the best impostor
    ax = fig.add_subplot(gs[1])
    fm = {f["frag_id"]: f for f in inst["frags"]}
    a, b = None, None
    for x, y in inst["joins"]:
        if fm[x]["bot_prof"].std() > 1e-6:
            a, b = x, y
            break
    if a is not None:
        yy = np.arange(synth.CANON_W)
        ax.plot(fm[a]["bot_prof"], yy, color=style.BLUE, lw=1.6,
                label="fragment A, lower break")
        ax.plot(fm[b]["top_prof"], yy, color=style.ORANGE, lw=1.6, ls="--",
                label="fragment B, upper break (true join)")
        others = [i for i in fm if i not in (a, b)]
        imp = max(others, key=lambda i: np.corrcoef(
            fm[a]["bot_prof"], fm[i]["top_prof"])[0, 1]
            if fm[i]["top_prof"].std() > 1e-6 else -1)
        ax.plot(fm[imp]["top_prof"], yy, color=style.MUTED, lw=1.1, ls=":",
                label="best impostor in this corpus")
    ax.set_xlabel("Break offset (px)")
    ax.set_ylabel("Across slip width")
    ax.legend(fontsize=6.2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    ax.set_title("b)  Fracture signature", loc="left", fontsize=9, pad=8)
    style.finish(ax, grid_axis="both")

    # (c) matcher accuracy along the preservation ladder
    ax = fig.add_subplot(gs[2])
    ks = np.array([1, 5, 10, 50])
    for i, (k, lab, _) in enumerate(pres.LADDER):
        ins = synth.generate_instance(
            synth.InstanceSpec(n_slips=1800, seed=0, damage=pres.damage(k)))
        S = M.score_matrix(model, _descs(ins))
        acc = M.topk_accuracy(S, ins["joins"], ks=tuple(ks))
        ax.plot(ks, [acc[f"top{j}"] for j in ks],
                color=style.SEQ_BLUE[-(i + 2)], marker=style.MARKERS[i],
                lw=1.4, label=f"{k} {lab}")
    ax.axhline(WISEPANDA_TOP50, color=style.MAGENTA, ls="--", lw=1.1)
    ax.annotate("WisePanda, Top-50 = 0.52\n(>1000 candidates)",
                (1.15, WISEPANDA_TOP50), fontsize=6.3, color=style.MAGENTA,
                va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("k (candidates inspected)\n5131-fragment corpus")
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.2, loc="lower right")
    ax.set_title("c)  Matcher accuracy", loc="left", fontsize=9)
    style.finish(ax, grid_axis="both")

    # (d) candidates conflict: how many fragments claim the same partner
    ax = fig.add_subplot(gs[3])
    ins = synth.generate_instance(
        synth.InstanceSpec(n_slips=400, seed=0, damage=pres.damage("P4")))
    S = M.score_matrix(model, _descs(ins))
    for i, kk in enumerate((1, 5)):
        top = np.argpartition(-S, kth=kk, axis=1)[:, :kk].ravel()
        cnt = np.bincount(top, minlength=S.shape[0])
        vals, freq = np.unique(cnt, return_counts=True)
        ax.plot(vals, freq / freq.sum(), marker=style.MARKERS[i],
                color=[style.BLUE, style.ORANGE][i], lw=1.4,
                label=f"top-{kk} lists")
    ax.set_yscale("log")
    ax.set_xlabel("Times a fragment is claimed as partner")
    ax.set_ylabel("Fraction of fragments")
    ax.legend(fontsize=6.5)
    ax.set_title("d)  Candidates conflict", loc="left", fontsize=9, pad=8)
    style.finish(ax, grid_axis="both")

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_case(model, out, n_slips=1155, seed=4242, state="P4"):
    """Yunmeng demonstration on a corpus sized to the Shuihudi find.

    Tomb 11 at Shuihudi (Yunmeng, Hubei, excavated 1975) yielded roughly 1,155
    slips together with a number of small fragments.  The instance here is
    synthetic, but its corpus geometry -- slip count, lengths, widths and the
    three binding cords -- is set to that find, so the figure shows the method
    working at the scale and shape of a real Qin corpus.
    """
    style.apply()
    D = pres.damage(state)
    ci = synth.generate_instance(
        synth.InstanceSpec(n_slips=200, seed=900000, damage=D))
    cal = assemble.fit_calibration(M.score_matrix(model, _descs(ci)), ci["joins"])

    inst = synth.generate_instance(
        synth.InstanceSpec(n_slips=n_slips, seed=seed, damage=D))
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = M.score_matrix(model, _descs(inst))
    W = assemble.log_odds(S, cal)
    n = len(meta)
    slip_of = [f["slip_id"] for f in inst["frags"]]
    acc = M.topk_accuracy(S, inst["joins"])

    import baselines
    from dataclasses import replace
    # theta and the constraint set are those selected on P4 calibration corpora
    cfg = assemble.AssemblyConfig(time_limit=900, workers=10, theta=1.73,
                                  use_latent=True)
    prob = assemble.build_problem(meta, W, cfg)
    r = assemble.solve_cpsat(prob)
    e_full = metrics.evaluate(n, r["joins"], inst["joins"], slip_of)
    mo = baselines.matching_only(meta, W, cfg)
    e_match = metrics.evaluate(n, mo, inst["joins"], slip_of)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5),
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1.25]))

    ax = axes[0]
    lbl = ["Join F1", "Slip ARI", "Exact slips", "Cross-slip\nerror rate"]
    keys = ["f1", "ari", "exact_slip", "cross_slip_rate"]
    xx = np.arange(len(keys))
    ax.bar(xx - 0.2, [e_match[k] for k in keys], 0.38, color=style.ORANGE,
           label="Bipartite matching")
    ax.bar(xx + 0.2, [e_full[k] for k in keys], 0.38, color=style.BLUE,
           label="Latent-slip assembly")
    ax.set_xticks(xx)
    ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("a)  Reconstruction quality", loc="left", fontsize=9)
    style.finish(ax)

    ax = axes[1]
    true_c = metrics.ground_truth_chains(n, inst["joins"])
    pred_c = assemble.chains_from_joins(n, r["joins"])
    tl = np.bincount([len(c) for c in true_c], minlength=8)[1:8]
    pl = np.bincount([len(c) for c in pred_c], minlength=8)[1:8]
    xx = np.arange(1, 8)
    ax.bar(xx - 0.2, tl, 0.38, color=style.MUTED, label="ground truth")
    ax.bar(xx + 0.2, pl, 0.38, color=style.BLUE, label="reconstructed")
    ax.set_xlabel("Fragments per reconstructed slip")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.set_title("b)  Chain-length distribution", loc="left", fontsize=9)
    style.finish(ax)

    ax = axes[2]
    ax.axis("off")
    txt = (f"Shuihudi-scale corpus\n"
           f"  slips                {n_slips}\n"
           f"  fragments            {n}\n"
           f"  true conjoins        {len(inst['joins'])}\n"
           f"  preservation         {state} ({pres.LABELS[state]})\n\n"
           f"Pairwise matcher\n"
           f"  Top-1                {acc['top1']:.3f}\n"
           f"  Top-50               {acc['top50']:.3f}\n\n"
           f"Bipartite matching\n"
           f"  join F1              {e_match['f1']:.3f}\n"
           f"  slip ARI             {e_match['ari']:.3f}\n"
           f"  exact slips          {e_match['exact_slip']:.3f}\n"
           f"  cross-slip errors    {e_match['cross_slip_joins']}\n\n"
           f"Latent-slip assembly\n"
           f"  join F1              {e_full['f1']:.3f}\n"
           f"  slip ARI             {e_full['ari']:.3f}\n"
           f"  exact slips          {e_full['exact_slip']:.3f}\n"
           f"  cross-slip errors    {e_full['cross_slip_joins']}\n"
           f"  solver status        {r['status']}")
    ax.text(0, 1, txt, fontsize=7.4, family="monospace", va="top",
            color=style.INK)
    ax.set_title("c)  Case summary", loc="left", fontsize=9)

    fig.suptitle("Demonstration at the scale of the Shuihudi Qin corpus "
                 "(Yunmeng, Hubei)", y=1.03, fontsize=11, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return dict(n_frag=n, n_join=len(inst["joins"]), acc=acc,
                match=e_match, full=e_full, status=r["status"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--which", default="benchmark,case")
    a = ap.parse_args()
    Path(a.figs).mkdir(parents=True, exist_ok=True)
    model = M.load_model(a.model)
    w = a.which.split(",")
    if "benchmark" in w:
        fig_benchmark(model, f"{a.figs}/fig1_benchmark.png")
        print("wrote fig1_benchmark.png")
    if "case" in w:
        import json
        res = fig_case(model, f"{a.figs}/fig7_yunmeng_case.png")
        Path("results").mkdir(exist_ok=True)
        with open("results/case_yunmeng.json", "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print("wrote fig7_yunmeng_case.png", json.dumps(
            {k: (v if not isinstance(v, dict) else
                 {kk: round(float(vv), 3) for kk, vv in v.items()
                  if isinstance(vv, (int, float))})
             for k, v in res.items()}, indent=1, default=str)[:600])
