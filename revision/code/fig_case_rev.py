"""Figure 9, drawn from the case as the manuscript reports it.

The submitted version of this figure ran its own experiment, with a unit drawn
once per slip and a threshold that was not tuned on the measure reported. Its
numbers therefore disagree with the case study in the text, which is rerun
under the deposition model. The figure is rebuilt here from the same
configuration as `exp_main.py`, and the metrics it draws are checked against
`rerun_case.csv` before anything is plotted.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

import common
import baselines
import exp_main
import matcher as M
from common import FIGS, RESULTS, assemble, synth

sys.path.insert(0, str(common.ROOT / "src"))
import style  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

style.apply()

STATE = "P4"
N_SLIPS = 1155


def chains(n, joins):
    """Fragment counts of the reconstructed chains."""
    nxt = {int(i): int(j) for i, j in joins}
    prv = {int(j): int(i) for i, j in joins}
    heads = [i for i in range(n) if i not in prv]
    out = []
    for h in heads:
        k, cur = 1, h
        while cur in nxt:
            cur = nxt[cur]; k += 1
        out.append(k)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    ref = pd.read_csv(RESULTS / "rerun_case.csv").set_index("method")
    model = common.load_matcher()
    dep = synth.DepositionSpec(**exp_main.DEP)
    cal, lr, radius = exp_main.context(model, STATE, a.cal_slips, dep, a.workers)
    base = assemble.AssemblyConfig(time_limit=a.time_limit, workers=a.workers,
                                   strat_lr=lr, strat_radius=radius)
    corpus = common.tune_corpus(model, STATE, a.cal_slips, cal, dep=dep)

    spec = common.make_spec(common.EVAL_SEED0 + 0, N_SLIPS, STATE, dep)
    inst, meta, S, slip_of = common.prep(model, spec)
    W = assemble.log_odds(S, cal)
    n = len(meta)
    got = {}
    for name in ("matching", "latent"):
        kw = exp_main.MAIN[name]
        cfg = replace(base, time_limit=exp_main.TUNE_LIMIT, strat_mode="soft", **kw)
        th, _ = assemble.tune_theta(*corpus, cfg,
                                    grid=assemble.theta_grid(assemble.WIDE_P_GRID),
                                    workers=a.workers, objective="ari")
        cfg = replace(base, theta=th, strat_mode="soft", **kw)
        r = assemble.solve_cpsat(assemble.build_problem(meta, W, cfg),
                                 workers=a.workers)
        e = common.score(n, r["joins"], inst["joins"], slip_of)
        got[name] = dict(joins=r["joins"], status=r["status"], **e)
        d = abs(e["ari"] - float(ref.loc[name, "ari"]))
        print(f"  {name}: ari {e['ari']:.3f} against {ref.loc[name,'ari']:.3f}"
              f"  (delta {d:.4f})", flush=True)
        if d > 0.02:
            print("    WARNING: does not reproduce rerun_case.csv", flush=True)

    # ---------------------------------------------------------------- figure
    s = style.scale_for(9.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4),
                             gridspec_kw=dict(wspace=0.34))

    # The bars are the values the text and Table 1 report. Above about 2500
    # fragments the solver returns a feasible rather than a proven optimal
    # solution, and repeated runs differ by about a point, so drawing this
    # panel from a fresh solve would put the figure a little away from the
    # numbers it illustrates.
    labs = ["join $F_1$", "partition\nindex", "slips\nintact", "fusing\njoins"]
    keys = ["f1", "ari", "exact_slip", "cross_slip_rate"]
    x = np.arange(len(keys))
    for k, (m, col, lab) in enumerate([
            ("matching", style.RAMP[0], "bipartite matching"),
            ("latent", style.ACCENT, "latent properties (this work)")]):
        axes[0].bar(x + (k - 0.5) * 0.38,
                    [float(ref.loc[m, q]) for q in keys],
                    width=0.38, color=col, label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labs)
    axes[0].set_ylabel("value")
    style.finish(axes[0])

    truth = chains(n, inst["joins"])
    rec = chains(n, got["latent"]["joins"])
    bins = np.arange(0.5, max(truth.max(), rec.max()) + 1.5)
    axes[1].hist([truth, rec], bins=bins, label=["ground truth", "reconstructed"],
                 color=[style.GREY_2, style.ACCENT])
    axes[1].set_xlabel("fragments per slip")
    axes[1].set_ylabel("count")
    axes[1].set_yscale("log")
    style.finish(axes[1])

    style.panel_titles(list(axes))
    style.legend_below(fig, axes[0], ncol=2, bottom=0.30)
    h2, l2 = axes[1].get_legend_handles_labels()
    axes[1].legend(h2, l2, loc="upper right")
    fig.savefig(FIGS / "figR11_case.png")
    plt.close(fig)
    print("figR11_case.png")

    # the chain lengths are kept so the figure can be redrawn without a
    # fifteen minute solve
    pd.DataFrame({"source": ["truth"] * len(truth) + ["reconstructed"] * len(rec),
                  "fragments": np.concatenate([truth, rec])}).to_csv(
        RESULTS / "case_chains.csv", index=False)
    pd.DataFrame([{**{k: got[m][k] for k in keys}, "method": m,
                   "status": got[m]["status"]} for m in got]).to_csv(
        RESULTS / "case_figure.csv", index=False)


if __name__ == "__main__":
    main()
