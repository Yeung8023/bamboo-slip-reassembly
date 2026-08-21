"""E7. The headline results, rerun under the deposition model.

The submitted main table, ablation and latent-versus-pairwise comparison were
all produced with one excavation unit drawn per slip, never missing and never
wrong. Section 4.4 of the revision shows that record is not one any excavation
delivers, and that reading it as a gate is harmful once fragments disperse. It
would be incoherent to demonstrate that and then leave the headline numbers
resting on it, so everything is rerun here with fragments deposited, dispersed
and recorded as Section 3.3 describes, and with the excavation record read as
graded evidence.

Three things are fixed at the same time, all of which were defects of the
submitted protocol rather than of the model.

  * Every method is tuned on calibration corpora, including the baselines. In
    the submitted case study the bipartite baseline inherited the threshold
    chosen for the proposed model.
  * Thresholds are tuned on the measure they are reported on. Tuning on join
    F1 and reporting the partition index put methods at different points of
    their own operating curves.
  * All five preservation states are reported for the latent against pairwise
    comparison, including the states where the latent formulation does not win.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import common          # noqa: F401  (puts ../src on sys.path)
import baselines
import matcher as M
from common import assemble, synth

TUNE_LIMIT = 12.0

# The reference deposition: slips in rolls on a chamber floor, fragments
# displaced by 30 mm, recorded on a 250 mm grid.
DEP = dict(model="spatial", disperse_sigma_mm=30.0, block_mm=250.0)

# Each method is a constraint subset of one model, as in the submitted work.
# What changes is that the excavation record now enters as graded evidence
# rather than as a gate on admissible joins.
MAIN = {
    "matching": dict(use_uniqueness=True, use_length=False, use_width=False,
                     use_notch=False, use_hand=False, use_strat=False),
    "morph":    dict(use_uniqueness=True, use_length=False, use_width=True,
                     use_notch=False, use_hand=False, use_strat=True),
    "length":   dict(use_uniqueness=True, use_length=True, use_width=True,
                     use_notch=False, use_hand=False, use_strat=True),
    "full":     dict(use_uniqueness=True, use_length=True, use_width=True,
                     use_notch=True, use_hand=True, use_strat=True),
    "latent":   dict(use_uniqueness=True, use_length=True, use_width=True,
                     use_notch=True, use_hand=True, use_strat=True,
                     use_latent=True, notch_order=True),
}

ABLATION = {
    "latent":     MAIN["latent"],
    "no_length":  dict(MAIN["latent"], use_length=False, use_notch=False),
    "no_notch":   dict(MAIN["latent"], use_notch=False),
    "no_width":   dict(MAIN["latent"], use_width=False),
    "no_strat":   dict(MAIN["latent"], use_strat=False),
    "no_hand":    dict(MAIN["latent"], use_hand=False),
    "full":       MAIN["full"],
    "full_no_notch": dict(MAIN["full"], use_notch=False),
    "full_no_hand":  dict(MAIN["full"], use_hand=False),
}


def context(model, state, cal_slips, dep, workers):
    """Score calibration, contextual log-odds and radius, per preservation state."""
    cal, _ = common.calibrate(model, state, cal_slips, dep=dep)
    spec = common.make_spec(common.CAL_SEED0 + 50, cal_slips, state, dep)
    inst, meta, _, _ = common.prep(model, spec)
    radius = common.choose_radius(meta, inst["joins"])
    lr = assemble.fit_context_lr(meta, inst["joins"], radius=radius)
    return cal, lr, radius


def run(model, state, methods, seeds, n_slips, cal_slips, time_limit, workers,
        dep, out, tag, objective="ari"):
    rows = []
    cal, lr, radius = context(model, state, cal_slips, dep, workers)
    base = assemble.AssemblyConfig(time_limit=time_limit, workers=workers,
                                   strat_lr=lr, strat_radius=radius)
    corpus = common.tune_corpus(model, state, cal_slips, cal, dep=dep)

    thetas, edges = {}, {}
    for name, kw in methods.items():
        cfg = replace(base, time_limit=TUNE_LIMIT, strat_mode="soft", **kw)
        # tuned on the partition index, which favours lower thresholds than
        # join F1 did, so the grid is widened downward to keep the optimum
        # inside the searched range for every method
        grid = assemble.theta_grid(assemble.WIDE_P_GRID)
        th, v = assemble.tune_theta(*corpus, cfg, grid=grid, workers=workers,
                                    objective=objective)
        thetas[name] = th
        edges[name] = bool(abs(th - grid[0]) < 1e-9 or abs(th - grid[-1]) < 1e-9)
    print(f"[{state}] " + " ".join(f"{k} {v:+.2f}{'*' if edges[k] else ''}"
                                   for k, v in thetas.items()), flush=True)

    for seed in seeds:
        spec = common.make_spec(common.EVAL_SEED0 + seed, n_slips, state, dep)
        inst, meta, S, slip_of = common.prep(model, spec)
        W = assemble.log_odds(S, cal)
        n = len(meta)
        acc = M.topk_accuracy(S, inst["joins"])
        comm = dict(tag=tag, state=state, seed=seed, n_frag=n,
                    n_true_joins=len(inst["joins"]), top1=acc["top1"],
                    top5=acc["top5"], top50=acc["top50"],
                    coherence=common.coherence(meta, inst["joins"]))

        if tag == "main":
            for h, fn in (("top1", baselines.top1),
                          ("mutual", baselines.mutual_best)):
                e = common.score(n, fn(W), inst["joins"], slip_of)
                rows.append({**comm, "method": h, "theta": np.nan,
                             "solve_s": 0.0, "status": "-", **e})

        for name, kw in methods.items():
            cfg = replace(base, theta=thetas[name], strat_mode="soft", **kw)
            t0 = time.time()
            prob = assemble.build_problem(meta, W, cfg)
            r = assemble.solve_cpsat(prob, workers=workers)
            e = common.score(n, r["joins"], inst["joins"], slip_of)
            rows.append({**comm, "method": name, "theta": thetas[name],
                         "theta_at_edge": edges[name],
                         "solve_s": time.time() - t0, "status": r["status"],
                         "n_arcs": len(prob["arcs"]), **e})
        got = {x["method"]: x["ari"] for x in rows[-len(methods):]}
        print(f"  [{state}] seed {seed} " +
              " ".join(f"{k} {v:.3f}" for k, v in got.items()), flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="main", help="main | ablation")
    ap.add_argument("--states", default="P1,P2,P3,P4,P5")
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    model = common.load_matcher()
    dep = synth.DepositionSpec(**DEP)
    methods = MAIN if a.what == "main" else ABLATION
    out = a.out or str(common.RESULTS / f"rerun_{a.what}.csv")
    rows = []
    for state in a.states.split(","):
        rows += run(model, state, methods, list(range(a.seeds)), a.slips,
                    a.cal_slips, a.time_limit, a.workers, dep, out, a.what)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
