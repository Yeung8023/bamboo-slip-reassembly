"""E3. What "feasible but not proven optimal" actually costs.

The submitted manuscript reported that above roughly 2500 fragments the solver
returns a feasible rather than a proven optimal solution, so those results were
lower bounds. Reviewer 2 is right that this leaves the ceiling of the method
unstated.

Three things are measured here, and together they answer it.

First, the gap is not unknown. The solver returns an upper bound on the
objective alongside its incumbent, so the distance to the optimum is reported
rather than guessed.

Second, the same instances are solved again with a much larger budget. Where
that proves optimality, the reconstruction it returns is compared against the
one found within the original budget. This is the number that matters to the
reviewer's concern: if the two reconstructions score the same, then reporting a
feasible solution understates nothing, and the gains in the manuscript are not
artefacts of an unfinished search.

Third, the corpus is solved component by component. Every variable of the model
belongs to one fragment and the only coupling is a candidate arc, so the
solution assembled from the components is the exact optimum of the whole
corpus. How much that helps depends on how far the arc graph actually
separates, which is measured here rather than assumed.

Timings are taken on an otherwise idle machine and replace the ones in the
submitted manuscript, which were measured while it was carrying other work.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common
from common import assemble, synth

STATE = "P4"


def _row(tag, solver, budget, n_slips, seed, n, prob, r, wall, e, ref=None):
    return dict(tag=tag, solver=solver, budget_s=budget, n_slips=n_slips,
                seed=seed, n_frag=n, n_arcs=len(prob["arcs"]), solve_s=wall,
                status=r["status"], objective=r.get("objective", np.nan),
                bound=r.get("bound", np.nan), gap=r.get("gap", np.nan),
                n_components=r.get("n_components", np.nan),
                max_component=r.get("max_component", np.nan),
                obj_vs_ref=(np.nan if ref is None else
                            r.get("objective", np.nan) - ref), **e)


def measure(model, n_slips, seed, kw, cal, lr, radius, short, long_, workers,
            dep):
    spec = common.make_spec(common.EVAL_SEED0 + seed, n_slips, STATE, dep=dep)
    inst, meta, S, slip_of = common.prep(model, spec)
    W = assemble.log_odds(S, cal)
    base = assemble.AssemblyConfig(workers=workers, strat_lr=lr,
                                   strat_radius=radius, **kw)
    prob = assemble.build_problem(meta, W, base)
    n = len(meta)
    rows = []

    def score(r):
        return common.score(n, r["joins"], inst["joins"], slip_of)

    t0 = time.time()
    rs = assemble.solve_cpsat(prob, time_limit=short, workers=workers)
    ts = time.time() - t0
    es = score(rs)
    rows.append(_row("scale", "monolithic", short, n_slips, seed, n, prob, rs,
                     ts, es))

    t0 = time.time()
    rl = assemble.solve_cpsat(prob, time_limit=long_, workers=workers)
    tl = time.time() - t0
    el = score(rl)
    rows.append(_row("scale", "monolithic", long_, n_slips, seed, n, prob, rl,
                     tl, el, ref=rs.get("objective")))

    rd = assemble.solve_decomposed(prob, time_limit=long_, workers=workers)
    ed = score(rd)
    rows.append(_row("scale", "decomposed", long_, n_slips, seed, n, prob, rd,
                     rd["wall"], ed, ref=rl.get("objective")))

    same = (rl["status"] == "OPTIMAL" and rd["status"] == "OPTIMAL"
            and abs(rl["objective"] - rd["objective"]) < 1e-3)
    print(f"  {n} frags seed {seed}: "
          f"{short:.0f}s {rs['status']} gap {rs.get('gap', np.nan):.4f} "
          f"ARI {es['ari']:.3f} | "
          f"{long_:.0f}s {rl['status']} {tl:.0f}s "
          f"gap {rl.get('gap', np.nan):.4f} ARI {el['ari']:.3f} | "
          f"decomposed {rd['status']} {rd['wall']:.0f}s "
          f"{rd['n_components']} comps (max {rd['max_component']}) "
          f"ARI {ed['ari']:.3f} | exact match {same}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="150,400,900,1800,2600")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--short", type=float, default=240.0,
                    help="the budget the submitted manuscript used")
    ap.add_argument("--long", type=float, default=2400.0,
                    help="a budget large enough to prove optimality")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--case", action="store_true",
                    help="also run the Shuihudi scale corpus")
    ap.add_argument("--out", default=str(common.RESULTS / "decomposition.csv"))
    a = ap.parse_args()

    model = common.load_matcher()
    dep = synth.DepositionSpec(model="spatial", disperse_sigma_mm=30.0,
                               block_mm=250.0)
    cal, _ = common.calibrate(model, STATE, a.cal_slips, dep=dep)
    cspec = common.make_spec(common.CAL_SEED0 + 50, a.cal_slips, STATE, dep)
    cinst, cmeta, _, _ = common.prep(model, cspec)
    radius = common.choose_radius(cmeta, cinst["joins"])
    lr = assemble.fit_context_lr(cmeta, cinst["joins"], radius=radius)

    kw = dict(use_latent=True, use_strat=True, strat_mode="soft",
              notch_order=True)
    cfg = common.cfg_for(
        assemble.AssemblyConfig(time_limit=12.0, workers=a.workers,
                                strat_lr=lr, strat_radius=radius), **kw)
    theta, edge = common.tune(model, STATE, a.cal_slips, cfg, dep=dep,
                              cal=cal, workers=a.workers)
    print(f"theta {theta:+.2f} radius {radius}"
          f"{' (at grid endpoint)' if edge else ''}", flush=True)
    kw["theta"] = theta

    rows = []
    for n_slips in [int(x) for x in a.sizes.split(",")]:
        for seed in range(a.seeds):
            rows += measure(model, n_slips, seed, kw, cal, lr, radius,
                            a.short, a.long, a.workers, dep)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    if a.case:
        # The Shuihudi configuration itself, which the submitted manuscript
        # could only report as a lower bound.
        print("Shuihudi scale corpus", flush=True)
        rows += measure(model, 1155, 0, kw, cal, lr, radius, a.short,
                        max(a.long, 3600.0), a.workers, dep)
        for r in rows[-3:]:
            r["tag"] = "shuihudi"
        pd.DataFrame(rows).to_csv(a.out, index=False)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
