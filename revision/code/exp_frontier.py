"""The operating curve of the pairwise and the latent formulation.

Figure 7 of the submitted manuscript was drawn from a threshold sweep that
predates the deposition model, so it described a different experiment from the
main table. The sweep is repeated here under the deposition model, with the
excavation record read as graded evidence, so that the curve and the table
describe the same corpora.

Each point is one value of the join threshold. No threshold is tuned: the whole
curve is the result.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import pandas as pd

import common
import exp_main
from common import RESULTS, assemble, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="P2,P3,P4")
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS / "frontier_rev.csv"))
    a = ap.parse_args()

    model = common.load_matcher()
    dep = synth.DepositionSpec(**exp_main.DEP)
    grid = assemble.theta_grid(assemble.WIDE_P_GRID)
    rows = []
    for state in a.states.split(","):
        cal, lr, radius = exp_main.context(model, state, a.cal_slips, dep,
                                           a.workers)
        base = assemble.AssemblyConfig(time_limit=a.time_limit,
                                       workers=a.workers, strat_lr=lr,
                                       strat_radius=radius)
        for seed in range(a.seeds):
            spec = common.make_spec(common.EVAL_SEED0 + seed, a.slips, state,
                                    dep)
            inst, meta, S, slip_of = common.prep(model, spec)
            W = assemble.log_odds(S, cal)
            n = len(meta)
            for name in ("full", "latent"):
                kw = exp_main.MAIN[name]
                for th in grid:
                    cfg = replace(base, theta=th, strat_mode="soft", **kw)
                    t0 = time.time()
                    r = assemble.solve_cpsat(
                        assemble.build_problem(meta, W, cfg),
                        workers=a.workers)
                    e = common.score(n, r["joins"], inst["joins"], slip_of)
                    rows.append(dict(state=state, seed=seed, method=name,
                                     theta=th, n_frag=n, status=r["status"],
                                     solve_s=time.time() - t0, **e))
                pd.DataFrame(rows).to_csv(a.out, index=False)
                print(f"  [{state}] seed {seed} {name}: {len(grid)} thresholds",
                      flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
