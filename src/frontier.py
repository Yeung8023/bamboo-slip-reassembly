"""Operating frontier: reconstruction quality against catastrophic-error rate.

A single threshold trades recall for precision, so no one operating point
settles whether one formulation is better than another.  Sweeping the threshold
traces each formulation's whole frontier, and the question becomes whether one
frontier dominates -- whether it delivers more reconstruction at every level of
the error a conservator actually has to undo.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import assemble
import matcher as M
import metrics
import preservation as pres
import run_experiments as RX


def sweep(model, state, n_slips, seeds, methods, grid, time_limit, workers):
    D = pres.damage(state)
    ci, _, cS, _ = RX.prep(model, RX.CAL_SEED0, 120, D)
    cal = assemble.fit_calibration(cS, ci["joins"])
    base = assemble.AssemblyConfig(time_limit=time_limit, workers=workers)

    rows = []
    for seed in range(seeds):
        inst, meta, S, slip_of = RX.prep(model, seed, n_slips, D)
        W = assemble.log_odds(S, cal)
        n = len(meta)
        acc = M.topk_accuracy(S, inst["joins"])
        for name in methods:
            for th in grid:
                cfg = replace(base, theta=float(th), **RX.METHODS[name])
                prob = assemble.build_problem(meta, W, cfg)
                r = assemble.solve_cpsat(prob)
                e = metrics.evaluate(n, r["joins"], inst["joins"], slip_of)
                rows.append(dict(state=state, seed=seed, method=name,
                                 theta=float(th), n_frag=n, top1=acc["top1"],
                                 status=r["status"], **e))
                print(f"  {state} s{seed} {name:8s} th{th:+5.2f} "
                      f"F1 {e['f1']:.3f} ARI {e['ari']:.3f} "
                      f"cross {e['cross_slip_rate']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="P3,P4,P5")
    ap.add_argument("--slips", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", default="full,latent")
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/frontier.csv")
    a = ap.parse_args()

    grid = assemble.theta_grid((0.20, 0.35, 0.50, 0.70, 0.85, 0.95, 0.99, 0.997))
    model = M.load_model(a.model)
    rows = []
    for st in a.states.split(","):
        rows += sweep(model, st, a.slips, a.seeds, a.methods.split(","),
                      grid, a.time_limit, a.workers)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
