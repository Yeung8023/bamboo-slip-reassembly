"""E2b. The joint estimate of slip length across a roll.

Reviewer 1 suggested estimating slip length jointly across the slips of one
roll, so that the band of admissible heights around each binding cord narrows
from the spread of the whole corpus to the spread of one manuscript. This
script implements and tests that, and it took two attempts.

The obvious version pools the latent length over the fragments recorded in one
excavation square, on the argument that a square holds one roll. It does not.
A roll of 30 slips at a 9 mm pitch is 270 mm across and a square is 250 mm, so
a square routinely holds slips of two manuscripts, and forcing them to a single
length is worse than not pooling at all. The square is therefore given a small
number of latent lengths and each slip takes one of them, which is a mixture
over the manuscripts a square can contain rather than an assumption that it
contains one.

Reported as an operating curve rather than at a single threshold. A first pass
tuned each variant to its own best threshold and found that the notch
constraint halves the rate of joins that fuse two slips while lowering the
partition index, which is what moving along an operating curve looks like and
is not evidence of a better model. The question is whether the curve moves
outward, so each variant is swept over the whole threshold grid and the
trade-off between grouping quality and the damaging error is compared at
matched operating points, exactly as the latent formulation was compared
against the pairwise one in the submitted manuscript.

Two references: the same model with no notch constraint, and the same pooling
keyed on the true bundle, which is the upper bound available to an excavation
that records which bundle each fragment was lifted from.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common
from common import assemble, synth

STATE = "P4"
TUNE_LIMIT = 12.0

CASES = {
    "no_notch":    dict(notch=False, roll=False, key="unit", K=1),
    "notch":       dict(notch=True,  roll=False, key="unit", K=1),
    "roll_one":    dict(notch=True,  roll=True,  key="unit", K=1),
    "roll_two":    dict(notch=True,  roll=True,  key="unit", K=2),
    "roll_bundle": dict(notch=True,  roll=True,  key="roll", K=1),
}


def variant(lr, radius, notch, roll, key, K, tol=4.0):
    return dict(use_latent=True, use_strat=True, strat_mode="soft",
                strat_lr=lr, strat_radius=radius, notch_tol_mm=tol,
                notch_order=True, use_notch=notch, use_roll_length=roll,
                roll_key=key, roll_len_classes=K)


def run_block(model, dep, seeds, n_slips, cal_slips, time_limit, workers, tag,
              out):
    """Sweep the join threshold for every variant, on the same corpora."""
    rows = []
    cal, _ = common.calibrate(model, STATE, cal_slips, dep=dep)
    cspec = common.make_spec(common.CAL_SEED0 + 50, cal_slips, STATE, dep)
    cinst, cmeta, _, _ = common.prep(model, cspec)
    radius = common.choose_radius(cmeta, cinst["joins"])
    lr = assemble.fit_context_lr(cmeta, cinst["joins"], radius=radius)
    base = assemble.AssemblyConfig(time_limit=time_limit, workers=workers)
    grid = assemble.theta_grid()

    for seed in seeds:
        spec = common.make_spec(common.EVAL_SEED0 + seed, n_slips, STATE, dep)
        inst, meta, S, slip_of = common.prep(model, spec)
        W = assemble.log_odds(S, cal)
        for name, c in CASES.items():
            for th in grid:
                cfg = common.cfg_for(base, theta=float(th),
                                     **variant(lr, radius, **c))
                t0 = time.time()
                prob = assemble.build_problem(meta, W, cfg)
                if not prob["arcs"]:
                    continue
                r = assemble.solve_cpsat(prob, workers=workers)
                e = common.score(len(meta), r["joins"], inst["joins"], slip_of)
                rows.append(dict(tag=tag, case=name, seed=seed,
                                 n_frag=len(meta), theta=float(th),
                                 roll_len_sigma_mm=dep.roll_len_sigma_mm,
                                 slips_per_roll=dep.slips_per_roll,
                                 solve_s=time.time() - t0,
                                 status=r["status"], **c, **e))
            k = [x for x in rows if x["case"] == name and x["seed"] == seed]
            best = max(k, key=lambda x: x["ari"]) if k else None
            if best:
                print(f"  [{tag}] seed {seed} {name}: best ARI "
                      f"{best['ari']:.3f} at theta {best['theta']:+.2f}, "
                      f"cross-slip there {best['cross_slip_rate']:.3f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--only-control", action="store_true",
                    help="run only the no-roll-standard control block")
    ap.add_argument("--out", default=str(common.RESULTS / "notch_roll.csv"))
    a = ap.parse_args()
    if a.only_control:
        a.out = str(common.RESULTS / "notch_roll_control.csv")

    model = common.load_matcher()
    seeds = list(range(a.seeds))
    spatial = dict(model="spatial", disperse_sigma_mm=30.0, block_mm=250.0)
    rows = []

    blocks = ((0.0, "roll_none"),) if a.only_control else \
        ((2.0, "roll_typical"), (0.0, "roll_none"))
    for sig, tag in blocks:
        print(f"roll spread {sig} mm", flush=True)
        dep = synth.DepositionSpec(roll_len_sigma_mm=sig, slips_per_roll=30,
                                   **spatial)
        rows += run_block(model, dep, seeds, a.slips, a.cal_slips,
                          a.time_limit, a.workers, tag, a.out)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
