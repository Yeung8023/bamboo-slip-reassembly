"""E2. Why the binding notches were worth nothing, and what makes them worth something.

The submitted ablation found that the notch constraint adds 0.006 of grouping
index. Three explanations were available and the manuscript did not separate
them: the tolerance window is too wide, the corpus spreads slip length over
47 mm so every window is wide whatever the tolerance, or the constraint simply
fails to use the spacing between notches.

Part A takes the three apart. The tolerance is swept; the spread of slip length
in the corpus is swept independently of it; and the ordering constraint that
stops two notches on one fragment claiming the same cord is switched on and off.

Part B is the fix. A roll of slips is one manuscript and was trimmed to one
standard length, which is why the lengths at Shuihudi cluster by text rather
than spreading evenly. Estimating that length jointly over the slips recorded
in one excavation square collapses the admissible band around each cord from
the spread of the corpus to the spread of one roll. The same is reported with
the true bundle as the grouping key, which is the upper bound available if
excavations were to keep bundle membership.
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


def variants(lr, radius, tol, order, roll, roll_key, notch):
    return dict(use_latent=True, use_strat=True, strat_mode="soft",
                strat_lr=lr, strat_radius=radius, notch_tol_mm=tol,
                notch_order=order, use_roll_length=roll, roll_key=roll_key,
                use_notch=notch)


def run_block(model, dep, cases, seeds, n_slips, cal_slips, time_limit,
              workers, tag):
    """One deposition setting, many treatments of the notch evidence."""
    rows = []
    cal, _ = common.calibrate(model, STATE, cal_slips, dep=dep)
    cspec = common.make_spec(common.CAL_SEED0 + 50, cal_slips, STATE, dep)
    cinst, cmeta, _, _ = common.prep(model, cspec)
    radius = common.choose_radius(cmeta, cinst["joins"])
    lr = assemble.fit_context_lr(cmeta, cinst["joins"], radius=radius)

    base = assemble.AssemblyConfig(time_limit=time_limit, workers=workers)
    corpus = common.tune_corpus(model, STATE, cal_slips, cal, dep=dep)
    thetas = {}
    for name, kw in cases.items():
        cfg = common.cfg_for(base, time_limit=TUNE_LIMIT,
                             **variants(lr, radius, **kw))
        thetas[name], edge = common.tune_on(corpus, cfg, workers=workers)
        if edge:
            print(f"    warning: theta({name}) at a grid endpoint", flush=True)

    for seed in seeds:
        spec = common.make_spec(common.EVAL_SEED0 + seed, n_slips, STATE, dep)
        inst, meta, S, slip_of = common.prep(model, spec)
        W = assemble.log_odds(S, cal)
        for name, kw in cases.items():
            cfg = common.cfg_for(base, theta=thetas[name],
                                 **variants(lr, radius, **kw))
            t0 = time.time()
            prob = assemble.build_problem(meta, W, cfg)
            r = assemble.solve_cpsat(prob, workers=workers)
            e = common.score(len(meta), r["joins"], inst["joins"], slip_of)
            rows.append(dict(tag=tag, case=name, seed=seed, n_frag=len(meta),
                             roll_len_sigma_mm=dep.roll_len_sigma_mm,
                             slips_per_roll=dep.slips_per_roll,
                             theta=thetas[name], solve_s=time.time() - t0,
                             status=r["status"], **kw, **e))
        base_ari = [x["ari"] for x in rows[-len(cases):]
                    if x["case"] == "no_notch"]
        got = {x["case"]: x["ari"] for x in rows[-len(cases):]}
        print(f"  [{tag}] seed {seed} " +
              " ".join(f"{k} {v:.3f}" for k, v in got.items()), flush=True)
    return rows


# Part A: is it the tolerance, the spread of length in the corpus, or the
# failure to use the spacing between notches?
CASES_A = {
    "no_notch":   dict(tol=16.0, order=False, roll=False, roll_key="unit",
                       notch=False),
    "tol16":      dict(tol=16.0, order=False, roll=False, roll_key="unit",
                       notch=True),
    "tol8":       dict(tol=8.0, order=False, roll=False, roll_key="unit",
                       notch=True),
    "tol4":       dict(tol=4.0, order=False, roll=False, roll_key="unit",
                       notch=True),
    "tol2":       dict(tol=2.0, order=False, roll=False, roll_key="unit",
                       notch=True),
    "tol4_order": dict(tol=4.0, order=True, roll=False, roll_key="unit",
                       notch=True),
}

# Part B: the joint estimate of length across the slips of one roll.
CASES_B = {
    "no_notch":        dict(tol=4.0, order=True, roll=False, roll_key="unit",
                            notch=False),
    "notch":           dict(tol=4.0, order=True, roll=False, roll_key="unit",
                            notch=True),
    "roll_unit":       dict(tol=4.0, order=True, roll=True, roll_key="unit",
                            notch=True),
    "roll_true":       dict(tol=4.0, order=True, roll=True, roll_key="roll",
                            notch=True),
    "roll_unit_nonotch": dict(tol=4.0, order=True, roll=True,
                              roll_key="unit", notch=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(common.RESULTS / "notch.csv"))
    a = ap.parse_args()

    model = common.load_matcher()
    seeds = list(range(a.seeds))
    rows = []
    spatial = dict(model="spatial", disperse_sigma_mm=30.0, block_mm=250.0)

    # A1. lengths drawn independently for every slip, as in the submitted work
    print("A: tolerance sweep, slip lengths independent", flush=True)
    rows += run_block(model, synth.DepositionSpec(**spatial), CASES_A, seeds,
                      a.slips, a.cal_slips, a.time_limit, a.workers,
                      "tol_iid")
    pd.DataFrame(rows).to_csv(a.out, index=False)

    # A2. the same sweep when a roll imposes one standard length, so that the
    # effect of the tolerance can be told apart from the effect of the spread
    print("A: tolerance sweep, one standard length per roll", flush=True)
    rows += run_block(model, synth.DepositionSpec(roll_len_sigma_mm=2.0,
                                                  slips_per_roll=30, **spatial),
                      CASES_A, seeds, a.slips, a.cal_slips, a.time_limit,
                      a.workers, "tol_roll")
    pd.DataFrame(rows).to_csv(a.out, index=False)

    # B. the joint estimate, at three degrees of length standardisation
    for sig, tagg in ((1.0, "roll_tight"), (2.0, "roll_typical"),
                      (5.0, "roll_loose")):
        print(f"B: joint length estimate, roll sigma {sig} mm", flush=True)
        rows += run_block(model,
                          synth.DepositionSpec(roll_len_sigma_mm=sig,
                                               slips_per_roll=30, **spatial),
                          CASES_B, seeds, a.slips, a.cal_slips, a.time_limit,
                          a.workers, tagg)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    # B control: no roll standard at all, so the pooling has nothing to find
    print("B control: no roll standard", flush=True)
    rows += run_block(model, synth.DepositionSpec(**spatial), CASES_B, seeds,
                      a.slips, a.cal_slips, a.time_limit, a.workers,
                      "roll_none")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
