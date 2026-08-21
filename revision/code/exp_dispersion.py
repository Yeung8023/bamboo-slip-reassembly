"""E1. What the excavation record is worth, once fragments actually disperse.

The submitted manuscript gave every fragment of a slip the same excavation unit
and never got that record wrong. That is the best case an excavation could
deliver, and it is not a case any excavation delivers. Here the record is
produced by an explicit deposition model: slips are bound into rolls, rolls are
laid on the chamber floor, each fragment is displaced from where it was
deposited, and the unit recorded for it is the square of the excavation grid it
was finally recovered from.

Two sweeps. How far the fragments of one slip scattered, and how finely the
site was subdivided. Both are reported against the quantity that actually
governs the outcome, the fraction of true joins whose two fragments carry the
same record, which we call context coherence.

Three treatments of the same evidence are compared at every setting: the
published hard gate, the graded evidence introduced in the revision, and no
context at all.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common
from common import assemble, synth

STATE = "P4"
TUNE_LIMIT = 12.0   # threshold selection only has to rank thresholds


# Methods that read the excavation record. Only these are affected by how the
# fragments dispersed, so only these are rerun at every setting.
def treatments(lr, radius):
    return {
        "context_hard": dict(use_latent=True, use_strat=True,
                             strat_mode="hard"),
        "context_soft": dict(use_latent=True, use_strat=True,
                             strat_mode="soft", strat_lr=lr,
                             strat_radius=radius),
    }


# Methods that ignore it. The deposition model changes only which square a
# fragment was recorded in, and never the imagery, so these two produce the
# same result at every setting and are run once as the reference line.
REFERENCE = {
    "context_none": dict(use_latent=True, use_strat=False),
    "matching": dict(use_uniqueness=True, use_length=False, use_width=False,
                     use_notch=False, use_hand=False, use_strat=False),
}


def run_setting(model, dep, obs, seeds, n_slips, cal_slips, time_limit,
                workers, tag, reference=False):
    rows = []
    cal, _ = common.calibrate(model, STATE, cal_slips, dep=dep, obs=obs)

    # The radius and the graded weights are read off a calibration corpus
    # deposited the same way, never off an evaluation corpus.
    cspec = common.make_spec(common.CAL_SEED0 + 50, cal_slips, STATE, dep, obs)
    cinst, cmeta, cS, _ = common.prep(model, cspec)
    radius = common.choose_radius(cmeta, cinst["joins"])
    lr = assemble.fit_context_lr(cmeta, cinst["joins"], radius=radius)

    base = assemble.AssemblyConfig(time_limit=time_limit, workers=workers)
    tr = treatments(lr, radius)
    corpus = common.tune_corpus(model, STATE, cal_slips, cal, dep=dep, obs=obs)
    thetas = {}
    for name, kw in tr.items():
        cfg = common.cfg_for(base, time_limit=TUNE_LIMIT, **kw)
        thetas[name], edge = common.tune_on(corpus, cfg, workers=workers)
        if edge:
            print(f"    warning: theta({name}) is at a grid endpoint",
                  flush=True)

    if reference:
        tr = dict(tr, **REFERENCE)
        for name, kw in REFERENCE.items():
            cfg = common.cfg_for(base, time_limit=TUNE_LIMIT, **kw)
            thetas[name], edge = common.tune_on(corpus, cfg, workers=workers)
            if edge:
                print(f"    warning: theta({name}) is at a grid endpoint",
                      flush=True)

    for seed in seeds:
        spec = common.make_spec(common.EVAL_SEED0 + seed, n_slips, STATE,
                                dep, obs)
        inst, meta, S, slip_of = common.prep(model, spec)
        W = assemble.log_odds(S, cal)
        coh = common.coherence(meta, inst["joins"])
        for name, kw in tr.items():
            cfg = common.cfg_for(base, theta=thetas[name], **kw)
            t0 = time.time()
            prob = assemble.build_problem(meta, W, cfg)
            r = assemble.solve_cpsat(prob, workers=workers)
            dt = time.time() - t0
            e = common.score(len(meta), r["joins"], inst["joins"], slip_of)
            rows.append(dict(
                tag=tag, method=name, seed=seed, n_frag=len(meta),
                disperse_sigma_mm=dep.disperse_sigma_mm,
                block_mm=dep.block_mm,
                unit_known_frac=obs.unit_known_frac,
                unit_error_frac=obs.unit_error_frac,
                coherence=coh, radius=radius, theta=thetas[name],
                n_arcs=len(prob["arcs"]),
                arc_recall=common.arc_recall(prob, inst["joins"]),
                solve_s=dt, status=r["status"], **e))
        got = {x["method"]: x["ari"] for x in rows[-len(tr):]}
        print(f"  [{tag}] seed {seed} coherence {coh:.3f} " +
              " ".join(f"{k} {v:.3f}" for k, v in got.items()), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="",
                    help="run only these blocks: uniform,sigma,grid,record")
    ap.add_argument("--out", default=str(common.RESULTS / "dispersion.csv"))
    a = ap.parse_args()

    model = common.load_matcher()
    seeds = list(range(a.seeds))
    want = set(a.only.split(",")) if a.only.strip() else None
    rows = []

    def wanted(tag):
        return want is None or tag in want

    # The published treatment: one unit per slip, never wrong. The two methods
    # that ignore the record are run here and only here, since the deposition
    # model changes nothing they can see.
    if wanted("uniform"):
        print("uniform (as published), with the context-free reference line",
              flush=True)
        rows += run_setting(model, synth.DepositionSpec(model="uniform"),
                            synth.ObservationSpec(), seeds, a.slips,
                            a.cal_slips, a.time_limit, a.workers, "uniform",
                            reference=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    # How far the fragments of one slip scattered.
    if wanted("sigma"):
        for sigma in (0.0, 15.0, 30.0, 60.0, 120.0, 240.0):
            print(f"dispersion sigma {sigma:.0f} mm", flush=True)
            dep = synth.DepositionSpec(model="spatial",
                                       disperse_sigma_mm=sigma,
                                       block_mm=250.0)
            rows += run_setting(model, dep, synth.ObservationSpec(), seeds,
                                a.slips, a.cal_slips, a.time_limit, a.workers,
                                "sigma")
            pd.DataFrame(rows).to_csv(a.out, index=False)

    # How finely the site was subdivided.
    if wanted("grid"):
        for block in (125.0, 250.0, 500.0, 1000.0):
            print(f"grid {block:.0f} mm", flush=True)
            dep = synth.DepositionSpec(model="spatial", disperse_sigma_mm=30.0,
                                       block_mm=block)
            rows += run_setting(model, dep, synth.ObservationSpec(), seeds,
                                a.slips, a.cal_slips, a.time_limit, a.workers,
                                "grid")
            pd.DataFrame(rows).to_csv(a.out, index=False)

    # How good the record itself is.
    if wanted("record"):
        for known, err in ((1.0, 0.1), (1.0, 0.3), (0.7, 0.0), (0.4, 0.0),
                           (0.7, 0.2)):
            print(f"record known {known} error {err}", flush=True)
            dep = synth.DepositionSpec(model="spatial", disperse_sigma_mm=30.0,
                                       block_mm=250.0)
            obs = synth.ObservationSpec(unit_known_frac=known,
                                        unit_error_frac=err)
            rows += run_setting(model, dep, obs, seeds, a.slips, a.cal_slips,
                                a.time_limit, a.workers, "record")
            pd.DataFrame(rows).to_csv(a.out, index=False)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
