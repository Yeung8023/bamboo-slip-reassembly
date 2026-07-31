"""Experiment grid: preservation state x corpus size x seed x method.

Protocol, fixed in advance:

  * The matcher is trained once, on corpora drawn from a seed range disjoint
    from every corpus used here, with preservation randomised over the whole
    ladder so that it is not tuned to any one operating point.
  * Score calibration and the join-claiming threshold theta are fitted on
    dedicated calibration corpora (another disjoint seed range), separately for
    every method, so that each method is reported at its own best operating
    point.
  * Nothing is fitted on the evaluation corpora.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import assemble
import baselines
import matcher as M
import metrics
import preservation as pres
import synth

CAL_SEED0 = 900000
EVAL_SEED0 = 0


def descs(inst):
    F = inst["frags"]
    return dict(
        bot_prof=np.array([f["bot_prof"] for f in F]),
        bot_patch=np.array([f["bot_patch"] for f in F]),
        top_prof=np.array([f["top_prof"] for f in F]),
        top_patch=np.array([f["top_patch"] for f in F]),
    )


def prep(model, seed, n_slips, damage):
    inst = synth.generate_instance(
        synth.InstanceSpec(n_slips=n_slips, seed=seed, damage=damage))
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = M.score_matrix(model, descs(inst))
    slip_of = [f["slip_id"] for f in inst["frags"]]
    return inst, meta, S, slip_of


# Method = a constraint subset of the same model, so every comparison is an
# ablation rather than a contest between implementations.
METHODS = {
    "matching":  dict(use_uniqueness=True, use_length=False, use_width=False,
                      use_notch=False, use_hand=False, use_strat=False),
    "morph":     dict(use_uniqueness=True, use_length=False, use_width=True,
                      use_notch=False, use_hand=False, use_strat=True),
    "length":    dict(use_uniqueness=True, use_length=True, use_width=True,
                      use_notch=False, use_hand=False, use_strat=True),
    "full":      dict(use_uniqueness=True, use_length=True, use_width=True,
                      use_notch=True, use_hand=True, use_strat=True),
    # Per-slip latent length, width and scribal hand propagated along each
    # chain, replacing the pairwise proxies for them.
    "latent":    dict(use_uniqueness=True, use_length=True, use_width=True,
                      use_notch=True, use_hand=True, use_strat=True,
                      use_latent=True),
    "latent_no_notch": dict(use_uniqueness=True, use_length=True,
                            use_width=True, use_notch=False, use_hand=True,
                            use_strat=True, use_latent=True),
    "latent_no_hand": dict(use_uniqueness=True, use_length=True,
                           use_width=True, use_notch=True, use_hand=False,
                           use_strat=True, use_latent=True),
    "no_length": dict(use_uniqueness=True, use_length=False, use_width=True,
                      use_notch=False, use_hand=True, use_strat=True),
    "no_notch":  dict(use_uniqueness=True, use_length=True, use_width=True,
                      use_notch=False, use_hand=True, use_strat=True),
    "no_width":  dict(use_uniqueness=True, use_length=True, use_width=False,
                      use_notch=True, use_hand=True, use_strat=True),
    "no_strat":  dict(use_uniqueness=True, use_length=True, use_width=True,
                      use_notch=True, use_hand=True, use_strat=False),
    "no_hand":   dict(use_uniqueness=True, use_length=True, use_width=True,
                      use_notch=True, use_hand=False, use_strat=True),
}

HEURISTICS = ("top1", "mutual")


def run_cell(model, state, n_slips, seed, cal, thetas, base_cfg, methods):
    D = pres.damage(state)
    inst, meta, S, slip_of = prep(model, seed, n_slips, D)
    W = assemble.log_odds(S, cal)
    n = len(meta)
    tj = inst["joins"]
    acc = M.topk_accuracy(S, tj)
    rows = []

    common = dict(state=state, n_slips=n_slips, seed=seed, n_frag=n,
                  n_true_joins=len(tj), top1=acc["top1"], top5=acc["top5"],
                  top10=acc["top10"], top50=acc["top50"], mrr=acc["mrr"])

    for h in HEURISTICS:
        pj = baselines.top1(W) if h == "top1" else baselines.mutual_best(W)
        e = metrics.evaluate(n, pj, tj, slip_of)
        rows.append({**common, "method": h, "theta": np.nan, "solve_s": 0.0,
                     "status": "-", **e})

    for k in (1, 5, 10, 50):
        pj, cost = baselines.expert_topk(W, tj, k=k)
        e = metrics.evaluate(n, pj, tj, slip_of)
        rows.append({**common, "method": f"expert_top{k}", "theta": np.nan,
                     "solve_s": 0.0, "status": "-", "review_pairs": cost, **e})

    for name in methods:
        cfg = replace(base_cfg, theta=thetas.get(name, 0.0), **METHODS[name])
        t0 = time.time()
        prob = assemble.build_problem(meta, W, cfg)
        r = assemble.solve_cpsat(prob)
        dt = time.time() - t0
        e = metrics.evaluate(n, r["joins"], tj, slip_of)
        rows.append({**common, "method": name, "theta": cfg.theta,
                     "solve_s": dt, "status": r["status"],
                     "n_arcs": len(prob["arcs"]), "objective": r["objective"],
                     **e})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="P1,P2,P3,P4,P5")
    ap.add_argument("--sizes", default="150,400,900")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--methods", default="matching,morph,length,full")
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--out", default="results/runs.csv")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tune-time-limit", type=float, default=30.0)
    ap.add_argument("--cal-slips", type=int, default=150)
    ap.add_argument("--cal-n", type=int, default=2)
    ap.add_argument("--theta-grid", default="",
                    help="posterior thresholds (e.g. 0.5,0.9,0.99); "
                         "blank uses the default ladder-comparable grid")
    a = ap.parse_args()

    states = a.states.split(",")
    sizes = [int(s) for s in a.sizes.split(",")]
    methods = a.methods.split(",")
    grid = (assemble.theta_grid() if not a.theta_grid.strip()
            else assemble.theta_grid([float(x) for x in a.theta_grid.split(",")]))

    model = M.load_model(a.model)
    base_cfg = assemble.AssemblyConfig(time_limit=a.time_limit,
                                       workers=a.workers)

    rows, calib_log = [], []
    for state in states:
        D = pres.damage(state)
        t0 = time.time()
        ci, cm, cS, cslip = prep(model, CAL_SEED0, a.cal_slips, D)
        cal = assemble.fit_calibration(cS, ci["joins"])

        cal_meta, cal_W, cal_truth = [], [], []
        for r in range(a.cal_n):
            i2, m2, S2, s2 = prep(model, CAL_SEED0 + 100 + r, a.cal_slips, D)
            cal_meta.append(m2)
            cal_W.append(assemble.log_odds(S2, cal))
            cal_truth.append((i2["joins"], s2))

        thetas = {}
        for name in methods:
            cfg = replace(base_cfg, time_limit=a.tune_time_limit,
                          **METHODS[name])
            th, v = assemble.tune_theta(cal_meta, cal_W, cal_truth, cfg,
                                        grid=grid, workers=a.workers)
            thetas[name] = th
            calib_log.append(dict(state=state, method=name, theta=th, cal_f1=v))
            print(f"[{state}] theta({name}) = {th:+.1f}  (cal F1 {v:.3f})",
                  flush=True)
        print(f"[{state}] calibration a={cal[0]:.2f} b={cal[1]:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        for n_slips in sizes:
            for seed in range(a.seeds):
                t1 = time.time()
                rows += run_cell(model, state, n_slips, EVAL_SEED0 + seed,
                                 cal, thetas, base_cfg, methods)
                print(f"  [{state}] size {n_slips} seed {seed} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(calib_log).to_csv(
        str(Path(a.out).with_name("calibration.csv")), index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
