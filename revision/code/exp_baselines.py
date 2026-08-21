"""E4. How much of the gain is the object model, and how much is any global reasoning.

The submitted manuscript compared against maximum-weight bipartite matching
only. Reviewer 2 asks for the global methods used in puzzle solving, which
enforce loop consistency or assemble along shortest paths, since without them
it is unclear whether the gain comes from modelling the slip or merely from
reasoning globally at all.

Five methods that reason globally but carry no model of the object are run
against the constrained model, on the same candidate set, the same calibrated
weights and the same corpora: mutual best, greedy chain assembly, loop
consistent filtering, shortest-path assembly, and maximum-weight bipartite
matching. Each is given its own tuned threshold, so none is reported at an
operating point chosen to suit another.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import common
import global_baselines as gb
from common import assemble, synth

sys_baselines = None


def _import_published_baselines():
    global sys_baselines
    if sys_baselines is None:
        import baselines as b
        sys_baselines = b
    return sys_baselines


ARC_METHODS = ("greedy", "loop", "path", "bipartite")


def arc_only_cfg(base, theta):
    """The candidate set every global baseline is given.

    Uniqueness aside, none of these methods can use a constraint, so they are
    handed the arcs and weights of an unconstrained problem. Giving them the
    gated arc set instead would credit them with the excavation context and the
    morphometry they have no way to represent.
    """
    return replace(base, theta=theta, use_uniqueness=True, use_length=False,
                   use_width=False, use_notch=False, use_hand=False,
                   use_strat=False, use_latent=False)


def run_baseline(name, prob):
    if name == "greedy":
        return gb.greedy_chain(prob)
    if name == "loop":
        return gb.loop_consistency(prob)
    if name == "path":
        return gb.shortest_path_assembly(prob)
    if name == "bipartite":
        return assemble.solve_cpsat(prob)["joins"]
    raise KeyError(name)


def tune_baseline(model, state, cal_slips, base, dep, cal, workers):
    """Threshold for each baseline, on calibration corpora, by join F1."""
    spec = common.make_spec(common.CAL_SEED0 + 100, cal_slips, state, dep)
    inst, meta, S, slip_of = common.prep(model, spec)
    W = assemble.log_odds(S, cal)
    grid = assemble.theta_grid(assemble.WIDE_P_GRID)
    out = {}
    for name in ARC_METHODS:
        best, best_v = float(grid[0]), -np.inf
        for th in grid:
            prob = assemble.build_problem(meta, W, arc_only_cfg(base, float(th)))
            if not prob["arcs"]:
                continue
            e = common.score(len(meta), run_baseline(name, prob),
                             inst["joins"], slip_of)
            v = e["f1"] if np.isfinite(e["f1"]) else 0.0
            if v > best_v:
                best_v, best = v, float(th)
        out[name] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="P2,P3,P4,P5")
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(common.RESULTS / "baselines.csv"))
    a = ap.parse_args()

    model = common.load_matcher()
    dep = synth.DepositionSpec(model="spatial", disperse_sigma_mm=30.0,
                               block_mm=250.0)
    rows = []

    for state in a.states.split(","):
        cal, _ = common.calibrate(model, state, a.cal_slips, dep=dep)
        cspec = common.make_spec(common.CAL_SEED0 + 50, a.cal_slips, state, dep)
        cinst, cmeta, _, _ = common.prep(model, cspec)
        radius = common.choose_radius(cmeta, cinst["joins"])
        lr = assemble.fit_context_lr(cmeta, cinst["joins"], radius=radius)

        base = assemble.AssemblyConfig(time_limit=a.time_limit,
                                       workers=a.workers, strat_lr=lr,
                                       strat_radius=radius)
        thetas = tune_baseline(model, state, a.cal_slips, base, dep, cal,
                               a.workers)
        full_kw = dict(use_latent=True, use_strat=True, strat_mode="soft",
                       notch_order=True)
        cfg = common.cfg_for(base, time_limit=12.0, **full_kw)
        th_full, edge = common.tune(model, state, a.cal_slips, cfg, dep=dep,
                                    cal=cal, workers=a.workers)
        print(f"[{state}] thetas " +
              " ".join(f"{k} {v:+.2f}" for k, v in thetas.items()) +
              f" constrained {th_full:+.2f}"
              f"{' (endpoint)' if edge else ''}", flush=True)

        for seed in range(a.seeds):
            spec = common.make_spec(common.EVAL_SEED0 + seed, a.slips, state,
                                    dep)
            inst, meta, S, slip_of = common.prep(model, spec)
            W = assemble.log_odds(S, cal)
            n = len(meta)
            acc = __import__("matcher").topk_accuracy(S, inst["joins"])
            comm = dict(state=state, seed=seed, n_frag=n, top1=acc["top1"],
                        top50=acc["top50"])

            b = _import_published_baselines()
            for name, fn in (("top1", b.top1), ("mutual", b.mutual_best)):
                e = common.score(n, fn(W), inst["joins"], slip_of)
                rows.append({**comm, "method": name, "theta": np.nan,
                             "solve_s": 0.0, **e})

            for name in ARC_METHODS:
                prob = assemble.build_problem(
                    meta, W, arc_only_cfg(base, thetas[name]))
                t0 = time.time()
                pj = run_baseline(name, prob)
                dt = time.time() - t0
                e = common.score(n, pj, inst["joins"], slip_of)
                rows.append({**comm, "method": name, "theta": thetas[name],
                             "solve_s": dt, "n_arcs": len(prob["arcs"]), **e})

            cfg = common.cfg_for(base, theta=th_full, **full_kw)
            prob = assemble.build_problem(meta, W, cfg)
            t0 = time.time()
            r = assemble.solve_cpsat(prob, workers=a.workers)
            e = common.score(n, r["joins"], inst["joins"], slip_of)
            rows.append({**comm, "method": "constrained", "theta": th_full,
                         "solve_s": time.time() - t0, "status": r["status"],
                         "n_arcs": len(prob["arcs"]), **e})

            got = {x["method"]: x["ari"] for x in rows[-7:]}
            print(f"  [{state}] seed {seed} " +
                  " ".join(f"{k} {v:.3f}" for k, v in got.items()), flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
