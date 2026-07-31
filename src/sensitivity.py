"""Sobol sensitivity of the reconstruction to corpus condition and to the
method's own tolerances.

The question is whether the reported gains come from the constraint set or from
a fortunate choice of its two tolerances.  Burial condition and observation
quality are varied alongside the tolerances, so the comparison is on a common
footing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROBLEM = {
    "num_vars": 10,
    "names": [
        "face_loss_mm", "pullout_mean_mm", "face_texture_noise",
        "p_gap", "p_fragment_lost", "breaks_per_slip",
        "hand_accuracy", "notch_sigma_mm",
        "notch_tol_mm", "width_tol_mm",
    ],
    "bounds": [
        [0.20, 1.90], [0.45, 1.90], [0.05, 0.60],
        [0.15, 0.60], [0.02, 0.25], [1.2, 3.5],
        [0.50, 0.98], [0.20, 1.50],
        [6.0, 30.0], [0.20, 0.90],
    ],
}
DAMAGE_KEYS = {"face_loss_mm", "pullout_mean_mm", "face_texture_noise",
               "p_gap", "p_fragment_lost", "breaks_per_slip"}
OBS_KEYS = {"hand_accuracy", "notch_sigma_mm"}
CFG_KEYS = {"notch_tol_mm", "width_tol_mm"}

_G = {}


def _init(model_path, n_slips, seed, time_limit, theta):
    import matcher as M
    import assemble
    import synth
    import run_experiments as RX

    _G["M"] = M
    _G["assemble"] = assemble
    _G["synth"] = synth
    _G["RX"] = RX
    _G["model"] = M.load_model(model_path)
    _G["n_slips"] = n_slips
    _G["seed"] = seed
    _G["time_limit"] = time_limit
    _G["theta"] = theta


def _one(args):
    idx, vals = args
    M, assemble, synth, RX = _G["M"], _G["assemble"], _G["synth"], _G["RX"]
    import metrics

    p = dict(zip(PROBLEM["names"], vals))
    dmg = synth.DamageSpec(**{k: float(p[k]) for k in DAMAGE_KEYS})
    obs = synth.ObservationSpec(hand_accuracy=float(p["hand_accuracy"]),
                                notch_sigma_mm=float(p["notch_sigma_mm"]))
    spec = synth.InstanceSpec(n_slips=_G["n_slips"], seed=_G["seed"],
                              damage=dmg, obs=obs)
    inst = synth.generate_instance(spec)
    if len(inst["joins"]) < 5:
        return dict(idx=idx, ari=0.0, f1=0.0, exact=0.0, top1=np.nan)
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = M.score_matrix(_G["model"], RX.descs(inst))

    cspec = synth.InstanceSpec(n_slips=max(60, _G["n_slips"] // 2),
                               seed=_G["seed"] + 777000, damage=dmg, obs=obs)
    cinst = synth.generate_instance(cspec)
    cal = assemble.fit_calibration(M.score_matrix(_G["model"], RX.descs(cinst)),
                                   cinst["joins"])
    W = assemble.log_odds(S, cal)

    cfg = assemble.AssemblyConfig(
        time_limit=_G["time_limit"], workers=1, theta=_G["theta"],
        notch_tol_mm=float(p["notch_tol_mm"]),
        width_tol_mm=float(p["width_tol_mm"]),
        hand_accuracy=float(p["hand_accuracy"]))
    prob = assemble.build_problem(meta, W, cfg)
    r = assemble.solve_cpsat(prob, workers=1)
    e = metrics.evaluate(len(meta), r["joins"], inst["joins"],
                         [f["slip_id"] for f in inst["frags"]])
    acc = M.topk_accuracy(S, inst["joins"], ks=(1,))
    return dict(idx=idx, ari=e["ari"], f1=(0.0 if not np.isfinite(e["f1"]) else e["f1"]),
                exact=(0.0 if not np.isfinite(e["exact_slip"]) else e["exact_slip"]),
                top1=acc["top1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=32)
    ap.add_argument("--slips", type=int, default=120)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=25.0)
    ap.add_argument("--theta", type=float, default=2.94)
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--out", default="results/sensitivity_runs.csv")
    a = ap.parse_args()

    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    X = sobol_sample.sample(PROBLEM, a.n_base, calc_second_order=False)
    print(f"{len(X)} evaluations, {a.workers} workers", flush=True)

    rows = [None] * len(X)
    with ProcessPoolExecutor(
            max_workers=a.workers, initializer=_init,
            initargs=(a.model, a.slips, a.seed, a.time_limit, a.theta)) as ex:
        for i, r in enumerate(ex.map(_one, list(enumerate(X)), chunksize=1)):
            rows[r["idx"]] = r
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(X)}", flush=True)

    d = pd.DataFrame(rows).sort_values("idx")
    for j, nm in enumerate(PROBLEM["names"]):
        d[nm] = X[:, j]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)

    out = []
    for met in ("ari", "f1", "exact"):
        Y = d[met].values.astype(float)
        Y = np.nan_to_num(Y)
        Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=False,
                                   print_to_console=False)
        for j, nm in enumerate(PROBLEM["names"]):
            out.append(dict(metric=met, param=nm, S1=Si["S1"][j],
                            S1_conf=Si["S1_conf"][j], ST=Si["ST"][j],
                            ST_conf=Si["ST_conf"][j]))
    t = pd.DataFrame(out)
    t.to_csv(str(Path(a.out).with_name("table_sobol.csv")), index=False)
    print(t[t.metric == "ari"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
