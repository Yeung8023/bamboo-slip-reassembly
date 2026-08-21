"""E5. The pipeline on photographs of real excavated slips.

Reviewer 2 asks for a demonstration on real fragments. No public corpus of real
slips carries verified rejoining ground truth, and a reconstruction metric
cannot be computed without it, so the demonstration available is this one: real
material, generated fracture.

Every image the matcher sees here is a photograph of an excavated Qin or Han
slip \\cite{deepjiandu}, with its own grain, ink, staining, biological attack
and eroded edges. The fracture, the taphonomy applied afterwards, the
descriptors and the assembly model are the same code used throughout. The
matcher is the one trained entirely on synthetic corpora and is not retrained,
so what is measured is a zero-shot transfer onto real material.

The control is the identical experiment on the rendered substrate, at the same
seeds, the same damage and the same deposition, so the only thing that differs
between the two columns of the result is what the slips are made of.

The photographs used for calibration are disjoint from those used for
evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
import real_corpus
from common import assemble, synth

STATE = "P4"
TUNE_LIMIT = 12.0


def build(model, substrate, seed, n_slips, dep):
    """One corpus, scored, with the imagery stripped from the metadata."""
    spec = common.make_spec(seed, n_slips, STATE, dep=dep)
    inst = synth.generate_instance(spec, substrate=substrate)
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = __import__("matcher").score_matrix(model, common.descs(inst))
    slip_of = [f["slip_id"] for f in inst["frags"]]
    return inst, meta, S, slip_of


def run(model, tag, substrate_cal, substrate_eval, seeds, n_slips, cal_slips,
        dep, time_limit, workers):
    """Calibrate, tune and evaluate one substrate."""
    rows = []
    ci, cm, cS, _ = build(model, substrate_cal, common.CAL_SEED0, cal_slips,
                          dep)
    cal = assemble.fit_calibration(cS, ci["joins"])
    radius = common.choose_radius(cm, ci["joins"])
    lr = assemble.fit_context_lr(cm, ci["joins"], radius=radius)

    ti, tm, tS, tslip = build(model, substrate_cal, common.CAL_SEED0 + 100,
                              cal_slips, dep)
    corpus = ([tm], [assemble.log_odds(tS, cal)], [(ti["joins"], tslip)])

    base = assemble.AssemblyConfig(time_limit=time_limit, workers=workers,
                                   strat_lr=lr, strat_radius=radius)
    kws = {"constrained": dict(use_latent=True, use_strat=True,
                               strat_mode="soft", notch_order=True),
           "bipartite": dict(use_uniqueness=True, use_length=False,
                             use_width=False, use_notch=False, use_hand=False,
                             use_strat=False)}
    thetas = {}
    for name, kw in kws.items():
        cfg = common.cfg_for(base, time_limit=TUNE_LIMIT, **kw)
        thetas[name], edge = common.tune_on(corpus, cfg, workers=workers)
        if edge:
            print(f"    warning: theta({name}) at a grid endpoint", flush=True)

    for seed in seeds:
        inst, meta, S, slip_of = build(model, substrate_eval,
                                       common.EVAL_SEED0 + seed, n_slips, dep)
        W = assemble.log_odds(S, cal)
        acc = __import__("matcher").topk_accuracy(S, inst["joins"])
        for name, kw in kws.items():
            cfg = common.cfg_for(base, theta=thetas[name], **kw)
            t0 = time.time()
            prob = assemble.build_problem(meta, W, cfg)
            r = assemble.solve_cpsat(prob, workers=workers)
            e = common.score(len(meta), r["joins"], inst["joins"], slip_of)
            rows.append(dict(substrate=tag, method=name, seed=seed,
                             n_frag=len(meta), n_true_joins=len(inst["joins"]),
                             top1=acc["top1"], top5=acc["top5"],
                             top10=acc["top10"], top50=acc["top50"],
                             theta=thetas[name], solve_s=time.time() - t0,
                             status=r["status"], **e))
        print(f"  [{tag}] seed {seed} {len(meta)} frags top1 "
              f"{acc['top1']:.3f} top50 {acc['top50']:.3f} " +
              " ".join(f"{x['method']} {x['ari']:.3f}" for x in rows[-2:]),
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(common.ROOT / "revision" /
                                            "data_real" / "images"))
    ap.add_argument("--screened", default=str(common.ROOT / "revision" /
                                              "data_real" / "screened.json"))
    ap.add_argument("--slips", type=int, default=200)
    ap.add_argument("--cal-slips", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=1500)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(common.RESULTS / "real.csv"))
    a = ap.parse_args()

    if Path(a.screened).exists():
        paths = json.loads(Path(a.screened).read_text())
    else:
        cand = real_corpus.index_images(a.images, limit=a.max_images)
        print(f"screening {len(cand)} photographs", flush=True)
        paths = real_corpus.screen(cand, out_json=a.screened)
    print(f"{len(paths)} photographs segment into a single slip", flush=True)
    if len(paths) < 40:
        raise SystemExit("too few usable photographs")

    half = len(paths) // 2
    cal_paths, eval_paths = paths[:half], paths[half:]

    model = common.load_matcher()
    dep = synth.DepositionSpec(model="spatial", disperse_sigma_mm=30.0,
                               block_mm=250.0)
    seeds = list(range(a.seeds))

    rows = []
    print("rendered substrate (control)", flush=True)
    rows += run(model, "synthetic", None, None, seeds, a.slips, a.cal_slips,
                dep, a.time_limit, a.workers)
    pd.DataFrame(rows).to_csv(a.out, index=False)

    print("photographed slips", flush=True)
    rows += run(model, "real",
                real_corpus.RealSubstrate(cal_paths, seed=1),
                real_corpus.RealSubstrate(eval_paths, seed=2),
                seeds, a.slips, a.cal_slips, dep, a.time_limit, a.workers)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
