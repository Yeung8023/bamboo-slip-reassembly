"""E6. What happens when the corpus is not the corpus the method was built for.

A generator encodes the assumptions of whoever wrote it, and the only honest
response is to break those assumptions on purpose and report what survives.
Two ways of breaking them are used here, and in both the matcher, the score
calibration and the join threshold are fixed on the reference corpus first, as
they would be in practice, and then the corpus is changed underneath them.

Part A, misspecified taphonomy. The fracture physics, the rate of
fragmentation, the amount of material lost and the condition of the recovered
surfaces are all moved away from the reference at once, along a single
interpolation from the reference world to one the matcher has never seen.

Part B, uneven preservation. A real find is not uniformly preserved: pieces
from the top of a deposit are in a different state from those at the bottom. A
corpus is assembled from sub-corpora spanning P2 to P5 and reassembled as one.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common
from common import assemble, synth
import preservation as pres

STATE = "P4"

# A world the matcher was never shown: bamboo that broke into more and smaller
# pieces, along a crack that ran differently, and that lost more of itself
# before it was lifted.
OOD = dict(breaks_per_slip=4.2, pullout_shape=0.7, fibre_corr_mm=1.20,
           p_gap=0.55, gap_mean_mm=8.0, p_fragment_lost=0.28,
           erosion_mm=0.60, warp_deg=3.0, ink_fade=0.60, stain_rate=0.9,
           min_piece_mm=9.0)


def misspecified(level):
    """Damage spec a fraction ``level`` of the way from reference P4 to OOD."""
    ref = pres.damage(STATE)
    kw = {}
    for k, v in OOD.items():
        base = getattr(ref, k)
        kw[k] = float(base + level * (v - base))
    return pres.damage(STATE, **kw)


def merge(instances):
    """Concatenate corpora into one find, renumbering slips and fragments.

    Preservation varies within a deposit, so a corpus that is uniform in
    condition is itself an assumption. Merging sub-corpora of different states
    into one problem tests whether the method still works when the evidence is
    strong for some fragments and almost absent for others.
    """
    frags, joins, off, soff = [], [], 0, 0
    for part, inst in enumerate(instances):
        for f in inst["frags"]:
            g = dict(f)
            g["frag_id"] = f["frag_id"] + off
            g["slip_id"] = f["slip_id"] + soff
            # Each sub-corpus is generated on its own floor plan, so the unit
            # labels would otherwise collide and two fragments from different
            # parts of the find would appear to share an excavation square.
            # The parts are treated as different areas of the site, which is
            # also how preservation actually varies across a deposit.
            if g.get("unit", -1) >= 0:
                g["unit"] = int(g["unit"]) + part * 10_000_000
                rc = list(g.get("unit_rc", (0, 0)))
                rc[0] = int(rc[0]) + part * 10_000
                g["unit_rc"] = rc
            if g.get("roll", -1) >= 0:
                g["roll"] = int(g["roll"]) + part * 100_000
            frags.append(g)
        joins += [(a + off, b + off) for a, b in inst["joins"]]
        off += len(inst["frags"])
        soff += max((f["slip_id"] for f in inst["frags"]), default=-1) + 1
    return dict(frags=frags, joins=joins)


def evaluate(model, inst, cal, cfgs, workers):
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = __import__("matcher").score_matrix(model, common.descs(inst))
    slip_of = [f["slip_id"] for f in inst["frags"]]
    W = assemble.log_odds(S, cal)
    acc = __import__("matcher").topk_accuracy(S, inst["joins"])
    out = []
    for name, cfg in cfgs.items():
        t0 = time.time()
        prob = assemble.build_problem(meta, W, cfg)
        r = assemble.solve_cpsat(prob, workers=workers)
        e = common.score(len(meta), r["joins"], inst["joins"], slip_of)
        out.append(dict(method=name, n_frag=len(meta), top1=acc["top1"],
                        top50=acc["top50"], solve_s=time.time() - t0,
                        status=r["status"], coherence=common.coherence(
                            meta, inst["joins"]), **e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slips", type=int, default=400)
    ap.add_argument("--cal-slips", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="",
                    help="run only 'misspecification' or 'uneven'")
    ap.add_argument("--out", default=str(common.RESULTS / "realism.csv"))
    a = ap.parse_args()

    model = common.load_matcher()
    dep = synth.DepositionSpec(model="spatial", disperse_sigma_mm=30.0,
                               block_mm=250.0)

    # everything fitted once, on the reference corpus, and then left alone
    cal, _ = common.calibrate(model, STATE, a.cal_slips, dep=dep)
    cspec = common.make_spec(common.CAL_SEED0 + 50, a.cal_slips, STATE, dep)
    cinst, cmeta, _, _ = common.prep(model, cspec)
    radius = common.choose_radius(cmeta, cinst["joins"])
    lr = assemble.fit_context_lr(cmeta, cinst["joins"], radius=radius)
    base = assemble.AssemblyConfig(time_limit=a.time_limit, workers=a.workers,
                                   strat_lr=lr, strat_radius=radius)

    kws = {
        "constrained": dict(use_latent=True, use_strat=True,
                            strat_mode="soft", notch_order=True),
        "bipartite": dict(use_uniqueness=True, use_length=False,
                          use_width=False, use_notch=False, use_hand=False,
                          use_strat=False),
    }
    thetas = {}
    for name, kw in kws.items():
        cfg = common.cfg_for(base, time_limit=12.0, **kw)
        thetas[name], edge = common.tune(model, STATE, a.cal_slips, cfg,
                                         dep=dep, cal=cal, workers=a.workers)
        print(f"theta({name}) {thetas[name]:+.2f}"
              f"{' (endpoint)' if edge else ''}", flush=True)
    cfgs = {n: common.cfg_for(base, theta=thetas[n], **kw)
            for n, kw in kws.items()}

    rows = []
    want = set(a.only.split(",")) if a.only.strip() else None

    if want is None or "misspecification" in want:
        print("A: misspecified taphonomy", flush=True)
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            for seed in range(a.seeds):
                spec = common.make_spec(common.EVAL_SEED0 + seed, a.slips,
                                        STATE, dep=dep,
                                        damage=misspecified(level))
                inst = synth.generate_instance(spec)
                for r in evaluate(model, inst, cal, cfgs, a.workers):
                    rows.append(dict(part="misspecification", level=level,
                                     seed=seed, **r))
                got = {x["method"]: x["ari"] for x in rows[-2:]}
                print(f"  level {level} seed {seed} "
                      f"top1 {rows[-1]['top1']:.3f} " +
                      " ".join(f"{k} {v:.3f}" for k, v in got.items()),
                      flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)

    if want is None or "uneven" in want:
        print("B: uneven preservation", flush=True)
        mixes = {"P4_uniform": ["P4"] * 4,
                 "P3_to_P5": ["P3", "P4", "P4", "P5"],
                 "P2_to_P5": ["P2", "P3", "P4", "P5"]}
        for tag, states in mixes.items():
            for seed in range(a.seeds):
                parts = []
                for k, st in enumerate(states):
                    spec = common.make_spec(
                        common.EVAL_SEED0 + seed * 10 + k,
                        max(1, a.slips // len(states)), st, dep=dep)
                    parts.append(synth.generate_instance(spec))
                inst = merge(parts)
                for r in evaluate(model, inst, cal, cfgs, a.workers):
                    rows.append(dict(part="uneven", mix=tag, seed=seed, **r))
                got = {x["method"]: x["ari"] for x in rows[-2:]}
                print(f"  {tag} seed {seed} " +
                      " ".join(f"{k} {v:.3f}" for k, v in got.items()),
                      flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
