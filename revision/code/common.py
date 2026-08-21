"""Shared setup for the revision experiments.

Everything here loads the *published* matcher and the *published* calibration
protocol, so that the new experiments differ from the submitted ones only in
what is being tested.  Nothing is refitted on an evaluation corpus.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import assemble          # noqa: E402
import matcher as M      # noqa: E402
import metrics           # noqa: E402
import preservation as pres  # noqa: E402
import synth             # noqa: E402

CAL_SEED0 = 900000
EVAL_SEED0 = 0
MODEL = ROOT / "data" / "matcher.pt"
RESULTS = ROOT / "revision" / "results"
FIGS = ROOT / "revision" / "figs"


def load_matcher():
    return M.load_model(str(MODEL))


def descs(inst):
    F = inst["frags"]
    return dict(
        bot_prof=np.array([f["bot_prof"] for f in F]),
        bot_patch=np.array([f["bot_patch"] for f in F]),
        top_prof=np.array([f["top_prof"] for f in F]),
        top_patch=np.array([f["top_patch"] for f in F]),
    )


def make_spec(seed, n_slips, state, dep=None, obs=None, damage=None):
    return synth.InstanceSpec(
        n_slips=n_slips, seed=seed,
        damage=damage if damage is not None else pres.damage(state),
        obs=obs if obs is not None else synth.ObservationSpec(),
        dep=dep if dep is not None else synth.DepositionSpec())


def prep(model, spec):
    """Generate a corpus, score every candidate join, strip the imagery."""
    inst = synth.generate_instance(spec)
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = M.score_matrix(model, descs(inst))
    slip_of = [f["slip_id"] for f in inst["frags"]]
    return inst, meta, S, slip_of


def coherence(meta, joins):
    """Fraction of true joins whose two fragments carry the same context record.

    This is the number that makes a stratigraphic ablation interpretable: it
    says how much of the evidence survived the way the fragments dispersed and
    the way the site was subdivided, rather than assuming the record is a clean
    partition of the corpus.
    """
    u = np.array([m["unit"] for m in meta])
    j = np.asarray(joins).reshape(-1, 2)
    if len(j) == 0:
        return float("nan")
    a, b = u[j[:, 0]], u[j[:, 1]]
    # A pair in which either fragment has no record carries no context, so it
    # counts as context that did not survive rather than as context that
    # agreed. Counting two unrecorded fragments as a match would make an
    # incomplete excavation log look like a coherent one.
    return float(np.mean((a == b) & (a >= 0) & (b >= 0)))


def calibrate(model, state, n_slips, dep=None, obs=None, damage=None,
              radius=1, seed=CAL_SEED0):
    """Platt scaling and contextual log-odds, on a dedicated calibration corpus."""
    spec = make_spec(seed, n_slips, state, dep, obs, damage)
    inst, meta, S, _ = prep(model, spec)
    cal = assemble.fit_calibration(S, inst["joins"])
    lr = assemble.fit_context_lr(meta, inst["joins"], radius=radius)
    return cal, lr


def tune_corpus(model, state, n_slips, cal, dep=None, obs=None, damage=None,
                n_cal=1, seed=CAL_SEED0 + 100):
    """Build the calibration corpora once, so every method is tuned on the same
    material and the cost of tuning does not grow with the number of methods."""
    metas, Ws, truths = [], [], []
    for r in range(n_cal):
        spec = make_spec(seed + r, n_slips, state, dep, obs, damage)
        inst, meta, S, slip_of = prep(model, spec)
        metas.append(meta)
        Ws.append(assemble.log_odds(S, cal))
        truths.append((inst["joins"], slip_of))
    return metas, Ws, truths


def tune_on(corpus, cfg, workers=8, wide=False):
    """Choose theta for one method on a prepared set of calibration corpora."""
    grid = assemble.theta_grid(assemble.WIDE_P_GRID if wide
                               else assemble.DEFAULT_P_GRID)
    metas, Ws, truths = corpus
    th, v = assemble.tune_theta(metas, Ws, truths, cfg, grid=grid,
                                workers=workers)
    at_edge = bool(abs(th - grid[0]) < 1e-9 or abs(th - grid[-1]) < 1e-9)
    return th, at_edge


def tune(model, state, n_slips, cfg, dep=None, obs=None, damage=None,
         cal=None, n_cal=1, workers=8, seed=CAL_SEED0 + 100, wide=False):
    """Choose theta for one method on calibration corpora, as in the paper.

    Every method is tuned on the same grid and the same corpora. The graded
    context term is referenced to the same-square case precisely so that it
    does not shift a method along the grid, which would turn a comparison of
    models into a comparison of grids. ``at_edge`` reports whether the choice
    landed on an endpoint, which is the signal that the grid was too narrow.
    """
    corpus = tune_corpus(model, state, n_slips, cal, dep, obs, damage,
                         n_cal, seed)
    return tune_on(corpus, cfg, workers=workers, wide=wide)


def score(n, joins, true_joins, slip_of):
    return metrics.evaluate(n, joins, true_joins, slip_of)


def cfg_for(base, **kw):
    return replace(base, **kw)


def choose_radius(meta, joins, target=0.98, cap=4):
    """Smallest excavation radius that still admits the true joins.

    How far apart two fragments of one slip end up depends on how far material
    dispersed and how finely the site was gridded, and neither is known in
    advance. Rather than fixing the radius, we read it off a calibration corpus
    as the smallest one that keeps ``target`` of the true joins reachable. A
    site whose fragments barely moved gets a tight radius and a strong
    constraint; a site whose fragments scattered gets a loose one, and the
    graded weights inside it carry what is left of the evidence.
    """
    d = assemble.block_distance(meta)
    j = np.asarray(joins).reshape(-1, 2)
    if len(j) == 0:
        return 1
    dj = d[j[:, 0], j[:, 1]]
    known = dj >= 0
    if known.sum() == 0:
        return 1
    for r in range(cap + 1):
        if np.mean(dj[known] <= r) >= target:
            return r
    return cap


def arc_recall(prob, true_joins):
    """Fraction of the true joins that survived into the candidate set.

    A hard gate can only help if it deletes more impostors than truth. Once
    fragments disperse it starts deleting truth, and this is where that shows
    up: no solver can recover a join whose arc was removed before the search
    began.
    """
    A = set(map(tuple, prob["arcs"]))
    t = [tuple(map(int, x)) for x in np.asarray(true_joins).reshape(-1, 2)]
    return float(np.mean([x in A for x in t])) if t else float("nan")


def load_dispersion():
    """The dispersion results, with the record sweep merged in.

    The record sweep was rerun after the excavation gate was corrected to stop
    excluding fragments that carry no context record, so it lives in its own
    file and supersedes anything the first pass wrote.
    """
    import pandas as pd
    d = pd.read_csv(RESULTS / "dispersion.csv")
    rec = RESULTS / "dispersion_record.csv"
    if rec.exists():
        d = pd.concat([d[d.tag != "record"], pd.read_csv(rec)],
                      ignore_index=True)
    return d
