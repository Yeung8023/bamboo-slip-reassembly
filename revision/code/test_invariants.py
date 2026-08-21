"""Checks that have to hold before any revision result is believed.

1. The published behaviour is untouched.  Every new option defaults to what the
   submitted manuscript did, so a corpus and a solution generated with the
   defaults must be identical to the ones the submitted results came from.
2. The decomposition is exact.  Where the monolithic solver proves optimality,
   solving component by component must return the same objective.  This is the
   whole basis of the scalability claim, so it is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import common
from common import assemble, synth


def test_defaults_unchanged(model):
    inst, meta, S, slip_of = common.prep(
        model, common.make_spec(7, 40, "P4"))
    cal = assemble.fit_calibration(S, inst["joins"])
    W = assemble.log_odds(S, cal)
    cfg = assemble.AssemblyConfig(use_latent=True, theta=2.0, time_limit=60)
    prob = assemble.build_problem(meta, W, cfg)
    r = assemble.solve_cpsat(prob)
    e = common.score(len(meta), r["joins"], inst["joins"], slip_of)
    print(f"  legacy path: {len(meta)} frags, {len(prob['arcs'])} arcs, "
          f"obj {r['objective']:.3f} {r['status']}, "
          f"F1 {e['f1']:.3f} ARI {e['ari']:.3f}")
    assert r["status"] == "OPTIMAL"
    return prob, r


def test_decomposition_exact(prob, ref):
    d = assemble.solve_decomposed(prob, time_limit=120, workers=8)
    print(f"  decomposed: {d['n_components']} components, largest "
          f"{d['max_component']}, obj {d['objective']:.3f} {d['status']}, "
          f"{d['wall']:.2f}s")
    assert d["status"] == "OPTIMAL", d["status"]
    assert abs(d["objective"] - ref["objective"]) < 1e-3, \
        (d["objective"], ref["objective"])
    assert set(map(tuple, d["joins"])) == set(map(tuple, ref["joins"]))
    print("  objective and join set identical to the monolithic solve")


def test_spatial_context(model):
    """The spatial deposition model must change only the excavation record."""
    a = synth.generate_instance(common.make_spec(3, 30, "P4"))
    b = synth.generate_instance(common.make_spec(
        3, 30, "P4", dep=synth.DepositionSpec(model="spatial")))
    assert len(a["frags"]) == len(b["frags"])
    assert sorted(map(tuple, a["joins"])) == sorted(map(tuple, b["joins"]))
    for f, g in zip(a["frags"], b["frags"]):
        assert np.allclose(f["top_prof"], g["top_prof"])
        assert f["obs_width_mm"] == g["obs_width_mm"]
    ca = common.coherence([{"unit": f["unit"]} for f in a["frags"]], a["joins"])
    cb = common.coherence([{"unit": f["unit"]} for f in b["frags"]], b["joins"])
    print(f"  imagery identical; context coherence uniform {ca:.3f} "
          f"-> spatial {cb:.3f}")
    assert ca == 1.0 and cb < 1.0


def test_decomposition_exact_soft(model):
    """Same check with the soft context term and the roll-pooled length on."""
    dep = synth.DepositionSpec(model="spatial", roll_len_sigma_mm=2.0,
                               slips_per_roll=20)
    inst, meta, S, slip_of = common.prep(model, common.make_spec(7, 40, "P4", dep=dep))
    cal = assemble.fit_calibration(S, inst["joins"])
    lr = assemble.fit_context_lr(meta, inst["joins"])
    W = assemble.log_odds(S, cal)
    cfg = assemble.AssemblyConfig(use_latent=True, theta=2.0, time_limit=120,
                                  strat_mode="soft", strat_lr=lr,
                                  use_roll_length=True, notch_order=True)
    prob = assemble.build_problem(meta, W, cfg)
    ref = assemble.solve_cpsat(prob)
    d = assemble.solve_decomposed(prob, time_limit=120, workers=8)
    print(f"  soft+roll: monolithic obj {ref['objective']:.3f} {ref['status']}, "
          f"decomposed {d['objective']:.3f} {d['status']} in "
          f"{d['n_components']} components (largest {d['max_component']})")
    assert ref["status"] == "OPTIMAL" and d["status"] == "OPTIMAL"
    assert abs(d["objective"] - ref["objective"]) < 1e-3
    assert set(map(tuple, d["joins"])) == set(map(tuple, ref["joins"]))
    print("  exact under the pooled length as well")


if __name__ == "__main__":
    model = common.load_matcher()
    print("published defaults")
    prob, ref = test_defaults_unchanged(model)
    print("decomposition, published model")
    test_decomposition_exact(prob, ref)
    print("spatial deposition")
    test_spatial_context(model)
    print("decomposition, revised model")
    test_decomposition_exact_soft(model)
    print("\nall invariants hold")
