"""Print the numbers the revised manuscript and the response letter quote.

Run at any point; each block reports only what has finished.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import common

R = common.RESULTS
pd.set_option("display.width", 200)


def head(t):
    print(f"\n{'='*74}\n{t}\n{'='*74}")


def dispersion():
    if not (R / "dispersion.csv").exists():
        return
    d = common.load_dispersion()
    head("E1  excavation context under the deposition model")
    p = d.pivot_table(index=["tag", "disperse_sigma_mm", "block_mm",
                             "unit_known_frac", "unit_error_frac"],
                      columns="method",
                      values=["coherence", "arc_recall", "ari",
                              "cross_slip_rate", "exact_slip"])
    print(p.round(3).to_string())
    print("\nradius chosen per setting:")
    print(d.groupby(["tag", "disperse_sigma_mm", "block_mm",
                     "unit_known_frac", "unit_error_frac"])["radius"]
          .first().to_string())


def notch():
    f = R / "notch.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    head("E2  binding notches and the pooled slip length")
    for tag in d.tag.unique():
        sub = d[d.tag == tag]
        base = sub[sub.case == "no_notch"]["ari"].mean()
        g = sub.groupby("case")[["ari", "cross_slip_rate", "exact_slip",
                                 "f1"]].mean()
        g["delta_ari"] = g["ari"] - base
        g["ari_sd"] = sub.groupby("case")["ari"].std()
        print(f"\n[{tag}]  n seeds = {sub.seed.nunique()}")
        print(g.round(3).to_string())


def baselines():
    f = R / "baselines.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    head("E4  global reassembly baselines")
    print(d.pivot_table(index="method", columns="state",
                        values="ari").round(3).to_string())
    print("\ncross-slip rate")
    print(d.pivot_table(index="method", columns="state",
                        values="cross_slip_rate").round(3).to_string())


def decomposition():
    f = R / "decomposition.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    head("E3  decomposition")
    g = d.groupby(["tag", "n_frag", "solver"]).agg(
        time=("solve_s", "mean"), ari=("ari", "mean"),
        proven=("status", lambda x: float(np.mean(x == "OPTIMAL"))),
        comps=("n_components", "mean"), biggest=("max_component", "mean"),
        agree=("agree", "mean"))
    print(g.round(3).to_string())


def realism():
    f = R / "realism.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    u = R / "realism_uneven.csv"
    if u.exists():
        d = pd.concat([d[d.part != "uneven"], pd.read_csv(u)],
                      ignore_index=True)
    head("E6  misspecification and uneven preservation")
    a = d[d.part == "misspecification"]
    print(a.pivot_table(index="level", columns="method",
                        values=["top1", "ari", "cross_slip_rate"])
          .round(3).to_string())
    b = d[d.part == "uneven"]
    if len(b):
        print("\nuneven preservation")
        print(b.pivot_table(index="mix", columns="method",
                            values=["top1", "ari"]).round(3).to_string())


def real():
    f = R / "real.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    head("E5  photographed slips against rendered bamboo")
    print(d.pivot_table(index=["substrate", "method"],
                        values=["top1", "top5", "top50", "f1", "ari",
                                "exact_slip", "cross_slip_rate"])
          .round(3).to_string())


if __name__ == "__main__":
    dispersion()
    notch()
    baselines()
    decomposition()
    realism()
    real()
