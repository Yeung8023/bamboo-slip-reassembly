"""Evaluation.

Three questions are asked of every method, and they are not interchangeable:

  Are the individual joins right?     precision / recall / F1 over conjoins
  Are the slips right?                adjusted Rand index of the induced
                                      partition of fragments into slips
  Is the output usable as it stands?  the fraction of fragments delivered in a
                                      *completely correct* reconstructed slip,
                                      and the fraction of true slips recovered
                                      exactly end to end

The last is the one that matters to a conservator.  A reconstruction that is
95% correct at the level of individual joins can still leave almost every slip
with an error somewhere in it, and a slip with an error in it has to be checked
by hand -- which is the labour the whole exercise is meant to avoid.
"""

from __future__ import annotations

import numpy as np


def _chains(n, joins):
    nxt = {i: j for i, j in joins}
    prv = {j: i for i, j in joins}
    seen, out = set(), []
    for h in range(n):
        if h in prv or h in seen:
            continue
        c, cur = [], h
        while cur is not None and cur not in seen:
            c.append(cur); seen.add(cur); cur = nxt.get(cur)
        out.append(c)
    for i in range(n):
        if i not in seen:
            c, cur = [], i
            while cur is not None and cur not in seen:
                c.append(cur); seen.add(cur); cur = nxt.get(cur)
            out.append(c)
    return out


def _labels(n, chains):
    lab = np.empty(n, np.int64)
    for k, c in enumerate(chains):
        for i in c:
            lab[i] = k
    return lab


def evaluate(n, pred_joins, true_joins, slip_of=None):
    """Score one reconstruction against ground truth."""
    P = set(map(tuple, pred_joins))
    T = set(map(tuple, np.asarray(true_joins).reshape(-1, 2).tolist()))
    tp = len(P & T)
    prec = tp / len(P) if P else float("nan")
    rec = tp / len(T) if T else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if P and T and (prec + rec) > 0 else 0.0

    pred_c = _chains(n, list(P))
    true_c = _chains(n, list(T))

    # exact end-to-end recovery of a multi-fragment slip
    tset = {tuple(c) for c in true_c if len(c) > 1}
    pset = {tuple(c) for c in pred_c if len(c) > 1}
    exact = len(tset & pset) / len(tset) if tset else float("nan")

    # fragments delivered inside a wholly correct chain
    good = set()
    for c in pred_c:
        if tuple(c) in tset or (len(c) == 1 and tuple(c) in {tuple(t) for t in true_c if len(t) == 1}):
            good.update(c)
    clean = len(good) / n

    out = dict(precision=prec, recall=rec, f1=f1, exact_slip=exact,
               clean_fragments=clean, n_pred=len(P), n_true=len(T))

    if slip_of is not None:
        from sklearn.metrics import adjusted_rand_score
        s = np.asarray(slip_of)
        out["ari"] = float(adjusted_rand_score(s, _labels(n, pred_c)))
        # Not every wrong join is equally wrong.  Material lost at a break
        # leaves two fragments of one slip that no longer physically conjoin;
        # linking them is an error of adjacency but places the fragment in the
        # right slip, and a conservator loses little.  Joining fragments of two
        # *different* slips is the damaging error, because it fuses two
        # reconstructions and corrupts both.  These are reported separately.
        if P:
            same = [s[a] == s[b] for a, b in P]
            out["same_slip_precision"] = float(np.mean(same))
            out["cross_slip_joins"] = int(len(P) - int(np.sum(same)))
            out["cross_slip_rate"] = float(1.0 - np.mean(same))
        else:
            out["same_slip_precision"] = float("nan")
            out["cross_slip_joins"] = 0
            out["cross_slip_rate"] = float("nan")
    return out


def ground_truth_chains(n, true_joins):
    return _chains(n, np.asarray(true_joins).reshape(-1, 2).tolist())
