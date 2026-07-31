"""What is done now, and the strongest things that could be done instead.

These are the alternatives the global layer has to beat.  Two of them --
mutual-best and bipartite matching -- are genuinely global in the sense that
they resolve competition between candidates; what they cannot do is reason
about whether the resulting chain is a physically possible slip.  The expert
review baseline stands in for current practice, in which a specialist works
down a ranked candidate list by hand.
"""

from __future__ import annotations

import numpy as np


def top1(W, cfg=None):
    """Take each fragment's single best candidate.  Conflicts are left in."""
    n = W.shape[0]
    Wm = np.where(np.isfinite(W), W, -np.inf)
    j = np.argmax(Wm, axis=1)
    return [(i, int(j[i])) for i in range(n) if np.isfinite(Wm[i, j[i]])
            and Wm[i, j[i]] > 0]


def threshold(W, thr=0.0):
    """Every candidate above a score threshold."""
    ii, jj = np.where(np.isfinite(W) & (W > thr))
    return [(int(a), int(b)) for a, b in zip(ii, jj) if a != b]


def mutual_best(W):
    """Accept i->j only when j's best incoming candidate is also i.

    The standard practical heuristic, and a surprisingly strong one: it buys
    most of the precision that a full matching buys, for no solver at all.
    """
    Wm = np.where(np.isfinite(W), W, -np.inf)
    n = Wm.shape[0]
    best_out = np.argmax(Wm, axis=1)
    best_in = np.argmax(Wm, axis=0)
    return [(i, int(best_out[i])) for i in range(n)
            if best_in[best_out[i]] == i and Wm[i, best_out[i]] > 0]


def matching_only(meta, W, cfg):
    """Maximum-weight bipartite matching of bottom ends to top ends.

    The strongest baseline that does not model the slip: it resolves every
    conflict between competing candidates globally and optimally, but knows
    nothing about length, morphometry or where the binding cords ran.  The gap
    between this and the full model is precisely the value of knowing what a
    slip is.

    It is solved with the same solver on the same arc set as the full model,
    with every constraint but uniqueness switched off, so the comparison is a
    clean ablation rather than a comparison of two different implementations.
    A dense Hungarian solver would be equivalent but cannot reach the corpus
    sizes here -- it is cubic in the fragment count.
    """
    import assemble
    from dataclasses import replace

    c = replace(cfg, use_uniqueness=True, use_length=False, use_width=False,
                use_notch=False, use_hand=False, use_strat=False)
    prob = assemble.build_problem(meta, W, c)
    return assemble.solve_cpsat(prob)["joins"]


def expert_topk(W, true_joins, k=50):
    """Current practice: a specialist works down each ranked candidate list.

    We give it an oracle -- if the correct partner appears anywhere in the top
    k, the expert finds it and makes no mistakes.  This is an upper bound on
    any workflow whose output is a ranked list, and it is charged its true
    cost: the number of candidate pairs a human has to look at.
    """
    n = W.shape[0]
    Wm = np.where(np.isfinite(W), W, -np.inf)
    kk = min(k, n - 1)
    order = np.argpartition(-Wm, kth=kk - 1, axis=1)[:, :kk]
    truth = {}
    for a, b in np.asarray(true_joins).reshape(-1, 2).tolist():
        truth[int(a)] = int(b)
    out = []
    for i in range(n):
        t = truth.get(i)
        if t is not None and t in set(order[i].tolist()):
            out.append((i, t))
    return out, n * kk


def expert_review_cost(n, k):
    return n * k
