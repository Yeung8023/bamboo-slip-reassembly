"""Global reassembly methods from the puzzle-solving literature, in 1D.

Reviewer 2 is right that maximum-weight bipartite matching is not the only
global method available, and that without the others it is unclear how much of
the gain comes from modelling the object and how much from any global reasoning
at all. Three further families are implemented here.

Greedy assembly is what most puzzle solvers do first: take the strongest
candidate join that does not contradict what has already been accepted. Loop
consistency is the idea behind loop-constrained jigsaw solving, that the
relative displacements asserted by the accepted joins must agree around every
cycle; for a linear object this becomes one global set of positions that all
accepted joins have to agree on, which is recovered by weighted least squares
and then used to reject the joins that disagree. Shortest-path assembly is the
formulation used for fragment placement over a candidate graph, where a
reconstruction is a minimum-cost path and the corpus is covered by extracting
such paths in turn.

All three consume exactly the arc set and the calibrated weights the
constrained model is given, so the comparison isolates the reasoning and not
the input.
"""

from __future__ import annotations

import numpy as np


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def greedy_chain(prob):
    """Accept joins strongest first, refusing conflicts, branches and cycles.

    The standard greedy of the puzzle literature. It is global in that a join
    can be refused because of a join already accepted elsewhere, but it commits
    irrevocably and never reconsiders.
    """
    arcs, w = prob["arcs"], prob["weights"]
    order = np.argsort(-np.asarray(w))
    dsu = _DSU(prob["n"])
    used_out, used_in, out = set(), set(), []
    for k in order:
        i, j = arcs[int(k)]
        if i in used_out or j in used_in:
            continue
        if not dsu.union(i, j):
            continue
        used_out.add(i)
        used_in.add(j)
        out.append((i, j))
    return out


def loop_consistency(prob, tol_mm=6.0, damping=1e-6):
    """Reject the joins that disagree with a globally consistent set of positions.

    Every candidate join asserts a displacement: the top of the lower fragment
    sits one fragment height below the top of the upper one. Taken together the
    joins over-determine a position for each fragment, and a set of joins is
    loop consistent when those assertions agree around every cycle of the
    candidate graph. Solving the weighted least squares problem for the
    positions and discarding the joins whose residual is large is the linear
    analogue of the loop constraint used in jigsaw solving. What remains is
    resolved greedily, since consistency filters candidates but does not by
    itself produce an assembly.
    """
    n, arcs, w = prob["n"], prob["arcs"], prob["weights"]
    if not arcs:
        return []
    h = prob["height"]
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    m = len(arcs)
    rows = np.repeat(np.arange(m), 2)
    cols = np.array([c for (i, j) in arcs for c in (j, i)])
    vals = np.tile([1.0, -1.0], m)
    sw = np.sqrt(np.maximum(np.asarray(w, float), 1e-6))
    A = coo_matrix((vals * np.repeat(sw, 2), (rows, cols)), shape=(m, n))
    b = np.array([h[i] for (i, j) in arcs]) * sw
    # damping anchors the solution where the candidate graph is disconnected,
    # which it usually is: only relative positions within a component are
    # determined, and the residuals this test uses depend on nothing else
    p = lsqr(A.tocsr(), b, damp=damping)[0]

    res = np.array([abs((p[j] - p[i]) - h[i]) for (i, j) in arcs])
    keep = res <= tol_mm
    sub = dict(prob)
    sub["arcs"] = [a for a, k in zip(arcs, keep) if k]
    sub["weights"] = np.asarray(w)[keep]
    return greedy_chain(sub)


def shortest_path_assembly(prob, max_len=8):
    """Cover the corpus with minimum-cost paths over the candidate graph.

    Fragment placement posed as a path problem: the cost of a join is the
    negative log probability that it is real, a reconstruction is a path, and
    the best reconstruction available is the cheapest path per join it contains.
    Paths are extracted one at a time and their fragments removed, which is how
    a path formulation is turned into a cover of the whole corpus.
    """
    n, arcs, w = prob["n"], prob["arcs"], np.asarray(prob["weights"], float)
    if not arcs:
        return []
    cost = np.log1p(np.exp(-np.clip(w, -30, 30)))   # -log sigmoid(w), positive
    nxt = {}
    for (i, j), c in zip(arcs, cost):
        nxt.setdefault(i, []).append((j, float(c)))

    alive = np.ones(n, bool)
    out = []
    while True:
        # best mean-cost path of at least two fragments, by bounded dynamic
        # programming over path length
        best = None
        # dp[k][i] = (cost, path) of the cheapest path of k arcs starting at i
        dp = {i: (0.0, [i]) for i in range(n) if alive[i]}
        for _ in range(max_len - 1):
            nd = {}
            for i, (c, path) in dp.items():
                for j, cij in nxt.get(i, ()):
                    if not alive[j] or j in path:
                        continue
                    tot = c + cij
                    cand = (tot, path + [j])
                    if j not in nd or tot < nd[j][0]:
                        nd[j] = cand
                    k = len(path)
                    mean = tot / k
                    if best is None or mean < best[0]:
                        best = (mean, path + [j])
            if not nd:
                break
            dp = nd
        if best is None:
            break
        _, path = best
        for a, b in zip(path, path[1:]):
            out.append((a, b))
        for i in path:
            alive[i] = False
    return out
