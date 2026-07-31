"""Global consistent assembly.

A pairwise matcher returns, for every fragment, a ranked list of candidate
partners.  What a conservator needs is not that list but a *reconstruction*: a
set of chains, each of which is a physically admissible partial slip.  Turning
one into the other is a combinatorial problem, not a ranking problem, because
the candidates conflict -- two fragments may both claim the same partner, a
chain may grow past the length of a real slip, or it may require the binding
notches to sit somewhere no cord ever ran.

We state it as a maximum-weight path cover on the fragment graph subject to the
constraints a real corpus obeys:

  uniqueness    each fracture end joins at most one other end
  linearity     a slip is a chain, never a cycle or a branch
  length        an assembled chain cannot exceed the length of a slip
  morphometry   slip width is constant along a slip
  notches       binding notches sit at the heights where cords actually ran
  hand          one slip is written by one scribe
  stratigraphy  fragments from incompatible excavation units cannot conjoin

The first three are structural and are imposed exactly.  Morphometry and
stratigraphy prune the arc set.  Hand attribution is noisy, so it enters as
calibrated evidence in the objective rather than as a hard rule.  The notch
constraint is the interesting one: it is *absolute* positional information that
no pairwise scorer can use, because it only becomes a constraint once a
fragment has been assigned a position within a slip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MM = 10  # integer resolution: tenths of a millimetre


@dataclass
class AssemblyConfig:
    top_k: int = 50            # candidate arcs kept per fracture end
    width_tol_mm: float = 0.45  # morphometric gate
    notch_tol_mm: float = 16.0  # cord-height window half-width
    notch_penalty: float = 1.0  # cost of leaving an observed notch unexplained
    max_slip_mm: float = 285.0
    use_uniqueness: bool = True
    use_length: bool = True
    use_width: bool = True
    use_notch: bool = True
    use_hand: bool = True
    use_strat: bool = True
    # Latent per-slip variables (length, width, scribal hand) propagated along
    # each chain, instead of pairwise proxies for them.
    use_latent: bool = False
    width_meas_mm: float = 0.30  # tolerance between latent and observed width
    hand_accuracy: float = 0.80
    n_scribes: int = 6
    cord_fracs: tuple = (0.06, 0.5, 0.94)
    slip_len_mm: tuple = (231.0, 278.0)
    theta: float = 0.0         # cost of claiming a join; tuned on calibration
    time_limit: float = 60.0
    workers: int = 12


# --------------------------------------------------------------------------
# score calibration
# --------------------------------------------------------------------------

def fit_calibration(S, joins, n_neg=200000, seed=0):
    """Platt scaling of raw matcher scores into log-odds of a true conjoin.

    Fitted on a calibration corpus with known ground truth and at the *natural*
    class ratio, so the resulting log-odds carry the right prior: in a corpus of
    n fragments almost every pair is a non-join, and an objective built on
    posteriors has to know that.
    """
    rng = np.random.default_rng(seed)
    joins = np.asarray(joins).reshape(-1, 2)
    if len(joins) == 0:
        return 1.0, 0.0
    pos = S[joins[:, 0], joins[:, 1]]
    n = S.shape[0]
    ii = rng.integers(0, n, n_neg)
    jj = rng.integers(0, n, n_neg)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    jset = set(map(tuple, joins.tolist()))
    m = np.array([(int(a), int(b)) not in jset for a, b in zip(ii, jj)])
    neg = S[ii[m], jj[m]]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    x = np.concatenate([pos, neg])[:, None]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    w = np.concatenate([np.full(len(pos), len(neg) / max(len(pos), 1)),
                        np.ones(len(neg))])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e3, max_iter=1000)
    lr.fit(x, y, sample_weight=w)
    # Class-balanced fit, so this is the *likelihood ratio* of the score under
    # match vs non-match, carrying no corpus prior.  The prior belongs in the
    # structure, not in the arc weight: the uniqueness constraint already says
    # that an end has at most one partner, and the residual reluctance to claim
    # a join at all is the single tunable threshold theta, fitted on
    # calibration corpora in tune_theta().  Folding log(n) into the weight
    # instead would drive every arc negative and empty the problem.
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def log_odds(S, cal):
    a, b = cal
    return a * S + b


# --------------------------------------------------------------------------
# problem construction
# --------------------------------------------------------------------------

def build_problem(meta, W, cfg: AssemblyConfig):
    """Select candidate arcs and their weights.

    ``W`` is the calibrated log-odds matrix: W[i, j] is the evidence that
    fragment i's bottom conjoins fragment j's top.
    """
    n = len(meta)
    height = np.array([m["height_mm"] for m in meta], np.float64)
    width = np.array([m["obs_width_mm"] for m in meta], np.float64)
    hand = np.array([m["obs_hand"] for m in meta], np.int64)
    unit = np.array([m["unit"] for m in meta], np.int64)

    Wm = W.copy()
    np.fill_diagonal(Wm, -np.inf)

    # hard gates: these remove arcs no admissible reconstruction could use
    if cfg.use_width:
        bad = np.abs(width[:, None] - width[None, :]) > cfg.width_tol_mm
        Wm[bad] = -np.inf
    if cfg.use_strat:
        Wm[unit[:, None] != unit[None, :]] = -np.inf
    # a chain of two fragments must still fit inside one slip
    if cfg.use_length:
        Wm[height[:, None] + height[None, :] > cfg.max_slip_mm] = -np.inf

    # scribal hand is noisy evidence, not a gate: fold the likelihood ratio for
    # "same slip" given agreement/disagreement of the attribution into the score
    if cfg.use_hand and not cfg.use_latent:
        # With a latent per-slip hand this term would double-count the same
        # weak evidence once per arc along the chain, so it is applied here
        # only in the pairwise formulation.
        p, K = cfg.hand_accuracy, cfg.n_scribes
        p_agree_same = p * p + (1 - p) * (1 - p) / (K - 1)
        p_agree_diff = 1.0 / K
        lr_agree = np.log(p_agree_same / p_agree_diff)
        lr_disagree = np.log((1 - p_agree_same) / (1 - p_agree_diff))
        agree = hand[:, None] == hand[None, :]
        Wm = Wm + np.where(agree, lr_agree, lr_disagree)

    # sparsify: keep the top-k candidates for each fragment's bottom end
    k = min(cfg.top_k, n - 1)
    arcs, wts = [], []
    order = np.argpartition(-Wm, kth=k - 1, axis=1)[:, :k] if k < n else \
        np.tile(np.arange(n), (n, 1))
    for i in range(n):
        for j in order[i]:
            v = Wm[i, j] - cfg.theta
            if np.isfinite(v) and v > 0:
                arcs.append((i, int(j)))
                wts.append(float(v))
    return dict(n=n, arcs=arcs, weights=np.array(wts), height=height,
                width=width, hand=hand, unit=unit, meta=meta, cfg=cfg)


# --------------------------------------------------------------------------
# exact solver
# --------------------------------------------------------------------------

def solve_cpsat(prob, time_limit=None, workers=None, log=False, trace=False):
    """Exact (or best-found) maximum-weight consistent path cover via CP-SAT."""
    from ortools.sat.python import cp_model

    cfg = prob["cfg"]
    n, arcs, wts = prob["n"], prob["arcs"], prob["weights"]
    height, meta = prob["height"], prob["meta"]
    if not arcs:
        return dict(joins=[], objective=0.0, status="EMPTY")

    m = cp_model.CpModel()
    x = {a: m.NewBoolVar(f"x{a}") for a in arcs}
    notch_slack = []

    if cfg.use_uniqueness:
        out_, in_ = {}, {}
        for (i, j) in arcs:
            out_.setdefault(i, []).append(x[(i, j)])
            in_.setdefault(j, []).append(x[(i, j)])
        for v in out_.values():
            m.AddAtMostOne(v)
        for v in in_.values():
            m.AddAtMostOne(v)

    if cfg.use_length or cfg.use_notch:
        # pos[i] = distance in 0.1 mm from the top of the reconstructed slip to
        # the top of fragment i.  Because an arc forces pos to increase by the
        # predecessor's own height, this single variable enforces the linear
        # order, forbids cycles, and caps the assembled length at once.
        Lmax = int(round(cfg.max_slip_mm * MM))
        pos = [m.NewIntVar(0, max(0, Lmax - int(round(height[i] * MM))), f"p{i}")
               for i in range(n)]
        for (i, j) in arcs:
            hi = int(round(height[i] * MM))
            m.Add(pos[j] == pos[i] + hi).OnlyEnforceIf(x[(i, j)])

        lo_len, hi_len = cfg.slip_len_mm

        if cfg.use_latent:
            # A slip has one length, and every fragment of it shares that
            # length.  Making it an explicit latent variable, propagated along
            # the chain, is what turns the binding notches into real evidence:
            # anchored instead to corpus-wide windows, the 231-278 mm spread of
            # slip lengths widens the admissible band for the middle cord to
            # roughly +-23 mm, so the constraint almost never binds.  Tying the
            # notches on *different* fragments of one chain to a single shared
            # length is the information that anchoring to the corpus throws away.
            Lv = [m.NewIntVar(int(round(lo_len * MM)), int(round(hi_len * MM)),
                              f"L{i}") for i in range(n)]
            for (i, j) in arcs:
                m.Add(Lv[j] == Lv[i]).OnlyEnforceIf(x[(i, j)])
            # the assembled chain must fit inside *its own* slip, not merely
            # inside the longest slip in the corpus
            for i in range(n):
                m.Add(pos[i] + int(round(height[i] * MM)) <= Lv[i])

        if cfg.use_notch:
            tol = int(round(cfg.notch_tol_mm * MM))
            fr100 = [int(round(f * 100)) for f in cfg.cord_fracs]
            windows = [(int(round((lo_len * f - cfg.notch_tol_mm) * MM)),
                        int(round((hi_len * f + cfg.notch_tol_mm) * MM)))
                       for f in cfg.cord_fracs]
            for i in range(n):
                for nu in meta[i].get("notches_mm", []):
                    off = int(round(nu * MM))
                    sel = []
                    for c in range(len(cfg.cord_fracs)):
                        b = m.NewBoolVar("")
                        if cfg.use_latent:
                            # notch sits at fraction f of *this slip's* length
                            m.Add(100 * (pos[i] + off) - fr100[c] * Lv[i]
                                  >= -100 * tol).OnlyEnforceIf(b)
                            m.Add(100 * (pos[i] + off) - fr100[c] * Lv[i]
                                  <= 100 * tol).OnlyEnforceIf(b)
                        else:
                            w0, w1 = windows[c]
                            m.Add(pos[i] + off >= w0).OnlyEnforceIf(b)
                            m.Add(pos[i] + off <= w1).OnlyEnforceIf(b)
                        sel.append(b)
                    if sel:
                        # A notch that cannot be placed must cost something
                        # rather than making the corpus unreconstructable.  Cord
                        # jitter and the corpus's own length variation mean a
                        # genuine notch occasionally falls outside every window;
                        # as a hard requirement this renders the whole instance
                        # infeasible, which is both wrong and, in a sensitivity
                        # sweep over the tolerance, badly misleading.
                        unplaced = m.NewBoolVar("")
                        sel.append(unplaced)
                        notch_slack.append(unplaced)
                        m.AddExactlyOne(sel)

    scale = 1000
    obj = sum(int(round(w * scale)) * x[a] for a, w in zip(arcs, wts))
    if notch_slack:
        obj -= int(round(cfg.notch_penalty * scale)) * sum(notch_slack)

    if cfg.use_latent:
        # Latent width.  A pairwise gate lets width drift along a chain: with a
        # tolerance of tau between neighbours, a four-fragment chain can span
        # 3*tau end to end.  One shared latent width per slip removes the drift.
        width = prob["width"]
        wlo = int(round((float(np.min(width)) - 1.0) * 100))
        whi = int(round((float(np.max(width)) + 1.0) * 100))
        Wv = [m.NewIntVar(wlo, whi, f"w{i}") for i in range(n)]
        tolw = int(round(cfg.width_meas_mm * 100))
        for i in range(n):
            oi = int(round(float(width[i]) * 100))
            m.Add(Wv[i] >= oi - tolw)
            m.Add(Wv[i] <= oi + tolw)
        for (i, j) in arcs:
            m.Add(Wv[j] == Wv[i]).OnlyEnforceIf(x[(i, j)])

    if cfg.use_latent and cfg.use_hand:
        # Latent scribal hand.  One slip is written by one scribe; the
        # attributions we observe are noisy readings of that single hand.
        # Treating agreement pairwise instead -- as a bonus on each arc --
        # double-counts the same weak evidence along a chain and, at the poorly
        # preserved end, does more harm than good.
        hand = prob["hand"]
        K = max(int(cfg.n_scribes), int(hand.max()) + 1)
        Hv = [m.NewIntVar(0, K - 1, f"h{i}") for i in range(n)]
        for (i, j) in arcs:
            m.Add(Hv[j] == Hv[i]).OnlyEnforceIf(x[(i, j)])
        p = float(cfg.hand_accuracy)
        llr = np.log(p * (K - 1) / max(1.0 - p, 1e-6))  # match vs mismatch
        gain = int(round(float(llr) * scale))
        agree_terms = []
        for i in range(n):
            b = m.NewBoolVar("")
            m.Add(Hv[i] == int(hand[i])).OnlyEnforceIf(b)
            m.Add(Hv[i] != int(hand[i])).OnlyEnforceIf(b.Not())
            agree_terms.append(b)
        obj += gain * sum(agree_terms)

    m.Maximize(obj)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit or cfg.time_limit)
    s.parameters.num_workers = int(workers or cfg.workers)
    s.parameters.log_search_progress = log
    cb = None
    if trace:
        class _Trace(cp_model.CpSolverSolutionCallback):
            """Record the incumbent and the bound as the search proceeds, so
            that convergence can be reported rather than asserted."""

            def __init__(self):
                super().__init__()
                self.rows = []

            def on_solution_callback(self):
                self.rows.append((self.WallTime(),
                                  self.ObjectiveValue() / scale,
                                  self.BestObjectiveBound() / scale))

        cb = _Trace()
        st = s.Solve(m, cb)
    else:
        st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(joins=[], objective=0.0, status=s.StatusName(st))
    sel = [a for a in arcs if s.Value(x[a])]
    out = dict(joins=sel, objective=s.ObjectiveValue() / scale,
               status=s.StatusName(st), wall=s.WallTime(),
               bound=s.BestObjectiveBound() / scale,
               gap=(s.BestObjectiveBound() - s.ObjectiveValue()) / max(abs(s.ObjectiveValue()), 1e-9))
    if trace and cb is not None:
        out["trace"] = cb.rows
    return out


# --------------------------------------------------------------------------
# chains
# --------------------------------------------------------------------------

DEFAULT_P_GRID = (0.30, 0.50, 0.70, 0.85, 0.95, 0.99, 0.997, 0.9997)


def theta_grid(ps=DEFAULT_P_GRID):
    """Candidate thresholds, expressed as posterior confidence rather than as
    raw logits.

    The calibration slope varies by almost an order of magnitude across the
    preservation ladder -- a well-preserved corpus separates matches from
    non-matches so sharply that the fitted logits are huge -- so a grid fixed
    in logit units would mean something different at every rung, and would
    silently truncate the search for whichever method prefers a high
    threshold.  Posterior confidence is comparable across the whole ladder.
    """
    p = np.asarray(ps, float)
    return np.log(p / (1.0 - p))


def tune_theta(meta_list, W_list, truth_list, cfg, grid=None, workers=8,
               objective="f1"):
    """Choose the join-claiming threshold on calibration corpora.

    Tuned separately for every method, on corpora disjoint from the test set,
    so that each is evaluated at its own best operating point and none is
    handicapped by a threshold chosen to suit another.
    """
    import metrics
    from dataclasses import replace

    if grid is None:
        grid = theta_grid()
    best, best_v = 0.0, -np.inf
    for th in grid:
        vals = []
        for meta, W, (tj, slip_of) in zip(meta_list, W_list, truth_list):
            c = replace(cfg, theta=float(th))
            prob = build_problem(meta, W, c)
            r = solve_cpsat(prob, workers=workers)
            e = metrics.evaluate(len(meta), r["joins"], tj, slip_of)
            v = e[objective]
            vals.append(0.0 if not np.isfinite(v) else v)
        mv = float(np.mean(vals))
        if mv > best_v:
            best_v, best = mv, float(th)
    return best, best_v


def chains_from_joins(n, joins):
    """Recover the reconstructed slips (maximal chains) from selected arcs."""
    nxt = {i: j for i, j in joins}
    prv = {j: i for i, j in joins}
    heads = [i for i in range(n) if i not in prv]
    out, seen = [], set()
    for h in heads:
        c, cur = [], h
        while cur is not None and cur not in seen:
            c.append(cur); seen.add(cur)
            cur = nxt.get(cur)
        out.append(c)
    for i in range(n):  # any fragment left in a cycle (should not happen)
        if i not in seen:
            c, cur = [], i
            while cur is not None and cur not in seen:
                c.append(cur); seen.add(cur)
                cur = nxt.get(cur)
            out.append(c)
    return out
