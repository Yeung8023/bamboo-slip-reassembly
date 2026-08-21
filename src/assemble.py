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
  stratigraphy  fragments recovered far apart cannot conjoin
  roll length   the slips of one manuscript share a standard length

The first three are structural and are imposed exactly.  Morphometry prunes the
arc set.  Hand attribution is noisy, so it enters as calibrated evidence in the
objective rather than as a hard rule.  The notch constraint is the interesting
one: it is *absolute* positional information that no pairwise scorer can use,
because it only becomes a constraint once a fragment has been assigned a
position within a slip.

Excavation context is the one that had to be rethought.  Treated as a gate, as
it was in the first version of this work, it is right only when the record is a
clean partition of the corpus, and once fragments have dispersed it deletes
true joins faster than it deletes impostors.  ``strat_mode="soft"`` replaces the
gate with log-odds graded by how far apart the two recorded squares are, fitted
on the calibration corpus by fit_context_lr(), and says nothing at all about a
fragment recovered without a record.

Two consequences of the model being separable are worth naming.  Every variable
belongs to one fragment, so solve_decomposed() splits the corpus into connected
components of the arc graph and returns the exact optimum rather than a bound.
And because a manuscript was trimmed to one length, pooling the latent length
over the fragments recorded in one excavation square collapses the band around
each binding cord, which is what makes the notch constraint bind at all.
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
    # Excavation context.  "hard" is the published gate: two fragments may join
    # only if their recorded units agree.  That is right when the record is a
    # clean partition and wrong as soon as fragments disperse, because it then
    # deletes true joins outright.  "soft" grades the evidence by how far apart
    # the two squares are, cutting only beyond a radius and staying silent
    # about fragments recovered without a context record.
    strat_mode: str = "hard"       # "hard" | "soft"
    strat_radius: int = 1          # squares; arcs beyond this are cut in soft mode
    strat_lr: tuple = ()           # log-odds by block distance, from fit_context_lr
    # Joint estimate of slip length across the slips of one roll.  A cord sits
    # at a fraction of a length known only to within the range of the corpus,
    # which is 47 mm wide, so the admissible band around each cord is several
    # millimetres and the notch constraint almost never binds.  Slips of one
    # roll were trimmed to one standard length, so pooling the latent length
    # over a roll collapses that band.
    use_roll_length: bool = False
    roll_key: str = "unit"         # "unit" (what an excavation records) or
                                   # "roll" (upper bound, if bundles were kept)
    roll_len_tol_mm: float = 4.0
    # An excavation square is not a manuscript.  A roll of 30 slips is wider
    # than a 250 mm square, so a square routinely holds slips from two rolls,
    # and forcing them all to one length is worse than not pooling at all.
    # The square is therefore given a small number of latent lengths and each
    # slip takes one of them, which is a mixture over the manuscripts a square
    # can contain rather than an assumption that it contains one.
    roll_len_classes: int = 1
    # Notches on one fragment, read from its top downwards, must take cords in
    # the same order.  Without this each notch is placed independently and two
    # of them may claim the same cord.
    notch_order: bool = False
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


def _block_rc(meta):
    """Row and column of the excavation square each fragment came from."""
    rc = np.array([m.get("unit_rc", (m["unit"], 0)) for m in meta], np.int64)
    return rc[:, 0], rc[:, 1]


def block_distance(meta):
    """Chebyshev distance in excavation squares between every pair of fragments.

    Fragments recovered without a context record are marked -1, which every
    caller reads as "this pair carries no contextual evidence" rather than as
    "these fragments were found far apart".
    """
    r, c = _block_rc(meta)
    known = r > -900
    d = np.maximum(np.abs(r[:, None] - r[None, :]),
                   np.abs(c[:, None] - c[None, :]))
    d[~known, :] = -1
    d[:, ~known] = -1
    return d


def fit_context_lr(meta, joins, radius=1, n_neg=200000, seed=0):
    """Log-odds that two fragments belong to one slip, by excavation distance.

    Fitted on a calibration corpus with known ground truth, in the same spirit
    as the Platt scaling of the matcher scores: how informative the excavation
    record is depends on how far fragments dispersed and how finely the site
    was subdivided, and both vary between sites.  Estimating it from the corpus
    itself means the constraint carries the strength the record actually has,
    instead of a strength assumed by the modeller.

    Returns log-odds for distances 0..radius.  Anything beyond the radius is
    cut by the caller, and an unrecorded fragment gets no evidence either way.
    """
    rng = np.random.default_rng(seed)
    d = block_distance(meta)
    joins = np.asarray(joins).reshape(-1, 2)
    if len(joins) == 0:
        return tuple(0.0 for _ in range(radius + 1))
    pos = d[joins[:, 0], joins[:, 1]]
    n = len(meta)
    ii, jj = rng.integers(0, n, n_neg), rng.integers(0, n, n_neg)
    keep = ii != jj
    neg = d[ii[keep], jj[keep]]

    out = []
    for k in range(radius + 1):
        p = (np.sum(pos == k) + 1.0) / (np.sum(pos >= 0) + radius + 1.0)
        q = (np.sum(neg == k) + 1.0) / (np.sum(neg >= 0) + radius + 1.0)
        out.append(float(np.log(p / q)))
    # Referenced to the same-square case, which contributes nothing.  Only the
    # differences between distances are evidence; the part common to every arc
    # is exactly a reparameterisation of the join-claiming threshold.  Leaving
    # it in would put the graded variant on a different part of the threshold
    # grid from the gated one and turn a comparison of models into a comparison
    # of grids.
    return tuple(float(x - out[0]) for x in out)


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
    if cfg.use_strat and cfg.strat_mode == "soft":
        # Graded contextual evidence.  Fragments recovered from squares further
        # apart than the radius cannot have lain together and the arc is cut;
        # within the radius the record shifts the odds rather than deciding
        # them; a fragment recovered without a record is neither helped nor
        # excluded, which is what makes an incomplete excavation log usable at
        # all.
        d = block_distance(meta)
        lr = np.asarray(cfg.strat_lr if cfg.strat_lr else
                        (0.0,) * (cfg.strat_radius + 1), np.float64)
        ctx = np.zeros_like(Wm)
        for k in range(min(len(lr), cfg.strat_radius + 1)):
            ctx[d == k] = lr[k]
        Wm = Wm + ctx
        Wm[d > cfg.strat_radius] = -np.inf
    elif cfg.use_strat:
        # The gate excludes a join only when both fragments carry a record and
        # the records differ.  A fragment recovered without one is excluded
        # from nothing, which is the fairest reading of a gate: making it
        # joinable only with other unrecorded fragments would be a weaker
        # baseline than anyone would actually build.
        known = unit >= 0
        bad = (unit[:, None] != unit[None, :]) & known[:, None] & known[None, :]
        Wm[bad] = -np.inf
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
    roll = np.array([m.get("roll", -1) for m in meta], np.int64)
    # The size of the scribe alphabet fixes the weight of the hand term, so it
    # is measured once over the whole corpus.  Left to be inferred inside each
    # sub-problem it would price the same evidence differently in different
    # parts of the corpus, and the decomposition below would stop being exact.
    k_hand = int(max(int(cfg.n_scribes), int(hand.max()) + 1 if n else 0))
    return dict(n=n, arcs=arcs, weights=np.array(wts), height=height,
                width=width, hand=hand, unit=unit, roll=roll, meta=meta,
                cfg=cfg, k_hand=k_hand)


# --------------------------------------------------------------------------
# exact solver
# --------------------------------------------------------------------------

def solve_cpsat(prob, time_limit=None, workers=None, log=False, trace=False,
                constants=False):
    """Exact (or best-found) maximum-weight consistent path cover via CP-SAT.

    ``constants`` builds the model even when no candidate arc survives.  A
    fragment that joins nothing still contributes to the objective, through the
    hand term and through any notch it cannot place, and the decomposition
    below needs those terms to add up to the objective of the whole corpus.
    """
    from ortools.sat.python import cp_model

    cfg = prob["cfg"]
    n, arcs, wts = prob["n"], prob["arcs"], prob["weights"]
    height, meta = prob["height"], prob["meta"]
    if not arcs and not (constants and n):
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

            if cfg.use_roll_length:
                # One roll is one manuscript, and a manuscript was trimmed to a
                # single standard length.  Estimating that length jointly over
                # the slips of a roll, instead of allowing each chain the full
                # 231-278 mm of the corpus, is what makes the binding notches
                # bite: the admissible band around a cord shrinks from the
                # spread of the corpus to the spread of one roll.  The grouping
                # key is the excavation square, which is information a site
                # already records; grouping by the true bundle is reported
                # separately as the upper bound it is.
                key = prob["roll"] if cfg.roll_key == "roll" else prob["unit"]
                groups = {}
                for i in range(n):
                    g = int(key[i])
                    if g >= 0:
                        groups.setdefault(g, []).append(i)
                rtol = int(round(cfg.roll_len_tol_mm * MM))
                K = max(1, int(cfg.roll_len_classes))
                for g, idx in groups.items():
                    if len(idx) < 2:
                        continue
                    RL = [m.NewIntVar(int(round(lo_len * MM)),
                                      int(round(hi_len * MM)), f"RL{g}_{k}")
                          for k in range(K)]
                    for k in range(K - 1):   # ordered, to break the symmetry
                        m.Add(RL[k] <= RL[k + 1])
                    for i in idx:
                        if K == 1:
                            m.Add(Lv[i] - RL[0] <= rtol)
                            m.Add(RL[0] - Lv[i] <= rtol)
                            continue
                        sel = [m.NewBoolVar("") for _ in range(K)]
                        m.AddExactlyOne(sel)
                        for k in range(K):
                            m.Add(Lv[i] - RL[k] <= rtol).OnlyEnforceIf(sel[k])
                            m.Add(RL[k] - Lv[i] <= rtol).OnlyEnforceIf(sel[k])

        if cfg.use_notch:
            tol = int(round(cfg.notch_tol_mm * MM))
            fr100 = [int(round(f * 100)) for f in cfg.cord_fracs]
            windows = [(int(round((lo_len * f - cfg.notch_tol_mm) * MM)),
                        int(round((hi_len * f + cfg.notch_tol_mm) * MM)))
                       for f in cfg.cord_fracs]
            for i in range(n):
                per_notch = []
                for nu in sorted(meta[i].get("notches_mm", [])):
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
                        per_notch.append(sel[:len(cfg.cord_fracs)])

                if cfg.notch_order and len(per_notch) > 1:
                    # Notches read down a fragment must take cords in the same
                    # order.  Placed independently, as they were, two notches
                    # on one fragment can both claim the same cord, which drops
                    # exactly the relative spacing between them: the one piece
                    # of notch evidence that does not depend on knowing where
                    # the top of the slip is.
                    for a, b in zip(per_notch, per_notch[1:]):
                        for c1 in range(len(a)):
                            for c2 in range(len(b)):
                                if c2 <= c1:
                                    m.AddBoolOr([a[c1].Not(), b[c2].Not()])

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
        K = int(prob.get("k_hand", max(int(cfg.n_scribes), int(hand.max()) + 1)))
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
# exact decomposition
# --------------------------------------------------------------------------

def _components(prob):
    """Partition the fragments into groups that share no constraint.

    Every variable of the model is per fragment: the position along the slip,
    the latent length, width and hand, and the placement of each notch.  The
    only thing that couples two fragments is a candidate arc between them, and
    the gates delete most arcs before the solver ever sees them.  The problem
    is therefore separable over the weakly connected components of the arc
    graph, and solving each component on its own and concatenating the results
    returns the optimum of the whole corpus, not an approximation of it.

    The one exception is the roll-pooled length, which ties every fragment
    recorded in one excavation square to a shared value.  Where that is in use
    those fragments are placed in the same component, so the decomposition
    stays exact.
    """
    n = prob["n"]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j) in prob["arcs"]:
        union(i, j)

    cfg = prob["cfg"]
    if (cfg.use_roll_length and cfg.use_latent
            and (cfg.use_length or cfg.use_notch)):
        key = prob["roll"] if cfg.roll_key == "roll" else prob["unit"]
        first = {}
        for i in range(n):
            g = int(key[i])
            if g < 0:
                continue
            if g in first:
                union(first[g], i)
            else:
                first[g] = i

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def _subproblem(prob, idx):
    """The same problem restricted to a set of fragments, re-indexed."""
    remap = {g: l for l, g in enumerate(idx)}
    arcs, wts = [], []
    for a, w in zip(prob["arcs"], prob["weights"]):
        if a[0] in remap and a[1] in remap:
            arcs.append((remap[a[0]], remap[a[1]]))
            wts.append(w)
    take = np.asarray(idx, np.int64)
    return dict(n=len(idx), arcs=arcs, weights=np.array(wts, np.float64),
                height=prob["height"][take], width=prob["width"][take],
                hand=prob["hand"][take], unit=prob["unit"][take],
                roll=prob["roll"][take],
                meta=[prob["meta"][i] for i in idx], cfg=prob["cfg"],
                k_hand=prob.get("k_hand"))


def solve_decomposed(prob, time_limit=None, workers=None, min_size=2):
    """Solve the corpus one connected component at a time.

    Reported alongside the solution: how many components the corpus fell into
    and how large the largest was, since those are what decide whether a corpus
    can be solved to proven optimality at all.
    """
    import time as _time

    comps = _components(prob)
    big = [c for c in comps if len(c) >= min_size]
    small = [i for c in comps if len(c) < min_size for i in c]
    if small:
        big.append(small)   # isolated fragments interact with nothing; one model
    big.sort(key=len, reverse=True)

    joins, obj, bound = [], 0.0, 0.0
    statuses = []
    t0 = _time.time()
    for c in big:
        sub = _subproblem(prob, c)
        if sub["n"] == 0:
            continue
        r = solve_cpsat(sub, time_limit=time_limit, workers=workers,
                        constants=True)
        statuses.append(r["status"])
        obj += r.get("objective", 0.0)
        bound += r.get("bound", r.get("objective", 0.0))
        joins += [(c[a], c[b]) for a, b in r["joins"]]
    wall = _time.time() - t0

    status = "OPTIMAL" if all(x in ("OPTIMAL", "EMPTY") for x in statuses) \
        else ("FEASIBLE" if any(x in ("OPTIMAL", "FEASIBLE") for x in statuses)
              else (statuses[0] if statuses else "EMPTY"))
    return dict(joins=joins, objective=obj, bound=bound, status=status,
                wall=wall, gap=(bound - obj) / max(abs(obj), 1e-9),
                n_components=len(comps),
                max_component=max((len(c) for c in comps), default=0),
                n_solved=len(big))


# --------------------------------------------------------------------------
# chains
# --------------------------------------------------------------------------

DEFAULT_P_GRID = (0.30, 0.50, 0.70, 0.85, 0.95, 0.99, 0.997, 0.9997)

# Wider grid used from the revision onward.  A constraint that enters the
# objective rather than the arc set shifts every arc weight by a constant, so a
# grid that stops at 0.30 can put the optimum of one method inside the search
# and the optimum of another outside it.  Comparing methods at their own best
# operating point requires a grid wide enough that none of them is pinned to an
# endpoint, and every experiment reports whether any method was.
WIDE_P_GRID = (0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99, 0.997, 0.9997)


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
