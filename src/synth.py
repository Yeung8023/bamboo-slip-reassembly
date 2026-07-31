"""Physics-constrained synthetic generator for fragmented bamboo slips.

The generator produces benchmark instances with complete ground truth: intact
slips are rendered with an anisotropic bamboo substrate, a scribal hand and the
morphometric signature of a real corpus (width, binding-notch heights), and are
then fractured with a crack model that reflects how bamboo actually breaks --
transverse cracks that deflect along the longitudinal fibre bundles, producing
jagged, heavy-tailed pull-out profiles -- followed by taphonomic loss.

Defaults are calibrated to the Shuihudi Qin corpus (Yunmeng, Hubei; excavated
1975): slip length 23.1-27.8 cm, width 5-8 mm, three binding cords.

Everything is deterministic given ``seed``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

PX_PER_MM = 8.0

# Canonical raster width every fracture descriptor is resampled to, so that the
# neural matcher never sees the true slip width.  Width is withheld on purpose:
# it is the global layer's evidence, not the matcher's, which keeps the
# morphometric-constraint ablation free of leakage.
CANON_W = 48
PATCH_H = 32


@dataclass
class CorpusSpec:
    """Physical description of the corpus being simulated."""

    name: str = "shuihudi"
    length_mm: tuple[float, float] = (231.0, 278.0)
    width_mm: tuple[float, float] = (5.0, 8.0)
    thickness_mm: tuple[float, float] = (0.7, 1.2)
    n_cords: int = 3
    # Binding cords sit at fixed fractions of the slip length; the notches cut
    # for them are therefore an *absolute* positional anchor along the slip.
    cord_fracs: tuple[float, ...] = (0.06, 0.5, 0.94)
    cord_jitter_mm: float = 3.0
    notch_depth_mm: float = 0.8
    glyph_pitch_mm: float = 9.0
    n_scribes: int = 6
    n_units: int = 8  # stratigraphic / spatial excavation units


@dataclass
class DamageSpec:
    """Fracture and taphonomy controls."""

    breaks_per_slip: float = 2.2  # Poisson mean
    min_piece_mm: float = 12.0
    # Fibre pull-out: crack deflects along fibre bundles, giving a heavy-tailed
    # amplitude and a short correlation length across the width.
    pullout_mean_mm: float = 1.4
    pullout_shape: float = 1.3  # Gamma shape (<2 => heavy tail)
    fibre_corr_mm: float = 0.55
    # Material lost at a break: when it exceeds a hairline the two pieces no
    # longer physically conjoin even though they belong to the same slip.
    p_gap: float = 0.35
    gap_mean_mm: float = 4.0
    # Whole fragments never recovered.
    p_fragment_lost: float = 0.12
    # Differential loss at the fracture *face*, drawn independently for the two
    # sides of every break.  This is the parameter that governs how hard
    # rejoining actually is.  The two faces of one break are not the same curve
    # observed twice: each has lost its own sliver of material, each carries its
    # own consolidant and dirt, and the silhouette recovered from an orthophoto
    # is a projection of a three-dimensional relief that the two faces present
    # differently.  Whatever is destroyed here cannot be recovered by a better
    # matcher, however it is trained -- which is precisely why it, and not the
    # network, sets the ceiling on pairwise scoring.
    face_loss_mm: float = 0.55
    face_loss_corr_mm: float = 1.1
    # Independent corruption of the texture strip at each face: consolidant,
    # dirt, differential staining and the discolouration that follows a break.
    face_texture_noise: float = 0.10
    # Edge erosion, warping and photometric variation of the recovered pieces.
    erosion_mm: float = 0.25
    warp_deg: float = 1.2
    ink_fade: float = 0.35
    stain_rate: float = 0.6


@dataclass
class ObservationSpec:
    """Noise on the metadata a conservator would actually record."""

    width_sigma_mm: float = 0.12
    notch_sigma_mm: float = 0.6
    hand_accuracy: float = 0.80  # off-the-shelf writer-ID model
    unit_known_frac: float = 1.0  # excavation records are exact by default


@dataclass
class InstanceSpec:
    n_slips: int = 60
    seed: int = 0
    corpus: CorpusSpec = field(default_factory=CorpusSpec)
    damage: DamageSpec = field(default_factory=DamageSpec)
    obs: ObservationSpec = field(default_factory=ObservationSpec)


# --------------------------------------------------------------------------
# substrate
# --------------------------------------------------------------------------

def _smooth_noise(rng, shape, corr_y, corr_x):
    """Band-limited noise with the given correlation lengths, in pixels."""
    h, w = shape
    gy = max(2, int(round(h / max(corr_y, 1.0))) + 1)
    gx = max(2, int(round(w / max(corr_x, 1.0))) + 1)
    coarse = rng.standard_normal((gy, gx))
    yi = np.linspace(0, gy - 1, h)
    xi = np.linspace(0, gx - 1, w)
    y0 = np.clip(np.floor(yi).astype(int), 0, gy - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, gx - 2)
    ty = (yi - y0)[:, None]
    tx = (xi - x0)[None, :]
    ty = ty * ty * (3 - 2 * ty)  # smoothstep
    tx = tx * tx * (3 - 2 * tx)
    c = coarse
    top = c[y0][:, x0] * (1 - tx) + c[y0][:, x0 + 1] * tx
    bot = c[y0 + 1][:, x0] * (1 - tx) + c[y0 + 1][:, x0 + 1] * tx
    return top * (1 - ty) + bot * ty


def _bamboo_substrate(rng, h, w):
    """Longitudinal fibre striations: sharp across the width, slowly varying
    along the length.  A horizontal cut through this field is highly
    informative, which is the physical basis of texture continuity at a join."""
    fibre = _smooth_noise(rng, (h, w), corr_y=180.0, corr_x=1.8)
    coarse = _smooth_noise(rng, (h, w), corr_y=600.0, corr_x=14.0)
    img = 0.70 + 0.11 * fibre + 0.07 * coarse
    # vascular bundles: a few dark longitudinal lines held over the full length
    n_b = rng.integers(2, 5)
    xs = rng.integers(0, w, n_b)
    for x in xs:
        prof = np.exp(-0.5 * ((np.arange(w) - x) / 1.1) ** 2)
        img -= (0.05 + 0.04 * rng.random()) * prof[None, :]
    return np.clip(img, 0.05, 1.0).astype(np.float32)


def _scribe_style(rng, scribe_id, n_scribes):
    """Per-hand calligraphic parameters, stable across all slips of that hand."""
    r = np.random.default_rng(1000 + scribe_id)
    return dict(
        stroke_px=r.uniform(1.4, 2.8),
        slant=r.uniform(-0.16, 0.16),
        glyph_scale=r.uniform(0.78, 1.05),
        n_strokes=(int(r.integers(3, 6)), int(r.integers(6, 10))),
        ink=r.uniform(0.55, 0.85),
        pitch_jitter=r.uniform(0.03, 0.11),
    )


def _draw_glyph(canvas, rng, style, cx, cy, gw, gh, ink):
    """One clerical-script (隶书) style character.

    Qin-Han clerical script is built from a small stroke grammar dominated by
    horizontals: a stack of 1-4 horizontal bars, 1-2 verticals crossing them,
    and optional diagonals/hooks.  Rendering from that grammar rather than from
    unstructured strokes gives fragments the right glyph statistics, which is
    what a scribal-hand classifier keys on.
    """
    import cv2

    lw = max(1, int(round(style["stroke_px"])))
    sl = style["slant"]

    def seg(x0, y0, x1, y1, scale=1.0):
        # slant shears x with height, as a brush-held-at-an-angle would
        x0s = x0 + sl * (y0 - cy)
        x1s = x1 + sl * (y1 - cy)
        cv2.line(canvas, (int(round(x0s)), int(round(y0))),
                 (int(round(x1s)), int(round(y1))), float(ink),
                 max(1, int(round(lw * scale))), lineType=cv2.LINE_AA)

    hl, hr = cx - gw / 2, cx + gw / 2
    top, bot = cy - gh / 2, cy + gh / 2

    n_h = int(rng.integers(1, 5))
    ys = np.sort(rng.uniform(top, bot, n_h)) if n_h > 1 else np.array([cy])
    for i, y in enumerate(ys):
        f = rng.uniform(0.55, 1.0)
        x0 = cx - gw * f / 2
        x1 = cx + gw * f / 2
        seg(x0, y, x1, y, scale=rng.uniform(0.9, 1.35))
        # 波磔: the flared, downward-thickening tail of the dominant horizontal
        if i == n_h - 1 and rng.random() < 0.5:
            seg(x1 - gw * 0.12, y, x1, y + gh * 0.06, scale=1.6)

    n_v = int(rng.integers(1, 3))
    for _ in range(n_v):
        x = cx + rng.uniform(-0.3, 0.3) * gw
        y0 = top + rng.uniform(0, 0.25) * gh
        y1 = bot - rng.uniform(0, 0.25) * gh
        seg(x, y0, x, y1)
        if rng.random() < 0.3:  # 钩 hook at the foot
            seg(x, y1, x - gw * 0.16, y1 - gh * 0.08)

    for _ in range(int(rng.integers(0, 3))):
        x0 = cx + rng.uniform(-0.45, 0.1) * gw
        y0 = top + rng.uniform(0.0, 0.6) * gh
        seg(x0, y0, x0 + gw * rng.uniform(0.2, 0.5),
            y0 + gh * rng.uniform(0.15, 0.4) * rng.choice([-1, 1]))


def _draw_glyphs(img, rng, style, pitch_px, ink_gain):
    h, w = img.shape
    gh = pitch_px * 0.70 * style["glyph_scale"]
    gw = min(w * 0.62, gh * 0.9)
    cx = w * 0.46  # column sits left of centre, clear of the binding notches
    y = pitch_px * 0.7
    canvas = np.zeros(img.shape, np.float32)
    ink = style["ink"] * ink_gain
    while y < h - gh * 0.6:
        _draw_glyph(canvas, rng, style, cx, y, gw, gh,
                    ink * rng.uniform(0.85, 1.0))
        y += pitch_px * (1.0 + rng.normal(0, style["pitch_jitter"]))
    img = img - np.minimum(canvas, ink)
    return np.clip(img, 0.02, 1.0)


# --------------------------------------------------------------------------
# fracture
# --------------------------------------------------------------------------

def _fracture_profile(rng, w_px, damage: DamageSpec):
    """One transverse crack, as a per-column offset in pixels.

    Bamboo is strongly anisotropic: a transverse crack repeatedly deflects into
    the weak fibre-matrix interface, so the fracture surface is a sequence of
    pulled-out fibre bundles.  We model the offset as band-limited noise whose
    correlation length is the bundle spacing, scaled by a Gamma-distributed
    pull-out length with shape < 2 (heavy tail => occasional long tongues).
    """
    corr_px = max(1.0, damage.fibre_corr_mm * PX_PER_MM)
    base = _smooth_noise(rng, (1, w_px), corr_y=1e9, corr_x=corr_px)[0]
    base = base / (np.abs(base).max() + 1e-9)
    amp = rng.gamma(damage.pullout_shape,
                    damage.pullout_mean_mm / damage.pullout_shape) * PX_PER_MM
    prof = base * amp
    # a single dominant tongue is common: one bundle pulls far out
    if rng.random() < 0.45:
        c = rng.integers(0, w_px)
        width = max(1.5, rng.gamma(2.0, 1.5) * PX_PER_MM * 0.25)
        tongue = np.exp(-0.5 * ((np.arange(w_px) - c) / width) ** 2)
        prof += tongue * rng.gamma(1.6, amp * 0.9) * rng.choice([-1.0, 1.0])
    return prof - prof.mean()


def _cut(img, y_lo, y_hi, prof_top, prof_bot):
    """Extract the strip between two fracture profiles, returning image+mask."""
    h, w = img.shape
    lo = int(np.floor(y_lo + prof_top.min())) - 2
    hi = int(np.ceil(y_hi + prof_bot.max())) + 2
    lo, hi = max(0, lo), min(h, hi)
    if hi - lo < 4:
        return None, None
    sub = img[lo:hi].copy()
    yy = np.arange(lo, hi)[:, None]
    mask = ((yy >= (y_lo + prof_top)[None, :]) &
            (yy < (y_hi + prof_bot)[None, :])).astype(np.float32)
    return sub, mask


# --------------------------------------------------------------------------
# instance assembly
# --------------------------------------------------------------------------

def deskew(img, mask):
    """Rectify a fragment on its own long axis and crop to its silhouette.

    Every real rejoining pipeline does this before comparing break edges; we
    apply it here so that the photographic tilt injected by the generator has
    to be *estimated* rather than being handed to the matcher for free.  The
    residual error -- driven by the ragged ends and by erosion -- is what a
    real system would also be left with.
    """
    import cv2

    on = mask > 0.5
    if on.sum() < 20:
        return img, mask
    h, w = mask.shape

    # Rectify on the slip's two long machined edges, not on the silhouette's
    # principal axis: fragments can be nearly as wide as they are tall, and a
    # moment-based axis is then almost unidentified.  The side edges are the
    # cue a conservator actually uses.  Rows near the ragged ends are skipped.
    rows = np.where(on.any(axis=1))[0]
    lo = rows[0] + int(0.10 * len(rows))
    hi = rows[-1] - int(0.10 * len(rows))
    if hi - lo < 6:
        lo, hi = rows[0], rows[-1]
    rr = np.arange(lo, hi + 1)
    sub = on[lo:hi + 1]
    valid = sub.any(axis=1)
    if valid.sum() < 6:
        return img, mask
    rr = rr[valid]
    left = np.argmax(sub[valid], axis=1).astype(np.float64)
    right = (sub.shape[1] - 1 - np.argmax(sub[valid][:, ::-1], axis=1)).astype(np.float64)

    slopes = []
    for edge in (left, right):
        # Trimmed least squares: the binding notches bite into the right edge
        # and must not drag the fit, but differencing integer column indices
        # quantises the slope far too coarsely to use a rank estimator here.
        keep = np.ones(len(rr), bool)
        s = 0.0
        for _ in range(3):
            if keep.sum() < 6:
                break
            s, b = np.polyfit(rr[keep], edge[keep], 1)
            resid = edge - (s * rr + b)
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
            keep = np.abs(resid - np.median(resid)) < 3.0 * mad
        slopes.append(s)
    if not slopes:
        return img, mask
    slope = float(np.mean(slopes))  # dx/dy
    ang = -np.degrees(np.arctan(slope))
    ang = float(np.clip(ang, -12.0, 12.0))

    pad = int(np.ceil(abs(np.sin(np.radians(ang))) * h / 2)) + 3
    imgp = np.pad(img, ((0, 0), (pad, pad)))
    mskp = np.pad(mask, ((0, 0), (pad, pad)))
    w2 = w + 2 * pad
    M = cv2.getRotationMatrix2D((w2 / 2, h / 2), ang, 1.0)
    imgr = cv2.warpAffine(imgp, M, (w2, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    mskr = cv2.warpAffine(mskp, M, (w2, h), flags=cv2.INTER_NEAREST, borderValue=0.0)
    on2 = mskr > 0.5
    if not on2.any():
        return img, mask
    cols = np.where(on2.any(axis=0))[0]
    rows = np.where(on2.any(axis=1))[0]
    sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    return imgr[sl], mskr[sl]


def column_support(mask, side):
    """The slip's own column span at one end of the fragment.

    A fracture edge must be resampled over the slip's width -- not over the
    silhouette's bounding box, whose outer columns clip only a corner once any
    residual tilt is left after rectification.  On a near-intact slip a tilt of
    a fortieth of a degree already smears the piece over several extra columns,
    and those columns carry an offset of hundreds of pixels.

    The span is therefore measured *at the end in question*, in a band starting
    where the piece first reaches its full width (i.e. just past the ragged
    fracture zone), using the median first/last lit column so that the
    jaggedness and the binding notches cannot move it.
    """
    on = mask > 0.5
    counts = on.sum(axis=1)
    rows = np.where(counts > 0)[0]
    if rows.size < 6:
        return None
    w_full = np.percentile(counts[rows], 90)
    if w_full < 8:
        return None
    full = np.where(counts >= 0.9 * w_full)[0]
    if full.size == 0:
        full = rows
    band_len = min(60, max(6, len(rows) // 4))
    if side == "top":
        r0 = full[0]
        band = on[r0:r0 + band_len]
    else:
        r1 = full[-1]
        band = on[max(0, r1 - band_len + 1):r1 + 1]
    band = band[band.any(axis=1)]
    if band.shape[0] < 3:
        return None
    first = np.argmax(band, axis=1)
    last = band.shape[1] - 1 - np.argmax(band[:, ::-1], axis=1)
    c0 = int(np.median(first)) + 1
    c1 = int(np.median(last))  # inset one column at each side for safety
    if c1 - c0 < 8:
        return None
    return c0, c1


def _face_loss(rng, w, amp_px, corr_px):
    """Material lost from one fracture face, as an inward per-column offset.

    Positive-valued and spatially correlated: loss comes in patches, not as
    white noise.  Drawn independently for each of the two faces of a break.
    """
    if amp_px <= 0:
        return np.zeros(w, np.float32)
    base = _smooth_noise(rng, (1, w), corr_y=1e9, corr_x=max(1.0, corr_px))[0]
    base = base / (np.abs(base).max() + 1e-9)
    return (np.abs(base) * rng.gamma(2.0, amp_px / 2.0)).astype(np.float32)


def _edge_descriptors(img, mask, side, support=None, loss=None, rng=None,
                      tex_noise=0.0):
    """Resample a fracture edge to canonical width.

    Returns the silhouette profile (mean-removed, in px) and the texture patch
    immediately inside the edge, both at CANON_W columns.
    """
    h, w = mask.shape
    on = mask > 0.5
    if not on.any():
        return None, None
    has = on.any(axis=0)
    if support is None:
        support = column_support(mask, side)
    if support is None:
        return None, None
    c0, c1 = support
    c0 = max(0, c0)
    c1 = min(w, c1)
    if c1 - c0 < 8:
        return None, None

    # The silhouette boundary must be the first *sustained* run of material,
    # not the first lit pixel: resampling a ragged mask leaves isolated specks,
    # and a single one of them would drag that column's offset by tens of
    # pixels and swamp the mean-removed profile.
    K = 3
    acc = on.copy()
    if side == "top":
        for k in range(1, K):
            acc[:-k] &= on[k:]
        idx = np.argmax(acc, axis=0).astype(np.float32)
    else:
        for k in range(1, K):
            acc[k:] &= on[:-k]
        idx = (h - 1 - np.argmax(acc[::-1], axis=0)).astype(np.float32)
    has = acc.any(axis=0)
    if has.sum() < 3:
        return None, None

    good = has
    if good.sum() < 3:
        return None, None
    idx = np.interp(np.arange(w), np.arange(w)[good], idx[good])
    if loss is not None:
        # material gone from this face: the observed boundary sits further in
        idx = idx + (loss if side == "top" else -loss)
        idx = np.clip(idx, 0, h - 1)
    xs = np.linspace(c0, c1 - 1, CANON_W)
    prof = np.interp(xs, np.arange(w), idx)
    prof = prof - prof.mean()

    # texture strip taken *along* the fracture so it is edge-aligned: row k of
    # the patch is the pixel k rows inside the silhouette at that column.
    patch = np.zeros((PATCH_H, CANON_W), np.float32)
    sgn = 1 if side == "top" else -1
    for j, x in enumerate(xs):
        x0 = int(round(x))
        start = idx[x0]
        rows = np.clip(np.round(start + sgn * np.arange(PATCH_H)), 0, h - 1).astype(int)
        col = img[rows, x0] * mask[rows, x0]
        patch[:, j] = col
    if tex_noise and tex_noise > 0 and rng is not None:
        # smooth, low-frequency corruption (staining/consolidant) plus grain
        blot = _smooth_noise(rng, patch.shape, corr_y=10.0, corr_x=8.0)
        patch = patch * (1.0 + tex_noise * blot) + \
            tex_noise * 0.4 * rng.standard_normal(patch.shape)
        patch = np.clip(patch, 0.0, 1.5).astype(np.float32)
    return prof.astype(np.float32), patch


def generate_instance(spec: InstanceSpec, render_gallery: int = 0):
    """Build one benchmark instance.

    Returns a dict with per-fragment descriptors, observed metadata, and full
    ground truth (slip membership, order, and which adjacent pairs physically
    conjoin as opposed to being separated by lost material).
    """
    import cv2

    rng = np.random.default_rng(spec.seed)
    C, D, O = spec.corpus, spec.damage, spec.obs

    frags = []
    gallery = []
    slips_meta = []

    for sid in range(spec.n_slips):
        srng = np.random.default_rng(rng.integers(1 << 62))
        L = srng.uniform(*C.length_mm)
        W = srng.uniform(*C.width_mm)
        scribe = int(srng.integers(C.n_scribes))
        unit = int(srng.integers(C.n_units))
        cords = np.sort(np.array([
            L * f + srng.normal(0, C.cord_jitter_mm) for f in C.cord_fracs
        ]))
        cords = np.clip(cords, 3.0, L - 3.0)

        h = int(round(L * PX_PER_MM))
        w = int(round(W * PX_PER_MM))
        img = _bamboo_substrate(srng, h, w)
        style = _scribe_style(srng, scribe, C.n_scribes)
        img = _draw_glyphs(img, srng, style, C.glyph_pitch_mm * PX_PER_MM,
                           1.0 - D.ink_fade * srng.random())

        # binding notches cut into the right edge
        notch_mask = np.ones((h, w), np.float32)
        nd = int(round(C.notch_depth_mm * PX_PER_MM))
        nh = int(round(1.6 * PX_PER_MM))
        for cy in cords:
            y0 = int(round(cy * PX_PER_MM))
            for yy in range(max(0, y0 - nh // 2), min(h, y0 + nh // 2)):
                t = abs(yy - y0) / max(1, nh / 2)
                d = int(round(nd * (1 - t * t)))
                if d > 0:
                    notch_mask[yy, w - d:] = 0.0

        # break positions
        nbr = srng.poisson(D.breaks_per_slip)
        cuts = []
        for _ in range(200):
            if len(cuts) >= nbr:
                break
            c = srng.uniform(D.min_piece_mm, L - D.min_piece_mm)
            if all(abs(c - o) > D.min_piece_mm for o in cuts):
                cuts.append(c)
        cuts = sorted(cuts)

        # each break: a fracture profile plus possible material loss
        bounds = [0.0] + cuts + [L]
        profiles = [np.zeros(w, np.float32)]
        gaps = []
        for _ in cuts:
            profiles.append(_fracture_profile(srng, w, D).astype(np.float32))
            gaps.append(srng.exponential(D.gap_mean_mm)
                        if srng.random() < D.p_gap else 0.0)
        profiles.append(np.zeros(w, np.float32))

        slips_meta.append(dict(slip_id=sid, length_mm=float(L), width_mm=float(W),
                               scribe=scribe, unit=unit,
                               cords_mm=[float(c) for c in cords],
                               n_pieces=len(bounds) - 1))

        pieces = []
        for k in range(len(bounds) - 1):
            y_lo_mm = bounds[k] + (gaps[k - 1] if k > 0 else 0.0) * 0.5
            y_hi_mm = bounds[k + 1] - (gaps[k] if k < len(gaps) else 0.0) * 0.5
            if y_hi_mm - y_lo_mm < D.min_piece_mm * 0.5:
                pieces.append(None)
                continue
            y_lo = y_lo_mm * PX_PER_MM
            y_hi = y_hi_mm * PX_PER_MM
            sub, msk = _cut(img * notch_mask, y_lo, y_hi, profiles[k], profiles[k + 1])
            if sub is None:
                pieces.append(None)
                continue
            pieces.append(dict(img=sub, mask=msk, y_lo_mm=y_lo_mm, y_hi_mm=y_hi_mm,
                               order=k, top_gap=(gaps[k - 1] if k > 0 else 0.0),
                               bot_gap=(gaps[k] if k < len(gaps) else 0.0)))

        for k, p in enumerate(pieces):
            if p is None:
                continue
            if srng.random() < D.p_fragment_lost:
                pieces[k] = None

        for k, p in enumerate(pieces):
            if p is None:
                continue
            im, msk = p["img"], p["mask"]
            # taphonomy on the recovered piece
            er = int(round(D.erosion_mm * PX_PER_MM))
            if er > 0:
                msk = cv2.erode(msk, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
            if D.stain_rate > 0:
                ns = srng.poisson(D.stain_rate)
                for _ in range(ns):
                    cy = srng.integers(0, im.shape[0])
                    cx = srng.integers(0, im.shape[1])
                    rr = srng.uniform(2, 9) * PX_PER_MM * 0.35
                    yy, xx = np.ogrid[:im.shape[0], :im.shape[1]]
                    blob = np.exp(-0.5 * (((yy - cy) ** 2 + (xx - cx) ** 2) / rr ** 2))
                    im = im * (1 - 0.35 * srng.random() * blob)
            gain = srng.uniform(0.85, 1.15)
            off = srng.uniform(-0.05, 0.05)
            im = np.clip(im * gain + off, 0.0, 1.0)
            if D.warp_deg > 0:
                # Residual misalignment after a conservator rectifies the piece
                # on its long axis: a small tilt plus a gentle bow from the
                # natural curvature of the culm.  Pad first, otherwise a tall
                # narrow strip rotates straight out of its own frame.
                ang = srng.normal(0, D.warp_deg)
                bow = srng.normal(0, 0.35) * PX_PER_MM
                hh, ww = im.shape
                # Pad for the rotation *and* the bow together, before either is
                # applied: a shift past the margin would otherwise wrap the
                # silhouette around to the opposite edge.
                pad = (int(np.ceil(abs(np.sin(np.radians(ang))) * hh / 2))
                       + int(np.ceil(abs(bow))) + 4)
                im = np.pad(im, ((0, 0), (pad, pad)))
                msk = np.pad(msk, ((0, 0), (pad, pad)))
                ww2 = ww + 2 * pad
                yy = np.linspace(-1, 1, hh)
                shift = bow * (1 - yy ** 2)  # gentle bow from the culm curvature
                map_x = (np.arange(ww2)[None, :] - shift[:, None]).astype(np.float32)
                map_y = np.repeat(np.arange(hh, dtype=np.float32)[:, None], ww2, 1)
                M = cv2.getRotationMatrix2D((ww2 / 2, hh / 2), ang, 1.0)
                cos, sin = M[0, 0], M[0, 1]
                # compose rotation into the same remap so we resample once
                cx, cy = ww2 / 2, hh / 2
                dx, dy = map_x - cx, map_y - cy
                rx = cos * dx - sin * dy + cx
                ry = sin * dx + cos * dy + cy
                im = cv2.remap(im, rx.astype(np.float32), ry.astype(np.float32),
                               cv2.INTER_LINEAR, borderValue=0.0)
                msk = cv2.remap(msk, rx.astype(np.float32), ry.astype(np.float32),
                                cv2.INTER_NEAREST, borderValue=0.0)
            p["img"], p["mask"] = im, msk

        # observed metadata + descriptors
        for k, p in enumerate(pieces):
            if p is None:
                continue
            rimg, rmsk = deskew(p["img"], p["mask"])
            wpx = rmsk.shape[1]
            fl_amp = D.face_loss_mm * PX_PER_MM
            fl_corr = D.face_loss_corr_mm * PX_PER_MM
            tp, tpatch = _edge_descriptors(
                rimg, rmsk, "top", loss=_face_loss(srng, wpx, fl_amp, fl_corr),
                rng=srng, tex_noise=D.face_texture_noise)
            bp, bpatch = _edge_descriptors(
                rimg, rmsk, "bot", loss=_face_loss(srng, wpx, fl_amp, fl_corr),
                rng=srng, tex_noise=D.face_texture_noise)
            if tp is None or bp is None:
                continue
            fid = len(frags)
            obs_w = W + srng.normal(0, O.width_sigma_mm)
            # notches visible on this fragment, measured from its own top edge
            local_notches = [float(c - p["y_lo_mm"] + srng.normal(0, O.notch_sigma_mm))
                             for c in cords
                             if p["y_lo_mm"] <= c <= p["y_hi_mm"]]
            hand = scribe if srng.random() < O.hand_accuracy else \
                int(srng.integers(C.n_scribes))
            frags.append(dict(
                frag_id=fid, slip_id=sid, order=k,
                y_lo_mm=float(p["y_lo_mm"]), y_hi_mm=float(p["y_hi_mm"]),
                height_mm=float(p["y_hi_mm"] - p["y_lo_mm"]),
                true_width_mm=float(W), obs_width_mm=float(obs_w),
                notches_mm=local_notches, true_scribe=scribe, obs_hand=hand,
                unit=unit, top_gap=float(p["top_gap"]), bot_gap=float(p["bot_gap"]),
                top_prof=tp, bot_prof=bp, top_patch=tpatch, bot_patch=bpatch,
            ))
            if len(gallery) < render_gallery:
                gallery.append(dict(frag_id=fid, img=p["img"], mask=p["mask"]))

    # ground truth joins: consecutive surviving pieces of a slip that were not
    # separated by material loss
    by_slip = {}
    for f in frags:
        by_slip.setdefault(f["slip_id"], []).append(f)
    joins = []
    for sid, fl in by_slip.items():
        fl.sort(key=lambda f: f["order"])
        for a, b in zip(fl, fl[1:]):
            if b["order"] == a["order"] + 1 and a["bot_gap"] <= 0.5:
                joins.append((a["frag_id"], b["frag_id"]))

    return dict(spec=spec, frags=frags, joins=joins, slips=slips_meta,
                gallery=gallery)


def save_instance(inst, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frags = inst["frags"]
    n = len(frags)
    arrs = dict(
        top_prof=np.stack([f["top_prof"] for f in frags]) if n else np.zeros((0, CANON_W), np.float32),
        bot_prof=np.stack([f["bot_prof"] for f in frags]) if n else np.zeros((0, CANON_W), np.float32),
        top_patch=np.stack([f["top_patch"] for f in frags]).astype(np.float16) if n else np.zeros((0, PATCH_H, CANON_W), np.float16),
        bot_patch=np.stack([f["bot_patch"] for f in frags]).astype(np.float16) if n else np.zeros((0, PATCH_H, CANON_W), np.float16),
        joins=np.array(inst["joins"], np.int32).reshape(-1, 2),
    )
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in frags]
    np.savez_compressed(path, **arrs,
                        meta=np.array(json.dumps(meta)),
                        slips=np.array(json.dumps(inst["slips"])),
                        spec=np.array(json.dumps(_spec_to_dict(inst["spec"]))))
    return path


def _spec_to_dict(spec):
    d = asdict(spec)
    return d


def load_instance(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return dict(
        meta=meta,
        slips=json.loads(str(z["slips"])),
        spec=json.loads(str(z["spec"])),
        top_prof=z["top_prof"], bot_prof=z["bot_prof"],
        top_patch=z["top_patch"].astype(np.float32),
        bot_patch=z["bot_patch"].astype(np.float32),
        joins=z["joins"],
    )


if __name__ == "__main__":
    import time

    t0 = time.time()
    inst = generate_instance(InstanceSpec(n_slips=20, seed=0), render_gallery=4)
    dt = time.time() - t0
    n = len(inst["frags"])
    print(f"{n} fragments from 20 slips in {dt:.2f}s ({dt/20*1000:.0f} ms/slip)")
    print(f"true conjoining pairs: {len(inst['joins'])}")
    same_slip = sum(1 for s in inst["slips"] if s["n_pieces"] > 1)
    print(f"multi-piece slips: {same_slip}/20")
    f = inst["frags"][0]
    print("descriptor shapes:", f["top_prof"].shape, f["top_patch"].shape)
