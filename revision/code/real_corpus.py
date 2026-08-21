"""A benchmark corpus built from photographs of real excavated slips.

No public corpus of real bamboo slips carries verified rejoining ground truth.
The sets used by the matching literature come from Tomb 77 at Shuihudi, are not
yet formally published, and are available only on request. A reconstruction
metric needs complete ground truth, so a real corpus cannot supply one.

What is public is the material itself. DeepJiandu is 7,416 infrared photographs
of real Qin and Han bamboo and wooden slips from Gansu, in the condition they
were excavated in: faded ink, staining, biological attack, warping and eroded
edges. This module builds corpora whose substrate is those photographs and
whose fracture is generated, so that every image the matcher sees carries the
grain, the ink and the deterioration of a real object.

What that tests and what it does not. The pairwise stage is the only stage that
looks at an image, and it is here evaluated on real material with weights
trained entirely on synthetic corpora, which is a zero-shot transfer across the
domain gap the generator was meant to bridge. The fracture surface is still
generated, so the statistics of a real break are not tested here; that is what
the misspecification sweep in exp_realism.py is for, and it remains a stated
limitation rather than a closed question.

Reference: Liu et al., DeepJiandu dataset for character detection and
recognition on Jiandu manuscript, Scientific Data 12 (2025),
DOI 10.57760/sciencedb.08560, CC BY 4.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common
from common import synth


def _largest_component(mask):
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def segment_slip(path, min_h=1000, min_fill=0.55, min_aspect=6.0,
                 max_width_cv=0.25):
    """Cut the slip out of its photographic background.

    A global threshold followed by the largest connected component is enough on
    these plates, and anything that does not come out as one tall solid piece
    is discarded rather than repaired. Images are rejected, not rescued,
    because a bad segmentation would show up later as a fracture descriptor
    that describes the background.

    The height and aspect thresholds also keep the resampling honest. The
    photographs are crops at unknown scale, so a short crop stretched onto the
    height of a whole slip would distort the writing it carries far more than a
    tall one; taking only the tall crops keeps that distortion small.
    """
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape[0] < min_h:
        return None
    g = cv2.GaussianBlur(im, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Which side of the threshold is the slip is not fixed: a plate may show a
    # bright slip on a dark ground or the reverse, and a tight crop can make
    # the slip the majority of the frame. Rather than guess, both polarities
    # are tried and the one that yields a tall narrow solid object is kept.
    best = None
    for cand in (th > 0, th == 0):
        m = _largest_component(cand)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        if h < min_h or w < 20 or h / w < min_aspect:
            continue
        box = m[y0:y1, x0:x1]
        fill = box.mean()
        if fill < min_fill:
            continue
        # A slip has near constant width down its length. Requiring that
        # rejects the diagonal slivers and torn wedges a threshold
        # occasionally returns, which would otherwise enter the corpus as
        # fracture descriptors of something that is not a break.
        rw = box.sum(axis=1)
        rw = rw[rw > 0]
        if len(rw) < min_h or float(np.std(rw) / max(np.mean(rw), 1e-6)) \
                > max_width_cv:
            continue
        score = fill * min(h / w, 40.0)
        if best is None or score > best[0]:
            best = (score, m, (y0, y1, x0, x1))
    if best is None:
        return None
    _, m, (y0, y1, x0, x1) = best
    sub, msk = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    f = sub.astype(np.float32) / 255.0

    # The matcher was trained on dark ink over a lighter substrate, so a plate
    # recorded the other way round is inverted before anything else touches it.
    inside = f[msk > 0]
    if len(inside) and float(np.mean(inside)) < 0.45:
        f = 1.0 - f
    rimg, rmsk = synth.deskew(f * msk, msk.astype(np.float32))
    if rimg is None or rimg.shape[0] < min_h:
        return None
    return rimg, rmsk


class RealSubstrate:
    """Supplies the material of one real slip, scaled to the corpus dimensions.

    The photographs are crops of slips at unknown and varying scale, so the
    physical dimensions of the benchmark, the length and the width, are the
    ones the generator draws from the Shuihudi ranges, and the photograph is
    resampled onto them. The object supplies the material and its condition;
    the geometry is the geometry of the benchmark, and is labelled as such
    wherever these results are reported.
    """

    def __init__(self, paths, seed=0, cache_size=512):
        self.paths = list(paths)
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.cache_size = cache_size
        self.used = []

    def _load(self, k):
        if k in self.cache:
            return self.cache[k]
        out = segment_slip(self.paths[k])
        if len(self.cache) < self.cache_size:
            self.cache[k] = out
        return out

    def __call__(self, sid, rng, h, w):
        import cv2
        for _ in range(40):
            k = int(rng.integers(len(self.paths)))
            got = self._load(k)
            if got is None:
                continue
            img, msk = got
            im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            mk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
            im = im * (mk > 0.5)
            lo, hi = np.percentile(im[mk > 0.5], (2, 98)) if (mk > 0.5).any() \
                else (0.0, 1.0)
            if hi - lo < 1e-3:
                continue
            im = np.clip((im - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            self.used.append(Path(self.paths[k]).name)
            return im
        # every candidate failed segmentation: fall back to the rendered
        # substrate rather than returning something malformed
        return synth._bamboo_substrate(rng, h, w)


def index_images(root, limit=None):
    """All usable photographs under a directory, in a stable order."""
    root = Path(root)
    ex = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in ex)
    return files[:limit] if limit else files


def screen(paths, out_json=None, report_every=200):
    """Keep only the photographs that segment into a single slip."""
    keep = []
    for i, p in enumerate(paths):
        if segment_slip(p) is not None:
            keep.append(str(p))
        if report_every and (i + 1) % report_every == 0:
            print(f"  screened {i+1}/{len(paths)}, kept {len(keep)}",
                  flush=True)
    if out_json:
        Path(out_json).write_text(json.dumps(keep))
    return keep
