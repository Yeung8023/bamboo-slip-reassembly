"""Method figure: fragments in, architecture, reconstructed slips out.

Laid out as a U -- the learned half left to right along the top, the flow
turning down at the right, the combinatorial half right to left along the
bottom, ending with the reconstruction at the lower left.

A figure is only as good as its smallest legible element, so the canvas
deliberately carries little: six stages, each drawn large, with the model
detail placed on the drawing itself (operators on the arrows, channel counts on
the slabs, tensor shapes beneath them, and the constraints called out on the
chain they act on) rather than in any legend or side table.

Everything shown is real: the fragments come from the generator, the
compatibility block is the trained matcher's own output, and the assembled
slips are the solver's actual solution on that instance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import (FancyArrowPatch, Polygon, FancyBboxPatch,
                                Rectangle, Circle)

import assemble
import matcher as M
import metrics
import preservation as pres
import style
import synth

PXMM = synth.PX_PER_MM
CW, CH = 80.0, 62.0
Y1, YH1 = 43.0, 57.4          # top row: baseline, header
Y2, YH2 = 14.5, 28.8          # bottom row
INK, SEC = style.INK, style.INK_SECONDARY
GREY = "#b3b0a8"
TINT_L = "#f2f7fd"
TINT_C = "#fdf6ef"

FS_LETTER, FS_TITLE, FS_SUB = 13.0, 10.2, 7.4
FS_BODY, FS_SMALL, FS_TINY = 8.0, 7.2, 6.6


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def shade(c, f):
    c = c.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(min(255, int(v * f)) for v in (r, g, b))


def slab(ax, x, y, w, h, d, face, lw=0.9, ch=None, dim=None, z=5):
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                         closed=True, fc=face, ec=INK, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + d, y + h + d),
                          (x + d, y + h + d)], closed=True,
                         fc=shade(face, 1.22), ec=INK, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d),
                          (x + w + d, y + h + d), (x + w, y + h)], closed=True,
                         fc=shade(face, 0.74), ec=INK, lw=lw, zorder=z))
    if ch:
        ax.text(x + w / 2, y + h / 2, ch, ha="center", va="center",
                fontsize=FS_SMALL, color="white", fontweight="bold", zorder=z + 1)
    if dim:
        ax.text(x + w / 2 + d / 2, y - 1.25, dim, ha="center", fontsize=FS_TINY,
                color=SEC, zorder=z)


def op_arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.0, color=SEC, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y + 0.4, label, ha="center", va="bottom",
                fontsize=FS_TINY, color=SEC, linespacing=1.0)


def flow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=20, lw=2.6, zorder=8,
                                 color="#6f6c66"))


def header(ax, x, yh, letter, title, sub=None):
    # npj Heritage Science labels panels a), b), c), so the letter carries its
    # bracket and the title is shifted clear of it.
    ax.text(x, yh, f"{letter})", fontsize=FS_LETTER, fontweight="bold",
            color=INK, ha="left", va="baseline")
    ax.text(x + 3.4, yh, title, fontsize=FS_TITLE, fontweight="bold", color=INK,
            ha="left", va="baseline")
    if sub:
        ax.text(x + 3.4, yh - 2.1, sub, fontsize=FS_SUB, color=SEC, ha="left",
                va="baseline")


def bracket(ax, x0, y0, x1, y1, label, off=0.8, fs=None, color=None):
    c = color or SEC
    dx, dy = x1 - x0, y1 - y0
    n = np.hypot(dx, dy)
    px, py = -dy / n * off, dx / n * off
    ax.plot([x0 + px, x1 + px], [y0 + py, y1 + py], color=c, lw=0.9, zorder=6)
    for (a_, b_) in ((x0, y0), (x1, y1)):
        ax.plot([a_, a_ + 2 * px], [b_, b_ + 2 * py], color=c, lw=0.9, zorder=6)
    ax.text((x0 + x1) / 2 + 2.4 * px, (y0 + y1) / 2 + 2.4 * py, label,
            ha="center", va="center", fontsize=fs or FS_BODY, color=c, zorder=6)


def node(ax, x, y, w, h, fc, lw=1.0, z=5, label=None, fs=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.12,rounding_size=0.34",
                                fc=fc, ec=INK, lw=lw, zorder=z))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs or FS_BODY, color=INK, zorder=z + 1)


def callout(ax, xy, xytext, text, color=None, fs=None, ha="left"):
    """A leader line onto the thing being described, so the explanation sits on
    the drawing rather than beside it."""
    c = color or INK
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs or FS_TINY, color=c,
                ha=ha, va="center", zorder=9, linespacing=1.3,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=c,
                                shrinkA=1, shrinkB=3))


def band(ax, x0, y0, x1, y1, fc, tag, tagcolor, corner="tr"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.0,rounding_size=1.4",
                                fc=fc, ec="none", zorder=0))
    if corner == "tr":
        ax.text(x1 - 1.2, y1 - 1.5, tag, fontsize=FS_TINY, color=tagcolor,
                fontweight="bold", va="center", ha="right", zorder=1)
    else:
        ax.text(x1 - 1.2, y0 + 1.3, tag, fontsize=FS_TINY, color=tagcolor,
                fontweight="bold", va="center", ha="right", zorder=1)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _descs(inst):
    F = inst["frags"]
    return dict(bot_prof=np.array([f["bot_prof"] for f in F]),
                bot_patch=np.array([f["bot_patch"] for f in F]),
                top_prof=np.array([f["top_prof"] for f in F]),
                top_patch=np.array([f["top_patch"] for f in F]))


def rgba(g):
    return np.dstack([g["img"]] * 3 + [g["mask"]])


def build(model, state="P3", n_slips=45, seed=17):
    D = pres.damage(state)
    ci = synth.generate_instance(
        synth.InstanceSpec(n_slips=200, seed=900000, damage=D))
    cal = assemble.fit_calibration(M.score_matrix(model, _descs(ci)), ci["joins"])
    inst = synth.generate_instance(
        synth.InstanceSpec(n_slips=n_slips, seed=seed, damage=D),
        render_gallery=10000)
    meta = [{k: v for k, v in f.items()
             if k not in ("top_prof", "bot_prof", "top_patch", "bot_patch")}
            for f in inst["frags"]]
    S = M.score_matrix(model, _descs(inst))
    W_ = assemble.log_odds(S, cal)
    cfg = assemble.AssemblyConfig(time_limit=120, workers=8, theta=1.73,
                                  use_latent=True)
    r = assemble.solve_cpsat(assemble.build_problem(meta, W_, cfg))
    chains = assemble.chains_from_joins(len(meta), r["joins"])
    truth = {tuple(c) for c in metrics.ground_truth_chains(len(meta), inst["joins"])}
    return inst, S, chains, truth


# --------------------------------------------------------------------------
# top row
# --------------------------------------------------------------------------

def stage_a(ax, inst, gal, ids, x0):
    header(ax, x0, YH1, "a", "Fragments", "excavated, unordered")
    spots = [(x0 + 3.6, Y1 + 3.6, -24), (x0 + 8.6, Y1 + 4.8, 30),
             (x0 + 2.9, Y1 - 4.2, 13), (x0 + 8.0, Y1 - 5.0, -36),
             (x0 + 12.0, Y1 + 0.1, 8)]
    for (cx, cy, ang), fid in zip(spots, ids):
        im = rgba(gal[fid])
        hmm, wmm = im.shape[0] / PXMM, im.shape[1] / PXMM
        sc = 9.6 / max(hmm, 1.0)
        tr = mtransforms.Affine2D().rotate_deg(ang).translate(cx, cy) + ax.transData
        ax.imshow(im, extent=(-wmm * sc / 2, wmm * sc / 2,
                              -hmm * sc / 2, hmm * sc / 2),
                  transform=tr, interpolation="bilinear", zorder=3)
    ax.text(x0 + 7.4, Y1 - 10.2, f"{len(inst['frags'])} fragments",
            ha="center", fontsize=FS_SMALL, color=SEC)
    return x0 + 13.6


def stage_b(ax, inst, ids, x0):
    header(ax, x0, YH1, "b", "Descriptor", "two channels")
    f0 = inst["frags"][ids[0]]
    bx, w, h, d = x0 + 1.9, 7.0, 5.4, 1.6

    axt = ax.inset_axes([bx, Y1 - 2.6, w, h], transform=ax.transData, zorder=4)
    axt.imshow(f0["bot_patch"], cmap="gray", aspect="auto")
    axt.set_xticks([]); axt.set_yticks([])
    for s in axt.spines.values():
        s.set_color(INK); s.set_linewidth(0.9)
    ax.add_patch(Polygon([(bx, Y1 - 2.6 + h), (bx + w, Y1 - 2.6 + h),
                          (bx + w + d, Y1 - 2.6 + h + d),
                          (bx + d, Y1 - 2.6 + h + d)],
                         closed=True, fc="#d5d2ca", ec=INK, lw=0.9, zorder=3))

    axp = ax.inset_axes([bx + d, Y1 + 4.4, w, 3.6], transform=ax.transData,
                        zorder=4)
    axp.plot(np.arange(synth.CANON_W), f0["bot_prof"], color=style.BLUE, lw=1.7)
    axp.set_xticks([]); axp.set_yticks([])
    axp.set_facecolor("#e9f1fd")
    for s in axp.spines.values():
        s.set_color(INK); s.set_linewidth(0.9)

    ax.text(bx + w + d - 0.3, Y1 + 8.6, "ch 1 silhouette", fontsize=FS_TINY,
            color=style.BLUE, va="bottom", ha="right")
    ax.text(bx + w / 2, Y1 - 3.9, "ch 0 texture", fontsize=FS_TINY,
            color=INK, va="top", ha="center")
    bracket(ax, bx - 0.5, Y1 - 2.6, bx - 0.5, Y1 + 2.8, "32", off=0.7,
            fs=FS_TINY)
    ax.text(bx + w / 2, Y1 - 6.6, "$2\\times32\\times48$", ha="center",
            fontsize=FS_BODY + 0.6, color=INK, fontweight="bold")
    return x0 + 12.6


def stage_c(ax, x0):
    header(ax, x0, YH1, "c", "Encoder",
           "two towers, shared weights · height collapsed, width kept")
    specs = [(1.65, 5.6, 0.85, "64", "16×48", "3×3\n$\\downarrow$2"),
             (1.8, 4.2, 0.95, "128", "8×48", "3×3\n$\\downarrow$2"),
             (1.95, 2.9, 1.05, "128", "4×48", "3×3\n$\\downarrow$2"),
             (2.1, 1.8, 1.15, "128", "1×48", "3×3\n$\\downarrow$4"),
             (1.8, 1.2, 0.95, "128", "24", "1d\n$\\downarrow$2")]
    tx0 = x0 + 3.4
    emb_x = None
    for y0, col, lbl in [(Y1 + 2.2, style.BLUE, "lower break of fragment $i$"),
                         (Y1 - 9.4, style.ORANGE, "upper break of fragment $j$")]:
        x = tx0
        for w, h, d, ch, dim, op in specs:
            op_arrow(ax, x - 2.15, x - 0.4, y0 + 2.8, op)
            slab(ax, x, y0 + (5.6 - h) / 2, w, h, d, col, ch=ch, dim=dim)
            x += w + 2.5
        ax.text(tx0 - 2.9, y0 + 7.3, lbl, fontsize=FS_BODY, color=col,
                fontweight="bold", ha="left")
        ax.add_patch(Polygon([(x - 0.8, y0 + 0.6), (x + 0.7, y0 + 1.6),
                              (x + 0.7, y0 + 4.0), (x - 0.8, y0 + 5.0)],
                             closed=True, fc=shade(col, 0.88), ec=INK, lw=0.9,
                             zorder=5))
        op_arrow(ax, x - 2.15, x - 0.95, y0 + 2.8)
        ax.text(x - 0.05, y0 - 1.25, "3072-512-128", ha="center",
                fontsize=FS_TINY, color=SEC)
        for q in range(8):
            ax.add_patch(Rectangle((x + 1.8, y0 + 0.75 + q * 0.52), 0.85, 0.4,
                                   fc=shade(col, 1.0 + 0.02 * q), ec=INK,
                                   lw=0.6, zorder=6))
        ax.text(x + 2.23, y0 + 5.4, "$\\ell_2$", ha="center",
                fontsize=FS_SMALL, color=col, fontweight="bold")
        emb_x = x + 2.7

    dot_x = emb_x + 3.0
    for y0 in (Y1 + 5.0, Y1 - 6.6):
        ax.add_patch(FancyArrowPatch(
            (emb_x + 0.1, y0), (dot_x - 1.15, Y1 + 0.9 if y0 > Y1 else Y1 - 0.9),
            arrowstyle="-", lw=1.2, color=SEC, zorder=5,
            connectionstyle="arc3,rad=0.16"))
    ax.add_patch(Circle((dot_x, Y1), 0.62, fc=INK, ec="none", zorder=8))
    ax.text(dot_x + 1.3, Y1 + 0.9, "$s_{ij}$", ha="left", va="center",
            fontsize=FS_TITLE + 1.0, color=INK, fontweight="bold")
    ax.text(dot_x + 1.3, Y1 - 1.4, "$=\\langle z_i,\\,z_j\\rangle$", ha="left",
            va="center", fontsize=FS_SMALL, color=SEC)
    return dot_x + 1.4


# --------------------------------------------------------------------------
# bottom row, right to left
# --------------------------------------------------------------------------

def stage_d(ax, S, ids, xr):
    w = 13.4
    x0 = xr - w
    header(ax, x0, YH2, "d", "Compatibility", "noisy, conflicting")
    sel = ids[:14]
    sub = S[np.ix_(sel, sel)].copy()
    np.fill_diagonal(sub, np.nan)
    axh = ax.inset_axes([x0 + 2.2, Y2 - 4.8, 8.8, 8.8], transform=ax.transData,
                        zorder=4)
    axh.imshow(sub, cmap="magma", interpolation="nearest")
    axh.set_xticks([]); axh.set_yticks([])
    axh.set_xlabel("upper break $j$", fontsize=FS_TINY, labelpad=2)
    axh.set_ylabel("lower break $i$", fontsize=FS_TINY, labelpad=2)
    ax.text(x0 + 6.6, Y2 + 9.8, "$w_{ij}=a\\,s_{ij}+b-\\theta$",
            fontsize=FS_BODY + 1.4, ha="center", color=INK)
    ax.text(x0 + 6.6, Y2 + 8.2, "calibrated log-odds,\nless the cost of a claim",
            fontsize=FS_TINY, ha="center", color=SEC, va="top",
            linespacing=1.25)
    ax.text(x0 + 6.6, Y2 - 7.8, "top-$k$ kept per break end", fontsize=FS_TINY,
            ha="center", color=SEC)
    return x0


def stage_e(ax, xr):
    w = 39.0
    x0 = xr - w
    header(ax, x0 + 4.6, YH2, "e", "Assembly",
           "constrained maximum-weight path cover · exact CP-SAT")
    ax.text(x0 + 5.2, Y2 + 9.6, "$\\max\\;\\sum_{(i,j)\\in A} w_{ij}\\,x_{ij}$",
            fontsize=FS_TITLE + 0.8, ha="left", color=INK)

    nx0, ny = x0 + 0.4, Y2 + 0.4
    nw, nh = 3.7, 2.3
    pos = [(nx0, ny + 2.8), (nx0 + 6.0, ny + 4.6), (nx0 + 12.0, ny + 2.8),
           (nx0 + 3.0, ny - 3.4), (nx0 + 9.0, ny - 4.6)]
    for k, (px, py) in enumerate(pos):
        node(ax, px, py, nw, nh, "#f1efe9", label=f"$f_{k+1}$", fs=FS_SMALL)
    for i, j in [(0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (1, 2)]:
        ax.add_patch(FancyArrowPatch(
            (pos[i][0] + nw / 2, pos[i][1]),
            (pos[j][0] + nw / 2, pos[j][1] + nh),
            arrowstyle="-", lw=0.9, color=GREY, ls=(0, (2.4, 1.8)), zorder=3,
            connectionstyle="arc3,rad=0.12"))
    for i, j in [(0, 1), (1, 2)]:
        ax.add_patch(FancyArrowPatch(
            (pos[i][0] + nw, pos[i][1] + nh / 2), (pos[j][0], pos[j][1] + nh / 2),
            arrowstyle="-|>", mutation_scale=14, lw=2.7, color=style.BLUE,
            zorder=6, connectionstyle="arc3,rad=-0.16"))
    ax.text(nx0 + 4.4, ny + 8.0, "$x_{ij}=1$", fontsize=FS_BODY,
            color=style.BLUE, ha="center")
    ax.text(nx0 + 6.6, ny - 7.2, "dashed: candidate arcs\nblue: selected",
            fontsize=FS_SMALL, color=SEC, ha="center", va="top",
            linespacing=1.25)

    rx = nx0 + 19.0
    top, bot = ny + 5.4, ny - 4.4
    hs = [(top - bot) * f for f in (0.34, 0.28, 0.24)]
    y = top
    ys = []
    for k, hh in enumerate(hs):
        ax.add_patch(Rectangle((rx, y - hh), 2.4, hh, fc="#e4e0d6", ec=INK,
                               lw=1.0, zorder=5))
        ax.text(rx + 1.2, y - hh / 2, f"$f_{k+1}$", ha="center", va="center",
                fontsize=FS_SMALL, zorder=6)
        ax.plot([rx - 0.55, rx - 0.55], [y, top], color=style.BLUE, lw=1.0,
                zorder=5)
        ys.append(y - hh / 2)
        y -= hh + 0.3
    ax.text(rx - 1.0, (ys[1] + top) / 2, "$\\pi$", ha="right", va="center",
            fontsize=FS_BODY, color=style.BLUE)
    bracket(ax, rx + 2.9, bot, rx + 2.9, top, "$\\Lambda$", off=-0.55,
            fs=FS_TITLE, color=style.ORANGE)
    cordy = []
    for f in (0.16, 0.66):
        yy = bot + (top - bot) * f
        ax.add_patch(Rectangle((rx - 0.12, yy - 0.5), 2.65, 1.0, fc="none",
                               ec=style.ORANGE, lw=1.2, zorder=7))
        cordy.append(yy)

    tx = rx + 6.4
    callout(ax, (rx + 1.2, ys[1]), (tx, top + 1.2),
            "one arc per break end;\n$\\pi_j=\\pi_i+h_i$ fixes the\n"
            "order and forbids cycles", color=style.BLUE)
    callout(ax, (rx + 3.5, (top + bot) / 2), (tx, (top + bot) / 2 - 0.8),
            "the chain must fit inside\nits own slip $\\Lambda$",
            color=style.ORANGE)
    callout(ax, (rx + 2.53, cordy[0]), (tx, bot - 1.8),
            "a notch only where\na cord ran", color=style.ORANGE)
    ax.text(rx + 1.2, bot - 1.4, "$\\Lambda,\\,\\Omega,\\,\\eta$ shared",
            ha="center", va="top", fontsize=FS_TINY, color=style.ORANGE)
    return x0


def stage_f(ax, gal, show, xr):
    w = 12.4
    x0 = xr - w
    header(ax, x0, YH2, "f", "Reconstruction", "ordered, grouped")
    lane = w / len(show)
    gap = 0.24
    top, bot = Y2 + 7.8, Y2 - 6.4
    span = top - bot
    for j, c in enumerate(show):
        tot = sum(gal[i]["img"].shape[0] / PXMM for i in c) + 3.0 * (len(c) - 1)
        cx0 = x0 + j * lane + lane * 0.30
        cx1 = cx0 + lane * 0.42
        y = top
        for fid in c:
            g = gal[fid]
            hh = span * (g["img"].shape[0] / PXMM) / tot
            ax.imshow(rgba(g), extent=(cx0, cx1, y - hh, y), aspect="auto",
                      interpolation="bilinear", zorder=3)
            y -= hh + gap
        ax.plot([cx0 - 0.45, cx0 - 0.45], [y + gap, top], color=style.BLUE,
                lw=2.8, solid_capstyle="butt", zorder=4)
        ax.text((cx0 + cx1) / 2, bot - 0.6, f"{len(c)}", ha="center",
                fontsize=FS_TINY, color=SEC, va="top")
    ax.text(x0 + w / 2, bot - 2.2, "fragments per slip", ha="center",
            fontsize=FS_TINY, color=SEC)
    ax.text(x0 + w / 2, bot - 3.9, "blue: recovered exactly", ha="center",
            fontsize=FS_SMALL, color=style.BLUE, fontweight="bold")
    return x0


# --------------------------------------------------------------------------

def make(model, out, dpi=600):
    style.apply()
    inst, S, chains, truth = build(model)
    gal = {g["frag_id"]: g for g in inst["gallery"]}
    good = [c for c in chains if len(c) >= 2 and tuple(c) in truth]
    by_len = defaultdict(list)
    for c in good:
        by_len[len(c)].append(c)
    show = [by_len[L][0] for L in sorted(by_len, reverse=True)][:3]
    ids = [i for c in show for i in c]

    figw = 7.4
    s = style.scale_for(figw, key=Path(out).stem)
    # The panel is printed at 0.95\textwidth, so a size given here is divided
    # by s before a reader sees it. The constants above were chosen on the
    # canvas rather than on the page, which put the smallest of them at 4.4pt
    # in print; they are lifted so that the smallest prints at about 5.2pt.
    global FS_LETTER, FS_TITLE, FS_SUB, FS_BODY, FS_SMALL, FS_TINY
    k = 5.8 * s / FS_TINY
    FS_LETTER, FS_TITLE, FS_SUB = FS_LETTER * k, FS_TITLE * k, FS_SUB * k
    FS_BODY, FS_SMALL, FS_TINY = FS_BODY * k, FS_SMALL * k, FS_TINY * k
    fig = plt.figure(figsize=(figw, figw * CH / CW))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CW); ax.set_ylim(0, CH)
    ax.set_aspect("equal"); ax.axis("off")

    band(ax, 0.3, 30.8, CW - 0.3, CH - 0.4, TINT_L,
         "LEARNED  ·  pairwise scoring", "#8fb0da")
    band(ax, 0.3, 0.4, CW - 0.3, 30.2, TINT_C,
         "COMBINATORIAL  ·  global assembly", "#e0a678", corner="br")

    xa = stage_a(ax, inst, gal, ids, 1.4)
    flow(ax, xa + 0.5, xa + 4.2, Y1)
    xb = stage_b(ax, inst, ids, xa + 5.0)
    flow(ax, xb + 0.5, xb + 4.2, Y1)
    xc = stage_c(ax, xb + 5.0)

    turn = CW - 2.4
    ax.add_patch(FancyArrowPatch((turn, Y1 - 4.0), (turn, Y2 + 9.4),
                                 arrowstyle="-|>", mutation_scale=20, lw=2.6,
                                 color="#6f6c66", zorder=8))

    xd = stage_d(ax, S, ids, turn - 5.2)
    flow(ax, xd - 0.5, xd - 4.2, Y2)
    xe = stage_e(ax, xd - 5.0)
    flow(ax, xe - 0.5, xe - 4.2, Y2)
    stage_f(ax, gal, show, xe - 5.0)

    fig.savefig(out, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out} | top ends {xc:.1f}, bottom ends "
          f"{xe - 5.0 - 12.4:.1f}, canvas {CW}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--out", default="figs/fig0_overview.png")
    a = ap.parse_args()
    Path("figs").mkdir(exist_ok=True)
    make(M.load_model(a.model), a.out)
