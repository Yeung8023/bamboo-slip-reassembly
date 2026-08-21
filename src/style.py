"""Shared figure style for the manuscript.

Colours come from a palette validated for colour-vision deficiency: the three
all-pairs-safe slots are used wherever marks overlap (Pareto scatter, maps), and
the five adjacent-safe slots for grouped comparisons (boxplots). Because three
of these sit below 3:1 contrast on white, every figure carries either a legend
with direct labels or an accompanying table, and marker shape duplicates colour
so identity never rests on hue alone.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Refined palette.
#
# The optimisation methods form an *ordered* series, because each one adds a
# constraint to the one before it, so they take a sequential ramp rather than
# arbitrary hues. Ordered data encoded with an ordered ramp reads correctly at
# a glance and is what separates a figure that looks designed from one that
# looks assembled. The proposed method takes a warm accent, so it separates
# from the series it completes. The two heuristics are held in grey, because
# they are context rather than the comparison of interest.

INK = "#111114"
# Secondary text is still text: a reader has to read it, so it is black rather
# than a lighter grey. Only the chrome, which is the grid and the axis line,
# stays light, because it is reference and not content.
INK_SECONDARY = "#111114"
MUTED = "#2b2b30"
GRID = "#dcdce2"
AXIS = "#9a9aa4"

# ordered ramp, weakest to strongest constraint set
RAMP = ["#8fabc9", "#5f88b4", "#3a679c", "#1f4a7a"]
ACCENT = "#c04a2f"        # the proposed method
ACCENT_SOFT = "#e0a03c"   # secondary highlight
GREY_1, GREY_2 = "#7c7c88", "#3a3a44"

# kept for figures that need distinct categorical marks
BLUE = "#2a6db5"
ORANGE = "#c8622f"
AQUA = "#2f8f7a"
YELLOW = "#d9a441"
MAGENTA = "#b06a8f"

OVERLAP_SAFE = [BLUE, ORANGE, AQUA]
GROUPED_SAFE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

MARKERS = ["o", "s", "^", "D", "v", "P"]

# Sequential ramp for the preservation ladder, which is also ordered.
SEQ_BLUE = ["#dbe6f2", "#b5cce3", "#8fb2d4", "#6998c4", "#457cae",
            "#2d5f8f", "#1b4269"]

METHOD_COLORS = {
    "top1": GREY_1,
    "mutual": GREY_2,
    "matching": RAMP[0],
    "morph": RAMP[1],
    "length": RAMP[2],
    "full": RAMP[3],
    "latent": ACCENT,
}
METHOD_LABELS = {
    "latent": "Latent-slip assembly (proposed)",
    "full": "Pairwise-constraint assembly",
    "length": "+ slip length",
    "morph": "+ morphometry & context",
    "matching": "Bipartite matching (uniqueness only)",
    "mutual": "Mutual best match",
    "top1": "Top-1 candidate",
}
# Order used in every legend and table, weakest first.
METHOD_ORDER = ["top1", "mutual", "matching", "morph", "length", "full", "latent"]

# Line weight carries emphasis as well as colour.
METHOD_LW = {m: 1.5 for m in METHOD_ORDER}
METHOD_LW["latent"] = 2.4
METHOD_LW["matching"] = 1.9

STATE_LABELS = {
    "P1": "P1 well preserved", "P2": "P2 good", "P3": "P3 typical",
    "P4": "P4 poor", "P5": "P5 very poor",
}


def apply() -> None:
    """Install the manuscript rcParams. Call once before plotting."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,          # journal requirement for line art
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.0,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        # Recessive chrome: the data should be the darkest thing on the page.
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
    })


# Width, in inches, that a figure actually occupies on the page. The
# manuscript places figures at 0.95\textwidth and \textwidth is 372pt, so a
# figure drawn on a wider canvas is reduced by the ratio of the two before a
# reader ever sees it. Typography has to be authored for the printed size, not
# for the canvas: a 9pt label on an 11 inch canvas prints at 4pt, which is
# below what any journal accepts.
PLACED_IN = 0.95 * 372.0 / 72.0

# Target sizes as printed. npj Heritage Science follows the Nature Portfolio
# guidance of 5pt minimum and 7pt preferred for figure text.
PRINTED = {
    "font": 7.0, "title": 8.0, "label": 7.0, "legend": 6.2, "tick": 6.2,
    "linewidth": 0.85, "markersize": 2.4, "axes_lw": 0.42, "grid_lw": 0.34,
    "tick_len": 1.8,
}


def scale_for(fig_width_in: float, placed_in: float = PLACED_IN) -> float:
    """Set typography so a figure drawn this wide prints at PRINTED sizes.

    Returns the scale factor, so that sizes passed directly to a plotting call
    (marker size, cap size, an explicit fontsize) can be scaled with it.
    """
    s = float(fig_width_in) / float(placed_in)
    mpl.rcParams.update({
        "font.size": PRINTED["font"] * s,
        "axes.titlesize": PRINTED["title"] * s,
        "axes.labelsize": PRINTED["label"] * s,
        "legend.fontsize": PRINTED["legend"] * s,
        "xtick.labelsize": PRINTED["tick"] * s,
        "ytick.labelsize": PRINTED["tick"] * s,
        "lines.linewidth": PRINTED["linewidth"] * s,
        "lines.markersize": PRINTED["markersize"] * s,
        "axes.linewidth": PRINTED["axes_lw"] * s,
        "grid.linewidth": PRINTED["grid_lw"] * s,
        "xtick.major.width": 0.6 * s,
        "ytick.major.width": 0.6 * s,
    })
    return s



def legend_below(fig, ax, ncol=4, bottom=0.30):
    """One shared legend under the panels.

    A legend inside an axes has to sit somewhere, and on a panel printed 1.4 in
    wide every corner already holds a curve. Below the panels is the only
    placement that cannot cross the data.
    """
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom)
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.5,
               columnspacing=1.4)


def cjk_font() -> str | None:
    """Locate an installed CJK face so site names render in figures."""
    from matplotlib import font_manager
    for want in ("Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Zen Hei",
                 "Source Han Sans SC", "SimHei", "Microsoft YaHei"):
        for f in font_manager.fontManager.ttflist:
            if want.lower() in f.name.lower():
                return f.name
    return None


def use_cjk() -> bool:
    """Enable CJK rendering if a suitable font exists; report whether it worked."""
    name = cjk_font()
    if name:
        mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
        return True
    return False


def finish(ax, *, grid_axis: str = "y") -> None:
    """Apply the recessive grid convention to an axes."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, alpha=0.7)
    ax.tick_params(length=PRINTED["tick_len"]
                   * mpl.rcParams["font.size"] / PRINTED["font"])


def tex_label(s: str) -> str:
    """Escape a display label for LaTeX.

    METHOD_LABELS is written for matplotlib, which draws the string as it
    stands, so the escaping happens here at the boundary rather than in the
    label itself.
    """
    return s.replace("&", r"\&")


PANEL_LETTERS = "abcdefghij"


def panel_titles(axes, titles=None, *, pad=None) -> None:
    """Tag each panel a), b), c) and left align the title.

    npj Heritage Science requires multi-panel figures to be presented on one
    page with each panel labelled in the a), b), c) convention, and each panel
    described individually in the legend. Where a panel already carries a
    title the letter is prefixed to it; where it does not, the letter stands
    alone.
    """
    for k, ax in enumerate(axes):
        t = ax.get_title() if titles is None else titles[k]
        # A centred title and a left title are separate objects in matplotlib,
        # so the centred one has to be cleared or both would be drawn.
        ax.set_title("")
        lab = f"{PANEL_LETTERS[k]})  {t}".rstrip()
        ax.set_title(lab, loc="left", **({} if pad is None else {"pad": pad}))
