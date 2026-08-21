"""Measure what each figure occupies on the page, for style.scale_for.

Saving with `bbox_inches="tight"` crops the canvas, so a scale computed from
the figure width alone overstates the reduction and leaves the type larger than
intended, by a different amount for every figure. This measures the width each
saved figure actually has and records it, so that a second render lands every
label at the size `style.PRINTED` asks for.

Run it after the figures are built, then build them once more.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
FIGS = ROOT / "revision" / "figs"
DPI = 600.0


def main():
    widths = {}
    for f in sorted(FIGS.glob("*.png")):
        with Image.open(f) as im:
            widths[f.stem] = round(im.size[0] / DPI, 4)
    (FIGS / "widths.json").write_text(json.dumps(widths, indent=2, sort_keys=True))

    # what those widths mean once the manuscript places them
    tex = (ROOT / "revision" / "paper" / "main.tex").read_text()
    placed = {m.group(2).removesuffix(".png"): float(m.group(1)) * 372.0 / 72.0
              for m in re.finditer(
                  r"includegraphics\[width=([0-9.]+)\\textwidth\]\{([^}]+)\}", tex)}
    print(f"{len(widths)} figures measured -> {FIGS / 'widths.json'}")
    for k, v in sorted(placed.items()):
        if k in widths:
            print(f"  {k:26s} {widths[k]:5.2f} in drawn, {v:4.2f} in printed,"
                  f" x{v / widths[k]:.2f}")


if __name__ == "__main__":
    main()
