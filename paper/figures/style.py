"""Shared figure style: one place where colour and type decisions live.

Palette slots are taken unchanged from the validated reference set. Only the
first three categorical slots are used, because that subset is the one validated
across *all* pair combinations rather than only adjacent pairs, which matters here
since the tools appear together in scatter-like and small-multiple layouts as well
as in bars.

The union of detectors is deliberately not a fourth hue. It is not a peer entity
alongside the three tools but a combination of them, so it carries a neutral ink
and a distinct dash instead of competing for identity in the same colour space.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette (light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

TOOL_COLOR = {"lacuna": BLUE, "fpocket": ORANGE, "p2rank": AQUA,
              "ifsitepred": "#52514e"}
TOOL_LABEL = {"lacuna": "Lacuna", "fpocket": "fpocket", "p2rank": "P2Rank",
              "ifsitepred": "IF-SitePred"}
TOOL_ORDER = ("fpocket", "p2rank", "ifsitepred", "lacuna")

SURFACE = "#ffffff"
INK = "#0b0b0b"          # text-primary
INK_2 = "#52514e"        # text-secondary
INK_MUTED = "#8a8983"    # text-muted, for grid and recessive rules
UNION = "#2f2e2b"        # neutral ink for the combined-detector series

#: Segment fills for the composition figure. Recovered uses the entity colour;
#: the two failure modes are greys so the eye reads them as absence, not identity.
LOST_FILL = "#c9c8c2"
MISSING_FILL = "#ebeae4"

SINGLE_COL = 3.35        # inches, typical journal single column
DOUBLE_COL = 6.9


def use_paper_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e5df",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed as TrueType so editors can set text
        "ps.fonttype": 42,
    })


def save(fig, path_stem: str) -> None:
    """Write both a vector master for submission and a raster for previewing."""
    # 600 dpi so a figure placed at 6.5 in still clears the 300 dpi bioRxiv
    # recommends after any downsampling in conversion.
    for ext, dpi in ((".pdf", None), (".png", 800)):
        fig.savefig(f"{path_stem}{ext}", dpi=dpi)
    plt.close(fig)


def pct_axis(ax, axis="y", upper=1.0):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    if axis == "y":
        ax.set_ylim(0, upper)
    else:
        ax.set_xlim(0, upper)
