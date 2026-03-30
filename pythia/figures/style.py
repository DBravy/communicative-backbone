"""
Shared publication style for all paper figures.
================================================

Import this module at the top of each figure script:

    from style import *

This sets matplotlib rcParams globally and provides the color/marker/linestyle
palette plus common helpers.

Targeting: arXiv / NeurIPS / ICML column widths.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Page geometry (inches)
# ---------------------------------------------------------------------------

SINGLE_COL = 3.4     # NeurIPS / ICML single-column width
DOUBLE_COL = 7.0     # full text width
SINGLE_H   = 2.4     # default height for a single-panel figure
DOUBLE_H   = 4.8     # default height for a 2-row figure

# ---------------------------------------------------------------------------
# rcParams — clean, print-ready defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    # Lines
    "lines.linewidth": 1.4,
    "lines.markersize": 4,

    # Axes
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Ticks
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    # Output
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------------------
# Colorblind-friendly palette  (Paul Tol "bright")
# ---------------------------------------------------------------------------

COLORS = {
    "70m":  "#4477AA",   # blue
    "160m": "#EE6677",   # rose
    "410m": "#228833",   # green
    "1b":   "#CCBB44",   # olive
    "1.4b": "#AA3377",   # purple
}

MARKERS = {
    "70m":  "o",
    "160m": "s",
    "410m": "^",
    "1b":   "D",
    "1.4b": "v",
}

LINESTYLES = {
    "70m":  "-",
    "160m": "--",
    "410m": "-.",
    "1b":   ":",
    "1.4b": "-",
}

MODEL_ORDER = ["70m", "160m", "410m", "1b", "1.4b"]

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

CHECKPOINT_TICKS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]
CHECKPOINT_LABELS = ["0", "128", "512", "2K", "8K", "32K", "64K", "143K"]


def format_training_xaxis(ax):
    """Apply symlog scale with clean power-of-10 ticks for training steps."""
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    # Let symlog place ticks naturally at powers of 10; disable minor ticks
    # to keep it clean (0, 10², 10³, 10⁴, 10⁵).
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def model_label(key: str) -> str:
    return f"Pythia-{key.upper()}"


def save(fig, stem: str, out_dir: str = None):
    """Save figure as both PDF (vector) and PNG (raster)."""
    import os
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"Saved: {path}")
