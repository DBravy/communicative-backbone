"""
Shared publication style for OLMo paper figures.
=================================================

Import this module at the top of each figure script:

    from style import *

Mirrors the Pythia style but adapted for OLMo models and checkpoint schedule.
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
    "1b": "#4477AA",   # blue
}

MARKERS = {
    "1b": "o",
}

LINESTYLES = {
    "1b": "-",
}

MODEL_ORDER = ["1b"]

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

CHECKPOINT_TICKS = [0, 1000, 5000, 10000, 20000, 100000, 500000, 1000000]
CHECKPOINT_LABELS = ["0", "1K", "5K", "10K", "20K", "100K", "500K", "1M"]


def format_training_xaxis(ax):
    """Apply symlog scale with clean power-of-10 ticks for training steps."""
    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlabel("Training step")
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def model_label(key: str) -> str:
    return f"OLMo-{key.upper()}"


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
