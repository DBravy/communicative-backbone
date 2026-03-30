"""
Shared publication style for TinyLlama crossmodel figures.
==========================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Page geometry (inches)
# ---------------------------------------------------------------------------

SINGLE_COL = 3.4
DOUBLE_COL = 7.0
SINGLE_H   = 2.4
DOUBLE_H   = 4.8

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLOR = "#EE6677"  # rose — matches existing crossmodels convention


def model_label():
    return "TinyLlama-1.1B"


def step_label(step):
    """Human-readable label for a training step."""
    if step >= 1_000_000 and step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    if step >= 1_000 and step % 1_000 == 0:
        return f"{step // 1_000}K"
    return str(step)


def save(fig, stem, out_dir=None):
    import os
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"Saved: {path}")
