"""
Figure: Per-Boundary Subspace Coherence Across Training — Line Plot (OLMo)
==========================================================================

All boundaries as lines; ~5 representative boundaries highlighted and
labeled. Uses coolwarm colormap for depth-dependent visualization.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, SINGLE_COL, SINGLE_H,
    model_label, save,
)
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.colors as mcolors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_coherence")
K = 10


def load_model():
    path = os.path.join(DATA_DIR, "crosslayer_overlap_olmo_1b.json")
    with open(path) as f:
        return json.load(f)


def representative_indices(n_boundaries):
    """Pick ~5 evenly spaced boundary indices for labelling."""
    if n_boundaries <= 5:
        return list(range(n_boundaries))
    idxs = np.linspace(0, n_boundaries - 1, 5, dtype=int)
    return list(np.unique(idxs))


def make_figure(data):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H + 0.2))

    cmap = matplotlib.colormaps["coolwarm"]
    n_layers = data["n_layers"]
    n_boundaries = n_layers - 1

    pair_keys = [f"{i}_{i+1}" for i in range(n_boundaries)]
    norm = mcolors.Normalize(vmin=0, vmax=n_boundaries - 1)
    reps = representative_indices(n_boundaries)

    steps = sorted(data["checkpoints"].keys(), key=int)
    step_ints = [int(s) for s in steps]

    for bi, pk in enumerate(pair_keys):
        vals = []
        for s in steps:
            pairs = data["checkpoints"][s]["adjacent_pairs"]
            vals.append(pairs[pk][f"top{K}"]["mean_cosine"])

        color = cmap(norm(bi))
        is_rep = bi in reps
        ax.plot(step_ints, vals,
                color=color,
                linewidth=1.4 if is_rep else 0.6,
                alpha=1.0 if is_rep else 0.35,
                zorder=3 if is_rep else 2,
                label=f"L{bi}\u2013{bi+1}" if is_rep else None)

    # Random baseline
    bl = data["random_baselines"][str(K)]
    ax.axhline(bl["mean_cosine_mean"], color="#999999", linestyle=":",
               linewidth=0.7, zorder=1)

    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Cosine overlap (top-{K})")
    ax.set_title(model_label("1b"), fontsize=9)
    ax.legend(loc="upper left", fontsize=5.5, frameon=True,
              framealpha=0.9, edgecolor="none", ncol=1,
              handlelength=1.2)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_model()
    print("Loaded OLMo-1B")

    fig = make_figure(data)
    save(fig, "fig_boundary_coherence_lines_olmo", out_dir=OUT_DIR)
    plt.close(fig)
