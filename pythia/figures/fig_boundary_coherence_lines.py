"""
Figure: Per-Boundary Subspace Coherence Across Training (line plot)
===================================================================

One panel per model. Each panel shows top-10 mean cosine at every layer
boundary across training steps, colored by depth position (early → late).
A few representative boundaries are highlighted and labeled; the rest are
shown faintly.

Shows whether all boundaries commit simultaneously or whether there is a
depth-dependent sequence, and whether boundaries diverge during elaboration.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, DOUBLE_COL, SINGLE_H,
    format_training_xaxis, model_label, save,
)
import matplotlib.cm as cm
import matplotlib.colors as mcolors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_b")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_coherence")
MODELS = ["410m", "1b", "1.4b"]
CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]
K = 10


def load_model(key):
    path = os.path.join(DATA_DIR, f"crosslayer_overlap_{key}.json")
    with open(path) as f:
        return json.load(f)


def representative_indices(n_boundaries):
    """Pick ~5 evenly spaced boundary indices for labelling."""
    if n_boundaries <= 5:
        return list(range(n_boundaries))
    idxs = np.linspace(0, n_boundaries - 1, 5, dtype=int)
    return list(np.unique(idxs))


def make_figure(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_H + 0.2),
                             sharey=True)

    cmap = cm.get_cmap("coolwarm")

    for ax, key in zip(axes, MODELS):
        d = all_data[key]
        n_layers = d["n_layers"]
        n_boundaries = n_layers - 1

        pair_keys = [f"{i}_{i+1}" for i in range(n_boundaries)]
        norm = mcolors.Normalize(vmin=0, vmax=n_boundaries - 1)
        reps = representative_indices(n_boundaries)

        for bi, pk in enumerate(pair_keys):
            vals = []
            for step in CHECKPOINTS:
                pairs = d["checkpoints"][str(step)]["adjacent_pairs"]
                vals.append(pairs[pk][f"top{K}"]["mean_cosine"])

            color = cmap(norm(bi))
            is_rep = bi in reps
            ax.plot(CHECKPOINTS, vals,
                    color=color,
                    linewidth=1.4 if is_rep else 0.6,
                    alpha=1.0 if is_rep else 0.35,
                    zorder=3 if is_rep else 2,
                    label=f"L{bi}\u2013{bi+1}" if is_rep else None)

        # Random baseline
        bl = d["random_baselines"][str(K)]
        ax.axhline(bl["mean_cosine_mean"], color="#999999", linestyle=":",
                   linewidth=0.7, zorder=1)

        format_training_xaxis(ax)
        ax.set_title(model_label(key), fontsize=9)
        ax.legend(loc="upper left", fontsize=5.5, frameon=True,
                  framealpha=0.9, edgecolor="none", ncol=1,
                  handlelength=1.2)

    axes[0].set_ylabel(f"Cosine overlap (top-{K})")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data)
    save(fig, "fig_boundary_coherence_lines", out_dir=OUT_DIR)
    plt.close(fig)
