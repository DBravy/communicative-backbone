"""
Figure: Pairwise Subspace Overlap Heatmaps
============================================

One row per model, columns for selected checkpoints (init, early, final).
Entry (i, j) is the top-k mean cosine overlap between layers i and j.

At initialization the matrix is uniform noise. During training, strong
near-diagonal values emerge (adjacent layers aligned) while off-diagonal
values show how much subspaces have rotated across depth: local channels,
not a global subspace.

Requires pairwise overlap data computed by compute_pairwise_overlap.py.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, DOUBLE_COL, DOUBLE_H, model_label, save,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_b")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_heatmap")
MODELS = ["410m", "1b", "1.4b"]
SHOW_STEPS = [0, 512, 143000]
STEP_LABELS = ["Step 0\n(init)", "Step 512\n(early)", "Step 143K\n(trained)"]


def load_model(key):
    path = os.path.join(DATA_DIR, f"pairwise_overlap_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(all_data):
    n_rows = len(MODELS)
    n_cols = len(SHOW_STEPS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(DOUBLE_COL, DOUBLE_H - 0.4))

    for ri, key in enumerate(MODELS):
        d = all_data[key]
        n_layers = d["n_layers"]

        for ci, step in enumerate(SHOW_STEPS):
            ax = axes[ri, ci]
            matrix = np.array(d["checkpoints"][str(step)]["overlap_matrix"])

            im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="inferno",
                           origin="upper", aspect="equal")

            # Axis labels
            if ci == 0:
                ax.set_ylabel(model_label(key), fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=7)
            if ri == 0:
                ax.set_title(STEP_LABELS[ci], fontsize=8)

            # Ticks: show a few layer numbers
            tick_positions = np.linspace(0, n_layers - 1, min(5, n_layers),
                                        dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Mean cosine (top-{all_data[MODELS[0]]['k']})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data)
    save(fig, "fig_pairwise_heatmap", out_dir=OUT_DIR)
    plt.close(fig)
