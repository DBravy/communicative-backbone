"""
Figure: Pairwise Subspace Overlap Heatmaps (OLMo)
===================================================

Three columns: init, early, and final checkpoints.
Entry (i, j) is the top-k mean cosine overlap between layers i and j.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, DOUBLE_COL, DOUBLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_heatmap")
SHOW_STEPS = [0, 1000, 2000, 3000, 5000, 10000, 100000, 1000000]
STEP_LABELS = [
    "Step 0\n(init)", "Step 1K", "Step 2K", "Step 3K",
    "Step 5K", "Step 10K", "Step 100K", "Step 1M\n(trained)",
]


def load_model():
    path = os.path.join(DATA_DIR, "pairwise_overlap_olmo_1b.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data):
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(DOUBLE_COL, DOUBLE_H))

    n_layers = data["n_layers"]

    for ci, step in enumerate(SHOW_STEPS):
        row, col = divmod(ci, n_cols)
        ax = axes[row, col]
        matrix = np.array(data["checkpoints"][str(step)]["overlap_matrix"])

        im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="inferno",
                       origin="upper", aspect="equal")

        # Axis labels
        if col == 0:
            ax.set_ylabel("Layer", fontsize=7)
        if row == n_rows - 1:
            ax.set_xlabel("Layer", fontsize=7)
        ax.set_title(STEP_LABELS[ci], fontsize=8)

        # Ticks
        tick_positions = np.linspace(0, n_layers - 1, min(5, n_layers), dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(labelsize=6)

    fig.suptitle(model_label("1b"), fontsize=9, y=0.92)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Mean cosine (top-{data['k']})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


if __name__ == "__main__":
    data = load_model()
    print("Loaded OLMo-1B")

    fig = make_figure(data)
    save(fig, "fig_pairwise_heatmap_olmo", out_dir=OUT_DIR)
    plt.close(fig)
