"""
Figure: Pairwise Subspace Overlap Heatmaps (all checkpoints)
==============================================================

One figure per model. Each figure shows a row of 8 heatmaps, one per
checkpoint, showing the full NxN top-k mean cosine overlap matrix.

Requires pairwise overlap data computed by compute_pairwise_overlap.py.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, DOUBLE_COL, SINGLE_H, model_label, save,
    CHECKPOINT_LABELS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_b")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_heatmap")
MODELS = ["410m", "1b", "1.4b"]
CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]


def load_model(key):
    path = os.path.join(DATA_DIR, f"pairwise_overlap_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data, key):
    n_cols = len(CHECKPOINTS)
    n_layers = data["n_layers"]

    fig, axes = plt.subplots(1, n_cols,
                             figsize=(DOUBLE_COL, SINGLE_H - 0.2))

    for ci, step in enumerate(CHECKPOINTS):
        ax = axes[ci]
        matrix = np.array(data["checkpoints"][str(step)]["overlap_matrix"])

        im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="inferno",
                       origin="upper", aspect="equal")

        ax.set_title(CHECKPOINT_LABELS[ci], fontsize=7)

        if ci == 0:
            tick_pos = np.linspace(0, n_layers - 1, min(5, n_layers), dtype=int)
            ax.set_yticks(tick_pos)
            ax.tick_params(labelsize=5)
            ax.set_ylabel("Layer", fontsize=7)
        else:
            ax.set_yticks([])

        ax.set_xticks([])

    fig.suptitle(model_label(key), fontsize=9, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.93, 1])

    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine (top-{data['k']})", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    return fig


if __name__ == "__main__":
    for key in MODELS:
        data = load_model(key)
        print(f"Loaded {key}")
        fig = make_figure(data, key)
        save(fig, f"fig_pairwise_heatmap_{key}", out_dir=OUT_DIR)
        plt.close(fig)
