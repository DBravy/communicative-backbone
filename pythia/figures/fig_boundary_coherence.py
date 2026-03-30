"""
Figure: Per-Boundary Subspace Coherence Across Training
========================================================

Heatmap: x-axis is training step, y-axis is layer boundary (0–1 at top,
final pair at bottom), color is top-10 mean cosine overlap.

Shows the depth-dependent cascade: first and last boundaries spike early
(step 512), interior boundaries catch up later (~step 2000–8000).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, DOUBLE_COL, SINGLE_H,
    model_label, save,
    CHECKPOINT_TICKS, CHECKPOINT_LABELS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_b")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_coherence")
MODELS = ["410m", "1b", "1.4b"]
CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]
K = 10


def load_model(key):
    path = os.path.join(DATA_DIR, f"crosslayer_overlap_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(all_data):
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models,
                             figsize=(DOUBLE_COL, SINGLE_H + 0.4))

    for ax, key in zip(axes, MODELS):
        d = all_data[key]
        n_boundaries = d["n_layers"] - 1

        # Build matrix: rows = boundaries, cols = checkpoints
        matrix = np.zeros((n_boundaries, len(CHECKPOINTS)))
        for ci, step in enumerate(CHECKPOINTS):
            pairs = d["checkpoints"][str(step)]["adjacent_pairs"]
            for bi in range(n_boundaries):
                pk = f"{bi}_{bi+1}"
                matrix[bi, ci] = pairs[pk][f"top{K}"]["mean_cosine"]

        im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                       vmin=0.05, vmax=0.45, interpolation="nearest")

        # X-axis: checkpoint labels
        ax.set_xticks(range(len(CHECKPOINTS)))
        ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=6, rotation=45,
                           ha="right")
        ax.set_xlabel("Training step", fontsize=7)

        # Y-axis: boundary labels
        if n_boundaries <= 16:
            ax.set_yticks(range(n_boundaries))
            ax.set_yticklabels([f"{i}\u2013{i+1}" for i in range(n_boundaries)],
                               fontsize=5)
        else:
            tick_pos = np.linspace(0, n_boundaries - 1, 8, dtype=int)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels([f"{i}\u2013{i+1}" for i in tick_pos],
                               fontsize=5)

        ax.set_title(model_label(key), fontsize=9)

    axes[0].set_ylabel("Layer boundary", fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine overlap (top-{K})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data)
    save(fig, "fig_boundary_coherence", out_dir=OUT_DIR)
    plt.close(fig)
