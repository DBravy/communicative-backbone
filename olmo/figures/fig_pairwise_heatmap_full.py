"""
Figure: Pairwise Subspace Overlap Heatmaps — All Checkpoints (OLMo)
====================================================================

One figure with one column per checkpoint, showing the full NxN
top-k mean cosine overlap matrix at each training step.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, DOUBLE_COL, SINGLE_H,
    model_label, save,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_heatmap")


def load_model():
    path = os.path.join(DATA_DIR, "pairwise_overlap_olmo_1b.json")
    with open(path) as f:
        return json.load(f)


def _step_label(step):
    """Human-readable label for a training step."""
    if step >= 1_000_000 and step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    if step >= 1_000 and step % 1_000 == 0:
        return f"{step // 1_000}K"
    return str(step)


def make_figure(data):
    steps = sorted(data["checkpoints"].keys(), key=int)
    n_cols = len(steps)
    n_layers = data["n_layers"]

    fig, axes = plt.subplots(1, n_cols,
                             figsize=(DOUBLE_COL, SINGLE_H - 0.2))

    for ci, step_key in enumerate(steps):
        ax = axes[ci]
        matrix = np.array(data["checkpoints"][step_key]["overlap_matrix"])

        im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="inferno",
                       origin="upper", aspect="equal")

        ax.set_title(_step_label(int(step_key)), fontsize=7)

        if ci == 0:
            tick_pos = np.linspace(0, n_layers - 1, min(5, n_layers), dtype=int)
            ax.set_yticks(tick_pos)
            ax.tick_params(labelsize=5)
            ax.set_ylabel("Layer", fontsize=7)
        else:
            ax.set_yticks([])

        ax.set_xticks([])

    fig.suptitle(model_label("1b"), fontsize=9, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.93, 1])

    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine (top-{data['k']})", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    return fig


if __name__ == "__main__":
    data = load_model()
    print("Loaded OLMo-1B")
    fig = make_figure(data)
    save(fig, "fig_pairwise_heatmap_full_olmo", out_dir=OUT_DIR)
    plt.close(fig)
