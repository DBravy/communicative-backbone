"""
Figure: Per-Boundary Subspace Coherence Across Training (OLMo)
===============================================================

Heatmap: x-axis is training step, y-axis is layer boundary (0-1 at top,
final pair at bottom), color is top-10 mean cosine overlap.
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_coherence")
K = 10


def load_model():
    path = os.path.join(DATA_DIR, "crosslayer_overlap_olmo_1b.json")
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
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H + 0.4))

    n_boundaries = data["n_layers"] - 1
    steps = sorted(data["checkpoints"].keys(), key=int)

    # Build matrix: rows = boundaries, cols = checkpoints
    matrix = np.zeros((n_boundaries, len(steps)))
    for ci, step_key in enumerate(steps):
        pairs = data["checkpoints"][step_key]["adjacent_pairs"]
        for bi in range(n_boundaries):
            pk = f"{bi}_{bi+1}"
            matrix[bi, ci] = pairs[pk][f"top{K}"]["mean_cosine"]

    im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                   vmin=0.05, vmax=0.45, interpolation="nearest")

    # X-axis: checkpoint labels
    labels = [_step_label(int(s)) for s in steps]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_xlabel("Training step", fontsize=7)

    # Y-axis: boundary labels
    ax.set_yticks(range(n_boundaries))
    ax.set_yticklabels([f"{i}\u2013{i+1}" for i in range(n_boundaries)],
                       fontsize=5)
    ax.set_ylabel("Layer boundary", fontsize=8)

    ax.set_title(model_label("1b"), fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine overlap (top-{K})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


if __name__ == "__main__":
    data = load_model()
    print("Loaded OLMo-1B")

    fig = make_figure(data)
    save(fig, "fig_boundary_coherence_olmo", out_dir=OUT_DIR)
    plt.close(fig)
