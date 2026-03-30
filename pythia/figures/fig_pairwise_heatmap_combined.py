"""
Figure: Pairwise Subspace Overlap Heatmaps (all models combined)
================================================================

3 rows (410m, 1b, 1.4b) × 8 columns (checkpoints).
Each cell is the full NxN top-k mean cosine overlap matrix.
Single shared colorbar on the right.
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
from matplotlib.gridspec import GridSpec

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_b")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_heatmap")
MODELS = ["410m", "1b", "1.4b"]
CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]


def load_model(key):
    path = os.path.join(DATA_DIR, f"pairwise_overlap_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(all_data):
    n_rows = len(MODELS)
    n_cols = len(CHECKPOINTS)

    fig = plt.figure(figsize=(DOUBLE_COL, (SINGLE_H - 0.7) * n_rows + 0.4))
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.08, wspace=0.05,
                  left=0.06, right=0.91, top=0.95, bottom=0.04)

    im = None
    for ri, key in enumerate(MODELS):
        data = all_data[key]
        n_layers = data["n_layers"]

        for ci, step in enumerate(CHECKPOINTS):
            ax = fig.add_subplot(gs[ri, ci])
            matrix = np.array(data["checkpoints"][str(step)]["overlap_matrix"])

            im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="inferno",
                           origin="upper", aspect="equal")

            ax.set_title(CHECKPOINT_LABELS[ci], fontsize=7, pad=2)
            # Model name above middle panel (independently positioned)
            if ci == n_cols // 2:
                ax.text(0.5, 1.22, model_label(key), fontsize=9,
                        ha="center", va="bottom", transform=ax.transAxes)

            # Y-axis on leftmost column
            if ci == 0:
                tick_pos = np.linspace(0, n_layers - 1, min(5, n_layers), dtype=int)
                ax.set_yticks(tick_pos)
                ax.tick_params(labelsize=5)
                ax.set_ylabel("Layer", fontsize=7)
            else:
                ax.set_yticks([])

            ax.set_xticks([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine (top-{all_data[MODELS[0]]['k']})", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data)
    save(fig, "fig_pairwise_heatmap_combined", out_dir=OUT_DIR)
    plt.close(fig)
