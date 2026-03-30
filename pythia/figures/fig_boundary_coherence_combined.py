"""
Figure: Per-Boundary Subspace Coherence (combined)
===================================================

Top row:  line plots showing top-10 mean cosine at every layer boundary
          across training steps, colored by depth (early → late).
Bottom row: heatmaps (boundary × checkpoint) of the same metric.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, DOUBLE_COL, SINGLE_H,
    format_training_xaxis, model_label, save,
    CHECKPOINT_LABELS,
)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

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
    if n_boundaries <= 5:
        return list(range(n_boundaries))
    idxs = np.linspace(0, n_boundaries - 1, 5, dtype=int)
    return list(np.unique(idxs))


def make_figure(all_data):
    fig = plt.figure(figsize=(DOUBLE_COL, SINGLE_H * 2 + 0.6))

    # Top row slightly shorter than bottom (heatmap needs room for labels)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.15],
                  hspace=0.35, wspace=0.28,
                  left=0.07, right=0.91, top=0.94, bottom=0.08)

    line_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    heat_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    cmap_lines = cm.get_cmap("coolwarm")

    # ── Top row: line plots ──────────────────────────────────────────────
    for ax, key in zip(line_axes, MODELS):
        d = all_data[key]
        n_boundaries = d["n_layers"] - 1
        pair_keys = [f"{i}_{i+1}" for i in range(n_boundaries)]
        norm = mcolors.Normalize(vmin=0, vmax=n_boundaries - 1)
        reps = representative_indices(n_boundaries)

        for bi, pk in enumerate(pair_keys):
            vals = []
            for step in CHECKPOINTS:
                pairs = d["checkpoints"][str(step)]["adjacent_pairs"]
                vals.append(pairs[pk][f"top{K}"]["mean_cosine"])

            color = cmap_lines(norm(bi))
            is_rep = bi in reps
            ax.plot(CHECKPOINTS, vals,
                    color=color,
                    linewidth=1.4 if is_rep else 0.6,
                    alpha=1.0 if is_rep else 0.35,
                    zorder=3 if is_rep else 2,
                    label=f"L{bi}\u2013{bi+1}" if is_rep else None)

        bl = d["random_baselines"][str(K)]
        ax.axhline(bl["mean_cosine_mean"], color="#999999", linestyle=":",
                   linewidth=0.7, zorder=1)

        format_training_xaxis(ax)
        ax.set_xlim(left=0)
        ax.set_title(model_label(key), fontsize=9)
        ax.legend(loc="upper left", fontsize=5.5, frameon=True,
                  framealpha=0.9, edgecolor="none", ncol=1,
                  handlelength=1.2)

    line_axes[0].set_ylabel(f"Cosine overlap (top-{K})")
    for ax in line_axes:
        ax.set_xlabel("Training step", fontsize=7)

    # ── Bottom row: heatmaps ─────────────────────────────────────────────
    im = None
    for ax, key in zip(heat_axes, MODELS):
        d = all_data[key]
        n_boundaries = d["n_layers"] - 1

        matrix = np.zeros((n_boundaries, len(CHECKPOINTS)))
        for ci, step in enumerate(CHECKPOINTS):
            pairs = d["checkpoints"][str(step)]["adjacent_pairs"]
            for bi in range(n_boundaries):
                pk = f"{bi}_{bi+1}"
                matrix[bi, ci] = pairs[pk][f"top{K}"]["mean_cosine"]

        im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                       vmin=0.05, vmax=0.45, interpolation="nearest")

        ax.set_title(model_label(key), fontsize=9)
        ax.set_xticks(range(len(CHECKPOINTS)))
        ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=6, rotation=45,
                           ha="right")
        ax.set_xlabel("Training step", fontsize=7)

        if n_boundaries <= 16:
            ax.set_yticks(range(n_boundaries))
            ax.set_yticklabels([f"{i}\u2013{i+1}" for i in range(n_boundaries)],
                               fontsize=5)
        else:
            tick_pos = np.linspace(0, n_boundaries - 1, 8, dtype=int)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels([f"{i}\u2013{i+1}" for i in tick_pos],
                               fontsize=5)

    heat_axes[0].set_ylabel("Layer boundary", fontsize=8)

    # Shared colorbar for heatmaps
    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.38])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine overlap (top-{K})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data)
    save(fig, "fig_boundary_coherence_combined", out_dir=OUT_DIR)
    plt.close(fig)
