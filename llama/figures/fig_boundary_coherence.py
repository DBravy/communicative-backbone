"""
Figure: Per-Boundary Subspace Coherence Across Training — TinyLlama
====================================================================

Heatmap: x-axis is training step, y-axis is layer boundary,
color is top-10 mean cosine overlap.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, SINGLE_COL, SINGLE_H, model_label, step_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_coherence")
K = 10


def load_model():
    path = os.path.join(DATA_DIR, "crosslayer_overlap_tinyllama_1b.json")
    with open(path) as f:
        return json.load(f)


def make_heatmap(data):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H + 0.4))

    n_boundaries = data["n_layers"] - 1
    steps = sorted(data["checkpoints"].keys(), key=int)

    matrix = np.zeros((n_boundaries, len(steps)))
    for ci, s in enumerate(steps):
        pairs = data["checkpoints"][s]["adjacent_pairs"]
        for bi in range(n_boundaries):
            pk = f"{bi}_{bi+1}"
            matrix[bi, ci] = pairs[pk][f"top{K}"]["mean_cosine"]

    im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                   vmin=0.05, vmax=0.45, interpolation="nearest")

    labels = [step_label(int(s)) for s in steps]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_xlabel("Training step", fontsize=7)

    ax.set_yticks(range(n_boundaries))
    ax.set_yticklabels([f"{i}\u2013{i+1}" for i in range(n_boundaries)],
                       fontsize=5)
    ax.set_ylabel("Layer boundary", fontsize=8)

    ax.set_title(model_label(), fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cosine overlap (top-{K})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


def make_lines(data):
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H + 0.2))

    cmap = matplotlib.colormaps["coolwarm"]
    n_layers = data["n_layers"]
    n_boundaries = n_layers - 1

    pair_keys = [f"{i}_{i+1}" for i in range(n_boundaries)]
    norm = mcolors.Normalize(vmin=0, vmax=n_boundaries - 1)

    # Pick ~5 representative boundaries for labels
    reps = list(np.unique(np.linspace(0, n_boundaries - 1, 5, dtype=int)))

    steps = sorted(data["checkpoints"].keys(), key=int)
    step_ints = [int(s) for s in steps]

    for bi, pk in enumerate(pair_keys):
        vals = [data["checkpoints"][s]["adjacent_pairs"][pk][f"top{K}"]["mean_cosine"]
                for s in steps]
        color = cmap(norm(bi))
        is_rep = bi in reps
        ax.plot(step_ints, vals,
                color=color,
                linewidth=1.4 if is_rep else 0.6,
                alpha=1.0 if is_rep else 0.35,
                zorder=3 if is_rep else 2,
                label=f"L{bi}\u2013{bi+1}" if is_rep else None)

    bl = data["random_baselines"][str(K)]
    ax.axhline(bl["mean_cosine_mean"], color="#999999", linestyle=":",
               linewidth=0.7, zorder=1)

    ax.set_xscale("symlog", linthresh=10000)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Cosine overlap (top-{K})")
    ax.set_title(model_label(), fontsize=9)
    ax.legend(loc="upper left", fontsize=5.5, frameon=True,
              framealpha=0.9, edgecolor="none", ncol=1,
              handlelength=1.2)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_model()
    print(f"Loaded {model_label()}")

    fig = make_heatmap(data)
    save(fig, "fig_boundary_coherence_tinyllama", out_dir=OUT_DIR)
    plt.close(fig)

    fig = make_lines(data)
    save(fig, "fig_boundary_coherence_lines_tinyllama", out_dir=OUT_DIR)
    plt.close(fig)
