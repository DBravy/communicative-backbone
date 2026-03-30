"""
Figure: Composed Effective Rank During Training (OLMo)
======================================================

Per-layer effective rank vs training step, plus two comparison views:
  - Multi-line plot: all layers on one axes, colored by depth (coolwarm)
  - Heatmap: layer x checkpoint, color = effective rank
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, COLORS, SINGLE_COL, SINGLE_H, DOUBLE_COL,
    model_label, save,
)
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effective_rank")


def load_model():
    path = os.path.join(DATA_DIR, "svd_emergence_olmo_1b.json")
    with open(path) as f:
        return json.load(f)


def _step_label(step):
    """Human-readable label for a training step."""
    if step >= 1_000_000 and step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    if step >= 1_000 and step % 1_000 == 0:
        return f"{step // 1_000}K"
    return str(step)


def make_figure(data, layer=0):
    """Single-layer effective rank vs training step."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    color = COLORS["1b"]
    layer_key = str(layer)

    steps = sorted(data["checkpoints"].keys(), key=int)
    step_ints = [int(s) for s in steps]
    vals = [data["checkpoints"][s]["layers"][layer_key]["effective_rank"]
            for s in steps]

    ax.plot(step_ints, vals,
            color=color, marker="o", linestyle="-",
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.2, label=model_label("1b"), zorder=3)

    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title(f"Effective rank (layer {layer})", fontsize=9)
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


def make_lines_figure(data):
    """All layers on one plot, colored by depth."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H + 0.3))

    n_layers = data["n_layers"]
    cmap = matplotlib.colormaps["coolwarm"]
    norm = mcolors.Normalize(vmin=0, vmax=n_layers - 1)

    steps = sorted(data["checkpoints"].keys(), key=int)
    step_ints = [int(s) for s in steps]

    # Pick ~5 representative layers for labels
    reps = list(np.unique(np.linspace(0, n_layers - 1, 5, dtype=int)))

    for li in range(n_layers):
        vals = [data["checkpoints"][s]["layers"][str(li)]["effective_rank"]
                for s in steps]
        color = cmap(norm(li))
        is_rep = li in reps
        ax.plot(step_ints, vals,
                color=color,
                linewidth=1.4 if is_rep else 0.6,
                alpha=1.0 if is_rep else 0.35,
                zorder=3 if is_rep else 2,
                label=f"Layer {li}" if is_rep else None)

    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title(f"Effective rank by layer ({model_label('1b')})", fontsize=9)
    ax.legend(loc="best", fontsize=5.5, frameon=True,
              framealpha=0.9, edgecolor="none", ncol=1,
              handlelength=1.2)
    fig.tight_layout()
    return fig


def make_heatmap_figure(data):
    """Heatmap: y-axis = layer, x-axis = checkpoint, color = effective rank."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H + 0.4))

    n_layers = data["n_layers"]
    steps = sorted(data["checkpoints"].keys(), key=int)

    matrix = np.zeros((n_layers, len(steps)))
    for ci, s in enumerate(steps):
        for li in range(n_layers):
            matrix[li, ci] = data["checkpoints"][s]["layers"][str(li)]["effective_rank"]

    im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                   interpolation="nearest", origin="upper")

    # X-axis
    labels = [_step_label(int(s)) for s in steps]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_xlabel("Training step", fontsize=7)

    # Y-axis
    tick_pos = np.linspace(0, n_layers - 1, min(6, n_layers), dtype=int)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_pos, fontsize=6)
    ax.set_ylabel("Layer", fontsize=8)

    ax.set_title(f"Effective rank ({model_label('1b')})", fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Effective rank", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return fig


if __name__ == "__main__":
    data = load_model()
    n_layers = data["n_layers"]
    print(f"OLMo-1B: {n_layers} layers")

    # Per-layer figures
    for li in range(n_layers):
        fig = make_figure(data, layer=li)
        save(fig, f"fig_effective_rank_layer{li}_olmo", out_dir=OUT_DIR)
        plt.close(fig)

    # Comparison: all layers as lines
    fig = make_lines_figure(data)
    save(fig, "fig_effective_rank_lines_olmo", out_dir=OUT_DIR)
    plt.close(fig)

    # Comparison: heatmap
    fig = make_heatmap_figure(data)
    save(fig, "fig_effective_rank_heatmap_olmo", out_dir=OUT_DIR)
    plt.close(fig)
