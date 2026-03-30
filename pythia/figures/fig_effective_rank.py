"""
Figure: Composed Effective Rank During Training
================================================

Two separate graphs for the three largest Pythia models:
  1. Final layer effective rank vs training step
  2. Layer 0 effective rank vs training step
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, COLORS, SINGLE_COL, SINGLE_H,
    format_training_xaxis, model_label, save,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_a")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effective_rank")
MODELS = ["410m", "1b", "1.4b"]
CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]


def load_model(key):
    path = os.path.join(DATA_DIR, f"svd_emergence_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(all_data, layer="final"):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    for key in MODELS:
        d = all_data[key]
        color = COLORS[key]

        if layer == "final":
            layer_key = str(d["n_layers"] - 1)
        else:
            layer_key = "0"

        vals = [d["checkpoints"][str(s)]["layers"][layer_key]["effective_rank"]
                for s in CHECKPOINTS]

        ax.plot(CHECKPOINTS, vals,
                color=color, marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.2, label=model_label(key), zorder=3)

    format_training_xaxis(ax)
    ax.set_ylabel("Effective rank")
    title = "Effective rank (final layer)" if layer == "final" else "Effective rank (layer 0)"
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data, layer="final")
    save(fig, "fig_effective_rank_final", out_dir=OUT_DIR)
    plt.close(fig)

    fig = make_figure(all_data, layer="0")
    save(fig, "fig_effective_rank_layer0", out_dir=OUT_DIR)
    plt.close(fig)
