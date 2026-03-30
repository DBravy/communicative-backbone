"""
Figure: Cross-Layer Subspace Coherence During Training
=======================================================

Mean top-10 subspace overlap between adjacent layers (y) vs training step (x)
for the three largest Pythia models.
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
import matplotlib.ticker as ticker

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_b")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crosslayer_coherence")
MODELS = ["410m", "1b", "1.4b"]
K = 10


def load_model(key):
    path = os.path.join(DATA_DIR, f"crosslayer_overlap_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(all_data):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    checkpoints = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]

    for key in MODELS:
        d = all_data[key]
        top_means = []
        for step in checkpoints:
            pairs = d["checkpoints"][str(step)]["adjacent_pairs"]
            vals = [pairs[pk][f"top{K}"]["mean_cosine"] for pk in pairs]
            top_means.append(np.mean(vals))

        ax.plot(checkpoints, top_means,
                color=COLORS[key], marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=COLORS[key],
                markeredgewidth=1.2, label=model_label(key), zorder=3)

    # Random baseline (use first model's baseline — they're all similar)
    bl = all_data[MODELS[0]]["random_baselines"][str(K)]
    bl_mean = bl["mean_cosine_mean"]
    bl_std = bl["mean_cosine_std"]
    ax.axhline(bl_mean, color="#999999", linestyle=":", linewidth=0.7, zorder=1)
    ax.fill_between(
        [checkpoints[0], checkpoints[-1]],
        bl_mean - 2 * bl_std, bl_mean + 2 * bl_std,
        color="#999999", alpha=0.1, linewidth=0,
    )
    ax.text(160000, bl_mean, "random", fontsize=6, color="#999999",
            ha="right", va="bottom")

    format_training_xaxis(ax)
    ax.set_title("Cross-layer coherence", fontsize=9)
    ax.set_ylabel(f"Mean cosine (top-{K}, adjacent layers)")

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")
    fig = make_figure(all_data)
    save(fig, "fig_crosslayer_coherence", out_dir=OUT_DIR)
    plt.close(fig)
