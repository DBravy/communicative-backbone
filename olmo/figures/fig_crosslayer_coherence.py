"""
Figure: Cross-Layer Subspace Coherence During Training (OLMo)
==============================================================

Mean top-10 subspace overlap between adjacent layers (y) vs training step (x).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, COLORS, SINGLE_COL, SINGLE_H,
    model_label, save,
)
import matplotlib.ticker as ticker

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crosslayer_coherence")
K = 10


def load_model():
    path = os.path.join(DATA_DIR, "crosslayer_overlap_olmo_1b.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    steps = sorted(data["checkpoints"].keys(), key=int)
    step_ints = [int(s) for s in steps]

    top_means = []
    for s in steps:
        pairs = data["checkpoints"][s]["adjacent_pairs"]
        vals = [pairs[pk][f"top{K}"]["mean_cosine"] for pk in pairs]
        top_means.append(np.mean(vals))

    ax.plot(step_ints, top_means,
            color=COLORS["1b"], marker="o", linestyle="-",
            markerfacecolor="white", markeredgecolor=COLORS["1b"],
            markeredgewidth=1.2, label=model_label("1b"), zorder=3)

    # Random baseline
    bl = data["random_baselines"][str(K)]
    bl_mean = bl["mean_cosine_mean"]
    bl_std = bl["mean_cosine_std"]
    ax.axhline(bl_mean, color="#999999", linestyle=":", linewidth=0.7, zorder=1)
    ax.fill_between(
        [step_ints[0], step_ints[-1]],
        bl_mean - 2 * bl_std, bl_mean + 2 * bl_std,
        color="#999999", alpha=0.1, linewidth=0,
    )
    ax.text(step_ints[-1] * 1.1, bl_mean, "random", fontsize=6,
            color="#999999", ha="right", va="bottom")

    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Training step")
    ax.set_title("Cross-layer coherence (OLMo)", fontsize=9)
    ax.set_ylabel(f"Mean cosine (top-{K}, adjacent layers)")

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


if __name__ == "__main__":
    data = load_model()
    print("Loaded OLMo-1B")
    fig = make_figure(data)
    save(fig, "fig_crosslayer_coherence_olmo", out_dir=OUT_DIR)
    plt.close(fig)
