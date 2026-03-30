"""
Figure: SVD Spectrum at Key Training Stages
=============================================

Top-50 singular values of the composed MLP product at
steps 512, 2000, and 143000. One figure per model, for
layer 0, middle layer, and final layer.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLORS, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_a")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_spectrum")
MODELS = ["410m", "1b", "1.4b"]
STEPS = [512, 2000, 143000]
STEP_COLORS = {"512": "#BBBBBB", "2000": "#999999"}  # final step uses model color
STEP_MARKERS = {"512": "s", "2000": "D"}


def load_model(key):
    path = os.path.join(DATA_DIR, f"svd_emergence_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data, key, layer="final"):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    n_layers = data["n_layers"]
    if layer == "final":
        layer_key = str(n_layers - 1)
        layer_label = "final layer"
    elif layer == "mid":
        mid = n_layers // 2
        layer_key = str(mid)
        layer_label = f"layer {mid}"
    else:
        layer_key = str(layer)
        layer_label = f"layer {layer}"

    color = COLORS[key]

    for step in STEPS:
        svs = np.array(data["checkpoints"][str(step)]["layers"][layer_key]["singular_values_top50"])
        idx = np.arange(1, len(svs) + 1)

        if step == STEPS[-1]:  # last step gets model color
            c = color
            marker = "o"
        else:
            c = STEP_COLORS[str(step)]
            marker = STEP_MARKERS[str(step)]

        ax.plot(idx, svs,
                color=c, marker=marker, linestyle="-",
                markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.0, markersize=3,
                label=f"Step {step:,}", zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Composed MLP SVD, {layer_label} ({model_label(key)})", fontsize=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

    return fig


if __name__ == "__main__":
    for key in MODELS:
        data = load_model(key)
        print(f"{key}: {data['n_layers']} layers")

        fig = make_figure(data, key, layer="final")
        save(fig, f"fig_svd_spectrum_{key}", out_dir=OUT_DIR)
        plt.close(fig)

        fig = make_figure(data, key, layer="mid")
        save(fig, f"fig_svd_spectrum_{key}_mid", out_dir=OUT_DIR)
        plt.close(fig)

        fig = make_figure(data, key, layer=0)
        save(fig, f"fig_svd_spectrum_{key}_layer0", out_dir=OUT_DIR)
        plt.close(fig)
