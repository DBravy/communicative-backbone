"""
Figure: SVD Spectrum at Final Layer — Steps 0, 512, 2000
=========================================================

Overlays the top-50 singular values of the final-layer OV matrix
at training steps 0, 512, and 2000 for each model,
showing how spectral structure emerges during early training.
Generates one figure per model.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLORS, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_a")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_spectrum_final_layer")

MODELS = ["70m", "160m", "410m", "1b", "1.4b"]

STEPS = ["0", "512", "2000"]
STEP_STYLES = {
    "0":    {"color": "#999999", "label": "Step 0 (init)", "ls": "-",  "marker": "o"},
    "512":  {"color": "#bbbbbb", "label": "Step 512",      "ls": "--", "marker": "s"},
    "2000": {"color": None,      "label": "Step 2000",     "ls": "-",  "marker": "o"},
}


def load_model(key):
    path = os.path.join(DATA_DIR, f"svd_emergence_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data, key):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    color = COLORS[key]
    # final layer is the last key in the layers dict
    final_layer = str(data["n_layers"] - 1)

    for step in STEPS:
        svs = np.array(
            data["checkpoints"][step]["layers"][final_layer]["singular_values_top50"]
        )
        idx = np.arange(len(svs))

        style = STEP_STYLES[step]
        c = style["color"] or color

        ax.plot(idx, svs,
                color=c, linestyle=style["ls"],
                marker=style["marker"], markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.0, markersize=3,
                label=style["label"], zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(model_label(key), fontsize=9)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

    return fig


if __name__ == "__main__":
    for key in MODELS:
        data = load_model(key)
        print(f"{key}: {data['n_layers']} layers, final layer = {data['n_layers'] - 1}")
        fig = make_figure(data, key)
        save(fig, f"fig_svd_spectrum_final_layer_{key}", out_dir=OUT_DIR)
        plt.close(fig)
