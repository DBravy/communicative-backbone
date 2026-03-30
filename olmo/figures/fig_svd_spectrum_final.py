"""
Figure: SVD Spectrum at Final Checkpoint (OLMo)
================================================

Top-50 singular values of the composed MLP product at the final
training step (1,000,000). One figure each for layer 0, middle
layer, and final layer.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLORS, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_spectrum_final")
FINAL_STEP = 1000000


def load_model():
    path = os.path.join(DATA_DIR, "svd_emergence_olmo_1b.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data, layer="final"):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    color = COLORS["1b"]
    n_layers = data["n_layers"]

    if layer == "final":
        layer_key = str(n_layers - 1)
        layer_label = "final layer"
    elif layer == "mid":
        mid = n_layers // 2
        layer_key = str(mid)
        layer_label = "middle layer"
    else:
        layer_key = str(layer)
        layer_label = f"layer {layer}"

    svs = np.array(data["checkpoints"][str(FINAL_STEP)]["layers"][layer_key]["singular_values_top50"])
    idx = np.arange(1, len(svs) + 1)

    ax.plot(idx, svs,
            color=color, marker="o", linestyle="-",
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.0, markersize=3,
            label=model_label("1b"), zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Composed MLP SVD, {layer_label} (step {FINAL_STEP:,})", fontsize=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

    return fig


if __name__ == "__main__":
    data = load_model()
    print(f"OLMo-1B: {data['n_layers']} layers")

    for layer_spec, suffix in [("final", ""), ("mid", "_mid"), (0, "_layer0")]:
        fig = make_figure(data, layer=layer_spec)
        save(fig, f"fig_svd_spectrum_final_olmo{suffix}", out_dir=OUT_DIR)
        plt.close(fig)
