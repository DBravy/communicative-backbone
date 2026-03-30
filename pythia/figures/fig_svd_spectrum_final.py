"""
Figure: SVD Spectrum at Final Checkpoint (Layer 0 & Final Layer)
================================================================

Top-50 singular values of the composed MLP product at step 143000,
comparing all three models. One figure each for layer 0, middle layer,
and final layer.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLORS, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_a")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_spectrum_final")
MODELS = ["410m", "1b", "1.4b"]
FINAL_STEP = 143000


def load_model(key):
    path = os.path.join(DATA_DIR, f"svd_emergence_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(all_data, layer="final"):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    for key in MODELS:
        d = all_data[key]
        color = COLORS[key]
        n_layers = d["n_layers"]

        if layer == "final":
            layer_key = str(n_layers - 1)
            layer_label = "final layer"
        elif layer == "mid":
            mid = n_layers // 2
            layer_key = str(mid)
            layer_label = f"middle layer"
        else:
            layer_key = str(layer)
            layer_label = f"layer {layer}"

        svs = np.array(d["checkpoints"][str(FINAL_STEP)]["layers"][layer_key]["singular_values_top50"])
        idx = np.arange(1, len(svs) + 1)

        ax.plot(idx, svs,
                color=color, marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, markersize=3,
                label=model_label(key), zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Composed MLP SVD, {layer_label} (step {FINAL_STEP:,})", fontsize=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

    return fig


if __name__ == "__main__":
    all_data = {key: load_model(key) for key in MODELS}
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_figure(all_data, layer="final")
    save(fig, "fig_svd_spectrum_final", out_dir=OUT_DIR)
    plt.close(fig)

    fig = make_figure(all_data, layer="mid")
    save(fig, "fig_svd_spectrum_final_mid", out_dir=OUT_DIR)
    plt.close(fig)

    fig = make_figure(all_data, layer=0)
    save(fig, "fig_svd_spectrum_final_layer0", out_dir=OUT_DIR)
    plt.close(fig)
