"""
Figure: SVD Spectrum per Layer — TinyLlama
===========================================

Top-50 singular values of the composed MLP product for every layer,
plus an effective-rank-by-layer bar chart.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLOR, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_spectrum")


def load_model():
    path = os.path.join(DATA_DIR, "svd_bulktail_tinyllama.json")
    with open(path) as f:
        return json.load(f)


def make_spectrum_figure(data, layer=0):
    """Single-layer SVD spectrum."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    layer_key = str(layer)
    svs = np.array(data["weight_level"]["layers"][layer_key]["singular_values_top50"])
    idx = np.arange(1, len(svs) + 1)

    ax.plot(idx, svs,
            color=COLOR, marker="o", linestyle="-",
            markerfacecolor="white", markeredgecolor=COLOR,
            markeredgewidth=1.0, markersize=3, zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Composed MLP SVD, layer {layer} ({model_label()})", fontsize=8)

    return fig


def make_effective_rank_bar(data):
    """Bar chart of effective rank across all layers."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    n_layers = data["n_layers"]
    layers = list(range(n_layers))
    ranks = [data["weight_level"]["layers"][str(li)]["effective_rank"]
             for li in layers]

    ax.bar(layers, ranks, color=COLOR, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective rank")
    ax.set_title(f"Effective rank by layer ({model_label()})", fontsize=9)
    ax.set_xticks(np.linspace(0, n_layers - 1, min(8, n_layers), dtype=int))

    return fig


if __name__ == "__main__":
    data = load_model()
    n_layers = data["n_layers"]
    print(f"TinyLlama: {n_layers} layers")

    # Per-layer SVD spectrum
    for li in range(n_layers):
        fig = make_spectrum_figure(data, layer=li)
        save(fig, f"fig_svd_spectrum_tinyllama_layer{li}", out_dir=OUT_DIR)
        plt.close(fig)

    # Effective rank bar chart
    fig = make_effective_rank_bar(data)
    save(fig, "fig_effective_rank_bar_tinyllama", out_dir=OUT_DIR)
    plt.close(fig)
