"""
Figure: SVD Spectrum at Layer 0 — GPT-2 and TinyLlama
======================================================

Top-50 singular values of the composed MLP product at layer 0.
One separate figure per model, matching the Pythia figure style.
"""

import json
import os
import sys

import numpy as np

# Reuse the publication style from pythia_chain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pythia_chain", "figures"))
from style import plt, SINGLE_COL, SINGLE_H, save

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = DATA_DIR

MODELS = [
    {
        "key": "gpt2",
        "file": "svd_bulktail_gpt2.json",
        "color": "#4477AA",   # blue
        "marker": "o",
    },
    {
        "key": "tinyllama",
        "file": "svd_bulktail_tinyllama.json",
        "color": "#EE6677",   # rose
        "marker": "s",
    },
]


def load_model(cfg):
    path = os.path.join(DATA_DIR, cfg["file"])
    with open(path) as f:
        return json.load(f)


def model_label(data):
    name = data["model"]
    if name == "gpt2":
        return "GPT-2"
    elif name == "tinyllama":
        return "TinyLlama-1.1B"
    return name.upper()


def make_figure(cfg, layer=0, layer_label=None):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    data = load_model(cfg)
    n_layers = data["n_layers"]
    layer_key = str(layer)
    svs = np.array(data["weight_level"]["layers"][layer_key]["singular_values_top50"])
    idx = np.arange(1, len(svs) + 1)

    color = cfg["color"]

    ax.plot(idx, svs,
            color=color, marker=cfg["marker"], linestyle="-",
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.0, markersize=3,
            zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")

    if layer_label is None:
        layer_label = f"layer {layer}"
    ax.set_title(f"Composed MLP SVD, {layer_label} ({model_label(data)})", fontsize=8)

    return fig


def layer_suffix(layer_type):
    if layer_type == "final":
        return "final"
    elif layer_type == "mid":
        return "mid"
    else:
        return f"layer{layer_type}"


if __name__ == "__main__":
    for cfg in MODELS:
        data = load_model(cfg)
        n_layers = data["n_layers"]

        mid = n_layers // 2
        layers = [
            (0, "layer0", "layer 0"),
            (mid, "mid", f"layer {mid}"),
            (n_layers - 1, "final", "final layer"),
        ]

        for layer, suffix, label in layers:
            fig = make_figure(cfg, layer=layer, layer_label=label)
            save(fig, f"fig_svd_spectrum_{cfg['key']}_{suffix}", out_dir=OUT_DIR)
            plt.close(fig)
