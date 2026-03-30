"""
Figure: SVD Spectrum at All Training Stages (OLMo)
===================================================

Top-50 singular values of the composed MLP product at
every checkpoint. One figure per layer position
(layer 0, middle layer, final layer).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLORS, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_spectrum")


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


def make_figure(data, layer="final"):
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

    color = COLORS["1b"]
    steps = sorted(data["checkpoints"].keys(), key=int)

    # Color gradient from light grey (early) to model color (final)
    n = len(steps)
    from matplotlib.colors import to_rgb
    light = np.array([0.8, 0.8, 0.8])
    dark = np.array(to_rgb(color))

    for i, step_key in enumerate(steps):
        step = int(step_key)
        svs = np.array(data["checkpoints"][step_key]["layers"][layer_key]["singular_values_top50"])
        idx = np.arange(1, len(svs) + 1)

        frac = i / max(n - 1, 1)
        c = light + frac * (dark - light)

        ax.plot(idx, svs,
                color=c, marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.0, markersize=3,
                label=f"Step {_step_label(step)}", zorder=3)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Composed MLP SVD, {layer_label} ({model_label('1b')})", fontsize=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=5, ncol=2)

    return fig


if __name__ == "__main__":
    data = load_model()
    print(f"OLMo-1B: {data['n_layers']} layers")

    for layer_spec, suffix in [("final", ""), ("mid", "_mid"), (0, "_layer0")]:
        fig = make_figure(data, layer=layer_spec)
        save(fig, f"fig_svd_spectrum_olmo_1b{suffix}", out_dir=OUT_DIR)
        plt.close(fig)
