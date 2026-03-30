"""
Figure: DCT Energy Spectrum — Initialization vs Trained
========================================================

Overlays the mean DCT energy spectrum at step 0 (init) and step 143000
(fully trained) for a given model. Generates one figure per model.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import plt, COLORS, SINGLE_COL, SINGLE_H, model_label, save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_c")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum_init_vs_trained")

MODELS = ["410m", "1b", "1.4b"]


def load_model(key):
    path = os.path.join(DATA_DIR, f"dct_training_{key}.json")
    with open(path) as f:
        return json.load(f)


def make_figure(data, key):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    color = COLORS[key]
    init_spectrum = np.array(data["checkpoints"]["0"]["mean_spectrum"])
    mid_spectrum = np.array(data["checkpoints"]["512"]["mean_spectrum"])
    trained_spectrum = np.array(data["checkpoints"]["143000"]["mean_spectrum"])
    freqs = np.arange(len(init_spectrum))

    mid_color = "#BBBBBB"
    ax.plot(freqs, init_spectrum,
            color="#999999", marker="o", linestyle="-",
            markerfacecolor="white", markeredgecolor="#999999",
            markeredgewidth=1.2, label="Step 0 (init)", zorder=3)
    ax.plot(freqs, mid_spectrum,
            color=mid_color, marker="s", linestyle="-",
            markerfacecolor="white", markeredgecolor=mid_color,
            markeredgewidth=1.2, label="Step 512", zorder=3)
    ax.plot(freqs, trained_spectrum,
            color=color, marker="o", linestyle="-",
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.2, label="Step 143K (trained)", zorder=3)

    ax.set_xlabel("DCT frequency index")
    ax.set_ylabel("Normalized energy")
    ax.set_title(model_label(key), fontsize=9)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

    # Show ticks every 5 for models with many layers, every 1 otherwise
    n = len(freqs)
    if n > 20:
        ax.set_xticks(freqs[::5])
    else:
        ax.set_xticks(freqs)

    return fig


if __name__ == "__main__":
    for key in MODELS:
        data = load_model(key)
        print(f"{key}: {data['n_dct_frequencies']} frequencies")
        fig = make_figure(data, key)
        save(fig, f"fig_spectrum_init_vs_trained_{key}", out_dir=OUT_DIR)
        plt.close(fig)
