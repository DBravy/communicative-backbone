"""
Figure: Spectral Differentiation During Training
=================================================

Normalized spectral entropy (y) vs training step (x) for all Pythia model sizes.
Outputs PDF (vector) and PNG (300 dpi raster).
"""

import json
import os
import sys

import numpy as np

# Shared publication style (sets rcParams, palette, helpers)
sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, COLORS, MARKERS, LINESTYLES, MODEL_ORDER,
    SINGLE_COL, SINGLE_H, format_training_xaxis, model_label, save,
)
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_c")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectral_entropy")


def load_results():
    data = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("dct_training_") and fname.endswith(".json"):
            with open(os.path.join(DATA_DIR, fname)) as f:
                d = json.load(f)
            data[d["model"]] = d
    return data


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(data: dict, models=None):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    for key in (models or MODEL_ORDER):
        if key not in data:
            continue
        d = data[key]
        steps = sorted(int(s) for s in d["checkpoints"].keys())
        ent_mean = np.array([d["checkpoints"][str(s)]["metrics"]["normalized_entropy_mean"]
                             for s in steps])
        ent_std = np.array([d["checkpoints"][str(s)]["metrics"]["normalized_entropy_std"]
                            for s in steps])

        ax.plot(steps, ent_mean,
                color=COLORS[key], marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=COLORS[key],
                markeredgewidth=1.2, label=model_label(key), zorder=3)
        ax.fill_between(steps, ent_mean - ent_std, ent_mean + ent_std,
                        color=COLORS[key], alpha=0.12, linewidth=0)

    format_training_xaxis(ax)
    ax.set_title("Spectral entropy", fontsize=9)
    ax.set_ylabel("Normalized spectral entropy")
    ax.set_ylim(0.2, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # Reference: entropy = 1.0 is a perfectly uniform spectrum
    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=0.7, zorder=1)

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_results()
    if not data:
        print(f"No JSON files found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded models: {', '.join(sorted(data.keys()))}")

    # All models
    fig = make_figure(data)
    save(fig, "fig_spectral_entropy", out_dir=OUT_DIR)
    plt.close(fig)

    # Large models only
    large = ["410m", "1b", "1.4b"]
    fig = make_figure(data, models=large)
    save(fig, "fig_spectral_entropy_large", out_dir=OUT_DIR)
    plt.close(fig)
