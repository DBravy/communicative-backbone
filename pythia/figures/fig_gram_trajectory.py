"""
Figure: Gram Matrix Trajectory During Training
================================================

Two separate figures for the three largest Pythia models:
  1. Gram effective rank vs training step
  2. Gram max/min eigenvalue ratio vs training step
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style import (
    plt, COLORS, SINGLE_COL, SINGLE_H,
    format_training_xaxis, model_label, save,
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_d",
    "gram_trajectory_results_full.json",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gram_trajectory")
MODELS = ["410m", "1b", "1.4b"]
CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def make_effective_rank(all_data):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    for key in MODELS:
        d = all_data[key]
        color = COLORS[key]
        vals = [d["checkpoints"][str(s)]["gram"]["gram_effective_rank"]
                for s in CHECKPOINTS]
        ax.plot(CHECKPOINTS, vals,
                color=color, marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.2, label=model_label(key), zorder=3)

    format_training_xaxis(ax)
    ax.set_ylabel("Effective rank")
    ax.set_title("Gram effective rank", fontsize=9)
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


def make_eigenvalue_ratio(all_data):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_H))

    for key in MODELS:
        d = all_data[key]
        color = COLORS[key]
        vals = [d["checkpoints"][str(s)]["gram"]["gram_max_min_ratio"]
                for s in CHECKPOINTS]
        ax.plot(CHECKPOINTS, vals,
                color=color, marker="o", linestyle="-",
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.2, label=model_label(key), zorder=3)

    format_training_xaxis(ax)
    ax.set_ylabel("Max / min eigenvalue ratio")
    ax.set_yscale("log")
    ax.set_title("Gram eigenvalue ratio", fontsize=9)
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none")
    return fig


if __name__ == "__main__":
    all_data = load_data()
    print(f"Loaded models: {', '.join(MODELS)}")

    fig = make_effective_rank(all_data)
    save(fig, "fig_gram_effective_rank", out_dir=OUT_DIR)
    plt.close(fig)

    fig = make_eigenvalue_ratio(all_data)
    save(fig, "fig_gram_eigenvalue_ratio", out_dir=OUT_DIR)
    plt.close(fig)
