"""
Experiment B (OLMo): Cross-Layer Subspace Coherence (Bulk vs Tail)
===================================================================

OLMo port of experiment_b_crosslayer_overlap.py (originally for Pythia).

Tests whether training selectively aligns the *structural* (top-k) singular
subspaces across layers while leaving the *flexible* (tail-k) subspaces
incoherent.

Architecture note:
    OLMo uses SwiGLU (gate_proj, up_proj, down_proj) while Pythia uses a
    simple two-matrix MLP (dense_h_to_4h, dense_4h_to_h). The composed
    product W_down @ W_up is used as the primary analysis target, matching
    the Pythia experiments. See experiment_a header for discussion.

Models: OLMo-1B, OLMo-7B (v1, with intermediate checkpoints)
Checkpoints: sampled across training (OLMo releases every 1000 steps)

Checkpoint loading:
    Requires the original (non-HF) model IDs with ai2-olmo:
        pip install ai2-olmo
    Models: allenai/OLMo-1B, allenai/OLMo-7B
    Revisions: "step{N}-tokens{T}B"

Usage:
    python experiment_b_crosslayer_overlap_olmo.py [--models 1b 7b]
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

CHECKPOINTS_1B = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

MODEL_CONFIGS = {
    "1b": {
        "name": "allenai/OLMo-2-0425-1B",
        "d_model": 2048, "d_ff": 8192, "n_layers": 16,
        "checkpoints": CHECKPOINTS_1B,
    },
}

K_VALUES = [5, 10, 20, 50]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def step_to_revision(step: int) -> str:
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


# ---------------------------------------------------------------------------
# Core computation (identical to Pythia version)
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between two subspaces."""
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return cosines


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> dict:
    """Compute summary statistics of subspace overlap."""
    cosines = principal_angles_cosines(U1, U2)
    angles = np.arccos(np.clip(cosines, -1, 1))

    return {
        "mean_cosine": float(np.mean(cosines)),
        "max_cosine": float(np.max(cosines)),
        "min_cosine": float(np.min(cosines)),
        "median_cosine": float(np.median(cosines)),
        "grassmann_distance": float(np.sqrt(np.sum(angles ** 2))),
        "overlap_frac_gt_0.5": float(np.mean(cosines > 0.5)),
        "n_angles": len(cosines),
    }


def random_subspace_baseline(d: int, k: int, n_trials: int = 100) -> dict:
    """Expected overlap between two random k-dimensional subspaces of R^d."""
    cosines_all = []
    for _ in range(n_trials):
        U1 = np.linalg.qr(np.random.randn(d, k))[0]
        U2 = np.linalg.qr(np.random.randn(d, k))[0]
        cosines = principal_angles_cosines(U1, U2)
        cosines_all.append(np.mean(cosines))
    return {
        "mean_cosine_mean": float(np.mean(cosines_all)),
        "mean_cosine_std": float(np.std(cosines_all)),
    }


# ---------------------------------------------------------------------------
# Model loading and SVD extraction
# ---------------------------------------------------------------------------

def get_layer_svd(model, layer_idx: int) -> tuple:
    """Get full SVD of composed MLP product W_down @ W_up at given layer."""
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()
    W_down = mlp.down_proj.weight.detach().float()

    composed = (W_down @ W_up).cpu().numpy()
    U, S, Vt = np.linalg.svd(composed, full_matrices=True)
    return U, S, Vt


def load_model_at_checkpoint(model_name: str, step: int):
    """Load an OLMo 2 model at a specific training checkpoint."""
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else model_name
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision}...")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(model_key: str, output_dir: str):
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["name"]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    checkpoints = cfg["checkpoints"]

    # Load existing results if available (to skip already-computed checkpoints)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"crosslayer_overlap_olmo_{model_key}.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_path} "
              f"({len(results['checkpoints'])} checkpoints already computed)")
        baselines = {int(k): v for k, v in results["random_baselines"].items()}
    else:
        print("Computing random subspace baselines...")
        baselines = {}
        for k in K_VALUES:
            baselines[k] = random_subspace_baseline(d_model, k, n_trials=200)
            print(f"  k={k}: random mean_cosine = "
                  f"{baselines[k]['mean_cosine_mean']:.4f} +/- "
                  f"{baselines[k]['mean_cosine_std']:.4f}")

        results = {
            "model": model_key,
            "model_name": model_name,
            "model_family": "olmo",
            "d_model": d_model,
            "n_layers": n_layers,
            "k_values": K_VALUES,
            "random_baselines": {str(k): v for k, v in baselines.items()},
            "checkpoints": {},
        }

    for step in checkpoints:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model = load_model_at_checkpoint(model_name, step)

        # Extract SVDs for all layers
        layer_svds = []
        for li in range(n_layers):
            U, S, Vt = get_layer_svd(model, li)
            layer_svds.append({"U": U, "S": S})

        step_results = {"adjacent_pairs": {}, "non_adjacent": {}}

        # --- Adjacent layer comparisons ---
        for li in range(n_layers - 1):
            pair_key = f"{li}_{li+1}"
            pair_data = {}

            for k in K_VALUES:
                U1_top = layer_svds[li]["U"][:, :k]
                U2_top = layer_svds[li + 1]["U"][:, :k]

                U1_bot = layer_svds[li]["U"][:, -k:]
                U2_bot = layer_svds[li + 1]["U"][:, -k:]

                top_overlap = subspace_overlap(U1_top, U2_top)
                bot_overlap = subspace_overlap(U1_bot, U2_bot)

                pair_data[f"top{k}"] = top_overlap
                pair_data[f"bot{k}"] = bot_overlap

                print(f"    Layers {li}-{li+1}, k={k:2d}: "
                      f"top={top_overlap['mean_cosine']:.4f}  "
                      f"bot={bot_overlap['mean_cosine']:.4f}  "
                      f"(random={baselines[k]['mean_cosine_mean']:.4f})")

            step_results["adjacent_pairs"][pair_key] = pair_data

        # --- Non-adjacent comparisons ---
        global_pairs = [
            (0, n_layers - 1, "first_last"),
            (0, n_layers // 2, "first_mid"),
            (n_layers // 2, n_layers - 1, "mid_last"),
        ]

        for l1, l2, label in global_pairs:
            pair_data = {}
            for k in K_VALUES:
                U1_top = layer_svds[l1]["U"][:, :k]
                U2_top = layer_svds[l2]["U"][:, :k]
                U1_bot = layer_svds[l1]["U"][:, -k:]
                U2_bot = layer_svds[l2]["U"][:, -k:]

                pair_data[f"top{k}"] = subspace_overlap(U1_top, U2_top)
                pair_data[f"bot{k}"] = subspace_overlap(U1_bot, U2_bot)

            step_results["non_adjacent"][label] = pair_data
            print(f"    Global {label} (layers {l1}-{l2}), k=10: "
                  f"top={pair_data['top10']['mean_cosine']:.4f}  "
                  f"bot={pair_data['bot10']['mean_cosine']:.4f}")

        results["checkpoints"][str(step)] = step_results

        del model, layer_svds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    model_key = results["model"]
    n_layers = results["n_layers"]
    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
    baselines = results["random_baselines"]

    k = 10

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Layer Subspace Coherence: OLMo-{model_key.upper()} (k={k})",
                 fontsize=14)

    # --- Panel 1: Mean adjacent top-k overlap over training ---
    ax = axes[0, 0]
    top_means = []
    bot_means = []
    for step in checkpoints:
        pairs = results["checkpoints"][str(step)]["adjacent_pairs"]
        top_vals = [pairs[pk][f"top{k}"]["mean_cosine"] for pk in pairs]
        bot_vals = [pairs[pk][f"bot{k}"]["mean_cosine"] for pk in pairs]
        top_means.append(np.mean(top_vals))
        bot_means.append(np.mean(bot_vals))

    ax.plot(checkpoints, top_means, 'o-', color='tab:red', label=f'Top-{k} (structural)')
    ax.plot(checkpoints, bot_means, 'o-', color='tab:blue', label=f'Bottom-{k} (flexible)')
    bl = baselines[str(k)]
    ax.axhline(bl["mean_cosine_mean"], color='gray', linestyle='--',
               label='Random baseline', alpha=0.7)
    ax.fill_between(
        checkpoints,
        bl["mean_cosine_mean"] - 2 * bl["mean_cosine_std"],
        bl["mean_cosine_mean"] + 2 * bl["mean_cosine_std"],
        color='gray', alpha=0.15
    )
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine (adjacent layers)")
    ax.set_title("Adjacent-layer subspace overlap")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # --- Panel 2: Top vs bot gap over training (delta from baseline) ---
    ax = axes[0, 1]
    bl_val = bl["mean_cosine_mean"]
    top_delta = [t - bl_val for t in top_means]
    bot_delta = [b - bl_val for b in bot_means]
    ax.plot(checkpoints, top_delta, 'o-', color='tab:red', label=f'Top-{k} above random')
    ax.plot(checkpoints, bot_delta, 'o-', color='tab:blue', label=f'Bottom-{k} above random')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine - random baseline")
    ax.set_title("Overlap above chance")
    ax.legend(fontsize=9)

    # --- Panel 3: Per-layer-pair heatmap at final checkpoint ---
    ax = axes[1, 0]
    final_step = str(checkpoints[-1])
    pairs = results["checkpoints"][final_step]["adjacent_pairs"]
    pair_keys = sorted(pairs.keys(), key=lambda x: int(x.split("_")[0]))
    top_by_pair = [pairs[pk][f"top{k}"]["mean_cosine"] for pk in pair_keys]
    bot_by_pair = [pairs[pk][f"bot{k}"]["mean_cosine"] for pk in pair_keys]

    x_pos = range(len(pair_keys))
    width = 0.35
    ax.bar([p - width/2 for p in x_pos], top_by_pair, width,
           color='tab:red', alpha=0.7, label=f'Top-{k}')
    ax.bar([p + width/2 for p in x_pos], bot_by_pair, width,
           color='tab:blue', alpha=0.7, label=f'Bottom-{k}')
    ax.axhline(bl_val, color='gray', linestyle='--', alpha=0.5)
    # Only label every Nth pair for readability
    n_pairs = len(pair_keys)
    tick_stride = max(1, n_pairs // 15)
    ax.set_xticks([i for i in range(0, n_pairs, tick_stride)])
    ax.set_xticklabels([pair_keys[i].replace("_", "-")
                        for i in range(0, n_pairs, tick_stride)],
                       rotation=45, fontsize=7)
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Mean cosine")
    ax.set_title(f"Per-pair overlap (step {checkpoints[-1]})")
    ax.legend(fontsize=9)

    # --- Panel 4: Multiple k values at final checkpoint ---
    ax = axes[1, 1]
    k_labels = []
    top_vals_k = []
    bot_vals_k = []
    bl_vals_k = []
    for kk in K_VALUES:
        pairs = results["checkpoints"][final_step]["adjacent_pairs"]
        top_v = np.mean([pairs[pk][f"top{kk}"]["mean_cosine"] for pk in pairs])
        bot_v = np.mean([pairs[pk][f"bot{kk}"]["mean_cosine"] for pk in pairs])
        k_labels.append(str(kk))
        top_vals_k.append(top_v)
        bot_vals_k.append(bot_v)
        bl_vals_k.append(baselines[str(kk)]["mean_cosine_mean"])

    x_pos = range(len(k_labels))
    ax.bar([p - 0.25 for p in x_pos], top_vals_k, 0.25,
           color='tab:red', alpha=0.7, label='Top-k')
    ax.bar([p for p in x_pos], bot_vals_k, 0.25,
           color='tab:blue', alpha=0.7, label='Bottom-k')
    ax.bar([p + 0.25 for p in x_pos], bl_vals_k, 0.25,
           color='gray', alpha=0.4, label='Random')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(k_labels)
    ax.set_xlabel("k (subspace dimension)")
    ax.set_ylabel("Mean cosine (averaged over pairs)")
    ax.set_title(f"Overlap by subspace size (step {checkpoints[-1]})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"crosslayer_overlap_olmo_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment B (OLMo): Cross-layer subspace coherence")
    parser.add_argument("--models", nargs="+", default=["1b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output_dir", default="results/experiment_b_olmo")
    args = parser.parse_args()

    for model_key in args.models:
        results = run_experiment(model_key, args.output_dir)
        plot_results(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
