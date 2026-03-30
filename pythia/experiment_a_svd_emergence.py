"""
Experiment A: SVD Bulk-Tail Emergence During Training
=====================================================

Tracks the emergence of spectral gap structure in composed MLP weight matrices
(W_down @ W_up) across Pythia training checkpoints.

Models: Pythia-70M, Pythia-410M
Checkpoints: 0, 128, 512, 2000, 8000, 32000, 64000, 143000

For each layer at each checkpoint, computes:
  1. Full SVD of composed product W_down @ W_up
  2. Effective rank (exp of entropy of normalized singular values)
  3. Top-k energy ratio (fraction of total energy in top-k singular values)
  4. Bulk-tail gap: ratio of k-th singular value to (k+1)-th, for several k
  5. Marchenko-Pastur deviation: KL divergence from MP distribution

Outputs a single JSON results file and a set of summary plots.

This complements existing alignment tracking data (pythia_alignment_tracking.py)
by showing that singular value *separation* co-evolves with singular *vector
alignment* during training.

Usage:
    python experiment_a_svd_emergence.py [--models 70m 410m] [--device cuda]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]

MODEL_CONFIGS = {
    "70m":  {"name": "EleutherAI/pythia-70m",  "d_model": 512,  "d_ff": 2048, "n_layers": 6},
    "160m": {"name": "EleutherAI/pythia-160m", "d_model": 768,  "d_ff": 3072, "n_layers": 12},
    "410m": {"name": "EleutherAI/pythia-410m", "d_model": 1024, "d_ff": 4096, "n_layers": 24},
    "1b":   {"name": "EleutherAI/pythia-1b",   "d_model": 2048, "d_ff": 8192, "n_layers": 16},
    "1.4b": {"name": "EleutherAI/pythia-1.4b", "d_model": 2048, "d_ff": 8192, "n_layers": 24},
}

# How many singular values to treat as "top-k" for energy ratio and gap metrics.
# We use several values of k to see where the natural break falls.
TOP_K_VALUES = [5, 10, 20, 50]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def effective_rank(singular_values: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized squared singular values).
    
    Roy & Bhattacharya (2007). Measures the effective dimensionality of the
    singular value distribution. A uniform distribution gives effective rank = N;
    a single nonzero value gives effective rank = 1.
    """
    sv_sq = singular_values ** 2
    total = sv_sq.sum()
    if total < 1e-12:
        return 0.0
    p = sv_sq / total
    # Clip to avoid log(0)
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def top_k_energy_ratio(singular_values: np.ndarray, k: int) -> float:
    """Fraction of total squared singular value energy in top-k components."""
    sv_sq = singular_values ** 2
    total = sv_sq.sum()
    if total < 1e-12:
        return 0.0
    return float(sv_sq[:k].sum() / total)


def bulk_tail_gap(singular_values: np.ndarray, k: int) -> float:
    """Ratio of k-th to (k+1)-th singular value.
    
    A large ratio indicates a clear break between tail (top-k) and bulk.
    """
    if k >= len(singular_values) - 1:
        return float('nan')
    sk = singular_values[k - 1]      # k-th largest (0-indexed: k-1)
    sk_plus_1 = singular_values[k]   # (k+1)-th largest
    if sk_plus_1 < 1e-12:
        return float('inf')
    return float(sk / sk_plus_1)


def marchenko_pastur_kl(singular_values: np.ndarray, aspect_ratio: float,
                        n_bins: int = 50) -> float:
    """KL divergence of the empirical squared-SV distribution from
    the Marchenko-Pastur distribution with the given aspect ratio.
    
    Higher values = more deviation from random initialization.
    """
    gamma = aspect_ratio  # rows/cols ratio
    sv_sq = singular_values ** 2
    # Normalize to unit variance per entry (as MP assumes)
    sigma_sq = sv_sq.mean()
    if sigma_sq < 1e-12:
        return 0.0
    sv_sq_normed = sv_sq / sigma_sq

    # MP bounds
    lambda_plus = (1 + np.sqrt(gamma)) ** 2
    lambda_minus = (1 - np.sqrt(gamma)) ** 2

    # Empirical histogram
    # Use range slightly wider than MP support to capture tail
    bin_max = max(lambda_plus * 2, sv_sq_normed.max() * 1.1)
    bins = np.linspace(0, bin_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    hist_emp, _ = np.histogram(sv_sq_normed, bins=bins, density=True)

    # MP density
    mp_density = np.zeros_like(bin_centers)
    for i, x in enumerate(bin_centers):
        if lambda_minus < x < lambda_plus:
            mp_density[i] = (1 / (2 * np.pi * gamma * x)) * \
                np.sqrt((lambda_plus - x) * (x - lambda_minus))

    # Normalize MP to integrate to 1 over bins
    mp_total = mp_density.sum() * bin_width
    if mp_total > 1e-12:
        mp_density /= mp_total

    # KL(empirical || MP), with smoothing to avoid log(0)
    eps = 1e-10
    p = hist_emp * bin_width + eps
    q = mp_density * bin_width + eps
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    return kl


# ---------------------------------------------------------------------------
# Model loading and weight extraction
# ---------------------------------------------------------------------------

def get_composed_mlp_weights(model, layer_idx: int) -> np.ndarray:
    """Extract W_down @ W_up for the given layer.
    
    Pythia (GPT-NeoX) MLP structure:
        dense_h_to_4h: (d_ff, d_model) -- "up" projection
        dense_4h_to_h: (d_model, d_ff) -- "down" projection
    
    Composed product: W_down @ W_up has shape (d_model, d_model).
    """
    layer = model.gpt_neox.layers[layer_idx].mlp
    W_up = layer.dense_h_to_4h.weight.detach().float()    # (d_ff, d_model)
    W_down = layer.dense_4h_to_h.weight.detach().float()   # (d_model, d_ff)
    composed = W_down @ W_up  # (d_model, d_model)
    return composed.cpu().numpy()


def load_model_at_checkpoint(model_name: str, step: int, device: str):
    """Load a Pythia model at a specific training checkpoint."""
    from transformers import AutoModelForCausalLM
    
    revision = f"step{step}"
    print(f"  Loading {model_name} at {revision}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        # Use low_cpu_mem_usage to avoid doubling memory during load
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    # We only need weights, not GPU compute, so keep on CPU
    # (SVD will be done in numpy)
    return model


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model_key: str, device: str, output_dir: str):
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["name"]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    d_ff = cfg["d_ff"]
    aspect_ratio = d_ff / d_model  # for MP comparison

    results = {
        "model": model_key,
        "model_name": model_name,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "checkpoints": {},
    }

    for step in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model = load_model_at_checkpoint(model_name, step, device)
        step_results = {"layers": {}}

        for layer_idx in range(n_layers):
            composed = get_composed_mlp_weights(model, layer_idx)
            
            # Full SVD (we need all singular values for effective rank, MP, etc.)
            sv = np.linalg.svd(composed, compute_uv=False)
            sv = np.sort(sv)[::-1]  # descending order

            layer_data = {
                "singular_values_top50": sv[:50].tolist(),
                "singular_values_tail20": sv[-20:].tolist(),
                "n_singular_values": len(sv),
                "effective_rank": effective_rank(sv),
                "mp_kl_divergence": marchenko_pastur_kl(sv, aspect_ratio),
                "max_sv": float(sv[0]),
                "median_sv": float(np.median(sv)),
                "max_over_median": float(sv[0] / np.median(sv)) if np.median(sv) > 1e-12 else float('inf'),
            }

            # Top-k metrics at several cutoffs
            for k in TOP_K_VALUES:
                if k < len(sv):
                    layer_data[f"energy_ratio_top{k}"] = top_k_energy_ratio(sv, k)
                    layer_data[f"gap_at_{k}"] = bulk_tail_gap(sv, k)

            step_results["layers"][str(layer_idx)] = layer_data
            print(f"    Layer {layer_idx:2d}: eff_rank={layer_data['effective_rank']:.1f}, "
                  f"top10_energy={layer_data.get('energy_ratio_top10', 'N/A'):.3f}, "
                  f"MP_KL={layer_data['mp_kl_divergence']:.3f}, "
                  f"max/med={layer_data['max_over_median']:.2f}")

        results["checkpoints"][str(step)] = step_results

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"svd_emergence_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_dir: str):
    """Generate summary plots from experiment results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    model_key = results["model"]
    n_layers = results["n_layers"]
    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"SVD Bulk-Tail Emergence: Pythia-{model_key}", fontsize=14)

    # Select a few representative layers
    if n_layers <= 6:
        layer_indices = list(range(n_layers))
    else:
        # First, 1/4, 1/2, 3/4, last
        layer_indices = [0, n_layers // 4, n_layers // 2,
                         3 * n_layers // 4, n_layers - 1]

    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # Panel 1: Effective rank over training
    ax = axes[0, 0]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)]["effective_rank"]
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title("Composed effective rank")
    ax.legend(fontsize=8)

    # Panel 2: Top-10 energy ratio over training
    ax = axes[0, 1]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)].get("energy_ratio_top10", 0)
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Energy in top-10 SVs")
    ax.set_title("Top-10 energy concentration")
    ax.legend(fontsize=8)

    # Panel 3: MP KL divergence over training
    ax = axes[0, 2]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)]["mp_kl_divergence"]
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL(empirical || MP)")
    ax.set_title("Deviation from random (MP) spectrum")
    ax.legend(fontsize=8)

    # Panel 4: Max/median SV ratio over training
    ax = axes[1, 0]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)]["max_over_median"]
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("max(sv) / median(sv)")
    ax.set_title("Max-to-median SV ratio (spread)")
    ax.legend(fontsize=8)

    # Panel 5: Gap at k=10 over training
    ax = axes[1, 1]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)].get("gap_at_10", 1.0)
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("σ₁₀ / σ₁₁")
    ax.set_title("Bulk-tail gap at k=10")
    ax.legend(fontsize=8)

    # Panel 6: SV spectrum snapshots for final layer
    ax = axes[1, 2]
    final_layer = str(n_layers - 1)
    snapshot_steps = [checkpoints[0], checkpoints[len(checkpoints)//2], checkpoints[-1]]
    for j, s in enumerate(snapshot_steps):
        sv_top = results["checkpoints"][str(s)]["layers"][final_layer]["singular_values_top50"]
        ax.plot(range(len(sv_top)), sv_top, '-', alpha=0.8, label=f"Step {s}")
    ax.set_xlabel("SV index (sorted descending)")
    ax.set_ylabel("Singular value")
    ax.set_title(f"SV spectrum snapshots (layer {final_layer})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"svd_emergence_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment A: SVD bulk-tail emergence")
    parser.add_argument("--models", nargs="+", default=["70m", "410m"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results/experiment_a")
    args = parser.parse_args()

    for model_key in args.models:
        results = run_experiment(model_key, args.device, args.output_dir)
        plot_results(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
