"""
Experiment A (OLMo): SVD Bulk-Tail Emergence During Training
=============================================================

OLMo port of experiment_a_svd_emergence.py (originally for Pythia).

Tracks the emergence of spectral gap structure in composed MLP weight matrices
across OLMo training checkpoints.

Models: OLMo-1B, OLMo-7B (v1, with intermediate checkpoints)
Checkpoints: sampled across training (every 1000 steps available)

Architecture note -- SwiGLU vs simple MLP:
    Pythia uses a two-matrix MLP:  output = W_down @ act(W_up @ x)
    OLMo  uses a SwiGLU MLP:      output = W_down @ (act(W_gate @ x) * W_up @ x)

    The composed product W_down @ W_up remains a meaningful object for spectral
    analysis: it captures the linear component of the MLP's residual-to-residual
    map. The gating branch (W_gate) modulates this multiplicatively but does not
    change the subspace geometry of W_down @ W_up itself. We optionally also
    compute W_down @ W_gate for comparison.

For each layer at each checkpoint, computes:
  1. Full SVD of composed product W_down @ W_up  (and optionally W_down @ W_gate)
  2. Effective rank (exp of entropy of normalized singular values)
  3. Top-k energy ratio (fraction of total energy in top-k singular values)
  4. Bulk-tail gap: ratio of k-th singular value to (k+1)-th, for several k
  5. Marchenko-Pastur deviation: KL divergence from MP distribution

Outputs a single JSON results file and a set of summary plots.

Checkpoint availability:
    OLMo v1 models release checkpoints every 1000 training steps.
    Revisions are named "step{N}-tokens{T}B" on HuggingFace.
    The original (non-HF) model IDs must be used for intermediate checkpoints:
        allenai/OLMo-1B   (NOT allenai/OLMo-1B-hf)
        allenai/OLMo-7B   (NOT allenai/OLMo-7B-hf)
    Requires: pip install ai2-olmo

    To list available revisions for a model:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs("allenai/OLMo-1B")
        print([b.name for b in refs.branches][:20])

Usage:
    python experiment_a_svd_emergence_olmo.py [--models 1b 7b] [--device cuda]
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OLMo 2 checkpoints are available every 1000 steps.
# Each step processes ~2.1M tokens (2048 * 1024).
#
# Steps 0-37000 are in the early-training repo.
# Later steps are in the main model repo.
#
# Revision format: "stage1-step{N}-tokens{T}B"

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

CHECKPOINTS_1B = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

MODEL_CONFIGS = {
    "1b": {
        "name": "allenai/OLMo-2-0425-1B",
        "d_model": 2048,
        "d_ff": 8192,         # intermediate_size
        "n_layers": 16,
        "checkpoints": CHECKPOINTS_1B,
    },
}

TOP_K_VALUES = [5, 10, 20, 50]


# ---------------------------------------------------------------------------
# Checkpoint revision helpers
# ---------------------------------------------------------------------------

def step_to_revision(step: int) -> str:
    """Convert a step number to an OLMo 2 HuggingFace revision string."""
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


def list_available_revisions(model_name: str, max_show: int = 20):
    """List available checkpoint revisions for an OLMo model on HuggingFace."""
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(model_name)
        branches = [b.name for b in refs.branches if b.name.startswith("step")]
        print(f"  Found {len(branches)} checkpoint branches for {model_name}")
        if branches:
            print(f"  First few: {branches[:max_show]}")
        return branches
    except Exception as e:
        print(f"  Could not list revisions: {e}")
        return []


# ---------------------------------------------------------------------------
# Metrics (identical to Pythia version)
# ---------------------------------------------------------------------------

def effective_rank(singular_values: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized squared singular values)."""
    sv_sq = singular_values ** 2
    total = sv_sq.sum()
    if total < 1e-12:
        return 0.0
    p = sv_sq / total
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
    """Ratio of k-th to (k+1)-th singular value."""
    if k >= len(singular_values) - 1:
        return float('nan')
    sk = singular_values[k - 1]
    sk_plus_1 = singular_values[k]
    if sk_plus_1 < 1e-12:
        return float('inf')
    return float(sk / sk_plus_1)


def marchenko_pastur_kl(singular_values: np.ndarray, aspect_ratio: float,
                        n_bins: int = 50) -> float:
    """KL divergence of the empirical squared-SV distribution from
    the Marchenko-Pastur distribution with the given aspect ratio."""
    gamma = aspect_ratio
    sv_sq = singular_values ** 2
    sigma_sq = sv_sq.mean()
    if sigma_sq < 1e-12:
        return 0.0
    sv_sq_normed = sv_sq / sigma_sq

    lambda_plus = (1 + np.sqrt(gamma)) ** 2
    lambda_minus = (1 - np.sqrt(gamma)) ** 2

    bin_max = max(lambda_plus * 2, sv_sq_normed.max() * 1.1)
    bins = np.linspace(0, bin_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    hist_emp, _ = np.histogram(sv_sq_normed, bins=bins, density=True)

    mp_density = np.zeros_like(bin_centers)
    for i, x in enumerate(bin_centers):
        if lambda_minus < x < lambda_plus:
            mp_density[i] = (1 / (2 * np.pi * gamma * x)) * \
                np.sqrt((lambda_plus - x) * (x - lambda_minus))

    mp_total = mp_density.sum() * bin_width
    if mp_total > 1e-12:
        mp_density /= mp_total

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

def get_composed_mlp_weights(model, layer_idx: int,
                              composition: str = "down_up") -> np.ndarray:
    """Extract the composed MLP product for the given layer.

    OLMo 2 SwiGLU MLP (standard HF format):
        gate_proj: (d_ff, d_model)  -- gating branch
        up_proj:   (d_ff, d_model)  -- value branch
        down_proj: (d_model, d_ff)  -- output projection

    Args:
        composition: which product to compute
            "down_up"   -> W_down @ W_up   (d_model, d_model) -- default, matches Pythia
            "down_gate" -> W_down @ W_gate (d_model, d_model) -- gating path

    Returns:
        composed: (d_model, d_model) numpy array
    """
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()       # (d_ff, d_model)
    W_gate = mlp.gate_proj.weight.detach().float()    # (d_ff, d_model)
    W_down = mlp.down_proj.weight.detach().float()    # (d_model, d_ff)

    if composition == "down_up":
        composed = W_down @ W_up
    elif composition == "down_gate":
        composed = W_down @ W_gate
    else:
        raise ValueError(f"Unknown composition: {composition}")

    return composed.cpu().numpy()


def load_model_at_checkpoint(model_name: str, step: int, device: str):
    """Load an OLMo 2 model at a specific training checkpoint."""
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else model_name
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision}...")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model_key: str, device: str, output_dir: str,
                   compositions: list = None):
    if compositions is None:
        compositions = ["down_up"]

    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["name"]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    d_ff = cfg["d_ff"]
    aspect_ratio = d_ff / d_model
    checkpoints = cfg["checkpoints"]

    # Load existing results if available (to skip already-computed checkpoints)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"svd_emergence_olmo_{model_key}.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_path} "
              f"({len(results['checkpoints'])} checkpoints already computed)")
    else:
        results = {
            "model": model_key,
            "model_name": model_name,
            "model_family": "olmo",
            "d_model": d_model,
            "d_ff": d_ff,
            "n_layers": n_layers,
            "compositions": compositions,
            "note": (
                "OLMo uses SwiGLU: output = W_down @ (act(W_gate @ x) * W_up @ x). "
                "The 'down_up' composition (W_down @ W_up) is the direct analog of "
                "the Pythia composed product. 'down_gate' captures the gating path."
            ),
            "checkpoints": {},
        }

    for step in checkpoints:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model = load_model_at_checkpoint(model_name, step, device)

        step_results = {"layers": {}}

        for layer_idx in range(n_layers):
            layer_data = {}

            for comp in compositions:
                composed = get_composed_mlp_weights(
                    model, layer_idx, composition=comp
                )
                sv = np.linalg.svd(composed, compute_uv=False)
                sv = np.sort(sv)[::-1]

                prefix = comp + "_" if len(compositions) > 1 else ""

                layer_data[f"{prefix}singular_values_top50"] = sv[:50].tolist()
                layer_data[f"{prefix}singular_values_tail20"] = sv[-20:].tolist()
                layer_data[f"{prefix}n_singular_values"] = len(sv)
                layer_data[f"{prefix}effective_rank"] = effective_rank(sv)
                layer_data[f"{prefix}mp_kl_divergence"] = marchenko_pastur_kl(sv, aspect_ratio)
                layer_data[f"{prefix}max_sv"] = float(sv[0])
                layer_data[f"{prefix}median_sv"] = float(np.median(sv))
                mm = float(sv[0] / np.median(sv)) if np.median(sv) > 1e-12 else float('inf')
                layer_data[f"{prefix}max_over_median"] = mm

                for k in TOP_K_VALUES:
                    if k < len(sv):
                        layer_data[f"{prefix}energy_ratio_top{k}"] = top_k_energy_ratio(sv, k)
                        layer_data[f"{prefix}gap_at_{k}"] = bulk_tail_gap(sv, k)

            step_results["layers"][str(layer_idx)] = layer_data

            # Print summary for down_up (or the only composition)
            p = (compositions[0] + "_") if len(compositions) > 1 else ""
            print(f"    Layer {layer_idx:2d}: "
                  f"eff_rank={layer_data[f'{p}effective_rank']:.1f}, "
                  f"top10_energy={layer_data.get(f'{p}energy_ratio_top10', 'N/A'):.3f}, "
                  f"MP_KL={layer_data[f'{p}mp_kl_divergence']:.3f}, "
                  f"max/med={layer_data[f'{p}max_over_median']:.2f}")

        results["checkpoints"][str(step)] = step_results

        del model
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
    fig.suptitle(f"SVD Bulk-Tail Emergence: OLMo-{model_key.upper()}", fontsize=14)

    if n_layers <= 6:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [0, n_layers // 4, n_layers // 2,
                         3 * n_layers // 4, n_layers - 1]

    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

    # Determine key prefix (empty string if single composition)
    has_prefix = any(
        k.startswith("down_up_") for k in
        results["checkpoints"][str(checkpoints[0])]["layers"]["0"]
    )
    p = "down_up_" if has_prefix else ""

    # Panel 1: Effective rank over training
    ax = axes[0, 0]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)][f"{p}effective_rank"]
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title("Composed effective rank (W_down @ W_up)")
    ax.legend(fontsize=8)

    # Panel 2: Top-10 energy ratio
    ax = axes[0, 1]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)].get(f"{p}energy_ratio_top10", 0)
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Energy in top-10 SVs")
    ax.set_title("Top-10 energy concentration")
    ax.legend(fontsize=8)

    # Panel 3: MP KL divergence
    ax = axes[0, 2]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)][f"{p}mp_kl_divergence"]
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL(empirical || MP)")
    ax.set_title("Deviation from random (MP) spectrum")
    ax.legend(fontsize=8)

    # Panel 4: Max/median SV ratio
    ax = axes[1, 0]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)][f"{p}max_over_median"]
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("max(sv) / median(sv)")
    ax.set_title("Max-to-median SV ratio (spread)")
    ax.legend(fontsize=8)

    # Panel 5: Gap at k=10
    ax = axes[1, 1]
    for i, li in enumerate(layer_indices):
        vals = [results["checkpoints"][str(s)]["layers"][str(li)].get(f"{p}gap_at_10", 1.0)
                for s in checkpoints]
        ax.plot(checkpoints, vals, 'o-', color=colors[i], label=f"Layer {li}")
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("\u03c3\u2081\u2080 / \u03c3\u2081\u2081")
    ax.set_title("Bulk-tail gap at k=10")
    ax.legend(fontsize=8)

    # Panel 6: SV spectrum snapshots for final layer
    ax = axes[1, 2]
    final_layer = str(n_layers - 1)
    snapshot_steps = [checkpoints[0], checkpoints[len(checkpoints)//2], checkpoints[-1]]
    for j, s in enumerate(snapshot_steps):
        sv_top = results["checkpoints"][str(s)]["layers"][final_layer][f"{p}singular_values_top50"]
        ax.plot(range(len(sv_top)), sv_top, '-', alpha=0.8, label=f"Step {s}")
    ax.set_xlabel("SV index (sorted descending)")
    ax.set_ylabel("Singular value")
    ax.set_title(f"SV spectrum snapshots (layer {final_layer})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"svd_emergence_olmo_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment A (OLMo): SVD bulk-tail emergence")
    parser.add_argument("--models", nargs="+", default=["1b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results/experiment_a_olmo")
    parser.add_argument("--both_compositions", action="store_true",
                        help="Compute both down_up and down_gate compositions")
    args = parser.parse_args()

    compositions = ["down_up", "down_gate"] if args.both_compositions else ["down_up"]

    for model_key in args.models:
        results = run_experiment(model_key, args.device, args.output_dir,
                                compositions=compositions)
        plot_results(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
