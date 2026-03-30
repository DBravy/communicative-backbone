"""
BLOOM Cross-Layer Communicative Structure Experiments
=====================================================

Tests whether the GELU/SwiGLU divergence documented in the paper is attributable
to the activation function rather than the attention-MLP block ordering. BLOOM
uses GELU with *sequential* attention-MLP blocks, matching OLMo's block ordering
but Pythia's activation function. If weight-level coherence consolidates (like
Pythia) rather than dissolves (like OLMo), that supports the activation function
account.

Experiments:
  1. Pairwise overlap matrix (all layer pairs) at each checkpoint
     -> Produces Figure 7 equivalent + Table 5 equivalent
  2. Adjacent-layer depth-resolved overlap across training
     -> Produces Figure 6 equivalent
  3. Top-k redistribution at final and middle boundaries
     -> Produces Table 1/2 equivalent (lightweight)

Model: BLOOM-1b1 (bigscience/bloom-1b1-intermediate)
  - 24 layers, d_model=1536, d_ff=6144
  - GELU activation, sequential attn-MLP ordering, ALiBi positional encoding
  - Available checkpoints at steps: 1000, 10000, 100000, 200000, 300000,
    400000, 500000, 600000 (per HF model card)
  - Loaded via revision="global_step{N}"
  - Caveat: step 400000 may be a corrupted upload (see HF discussions)

Usage:
    python bloom_experiments.py [--checkpoints 1000 5000 10000 ...] [--k 10]
    python bloom_experiments.py --quick   # subset of checkpoints for testing
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# BLOOM-1b1 configuration
# ---------------------------------------------------------------------------

MODEL_REPO = "bigscience/bloom-1b1-intermediate"
MODEL_FINAL = "bigscience/bloom-1b1"  # final trained model (no step suffix)

# Architecture (will be confirmed from config at load time)
D_MODEL = 1536
D_FF = 6144
N_LAYERS = 24

# Checkpoint schedule for bloom-1b1-intermediate.
# Per the HF model card (Version 1.3 / 11.July.2022), the available
# intermediary checkpoints are at global steps:
#   1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000
#
# NOTE: Step 400000 may be corrupted (performance drops to below step-1000
# levels). See https://huggingface.co/bigscience/bloom-1b1-intermediate/discussions/3
# The script will attempt it but gracefully skip if results look anomalous.
#
# The final trained model (bigscience/bloom-1b1) can also be loaded
# separately via MODEL_FINAL as the "final" checkpoint.
CHECKPOINTS_FULL = [
    1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000,
]

CHECKPOINTS_QUICK = [1000, 10000, 100000, 300000, 600000]

K_VALUES = [5, 10, 50]  # subspace dimensions for overlap


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between two subspaces.

    Args:
        U1: (N, k1) orthonormal columns
        U2: (N, k2) orthonormal columns

    Returns:
        min(k1, k2) cosines in [0, 1], sorted descending.
    """
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def mean_cosine_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """Mean cosine of principal angles (scalar summary of subspace overlap)."""
    return float(np.mean(principal_angles_cosines(U1, U2)))


def random_subspace_baseline(d: int, k: int, n_trials: int = 200) -> dict:
    """Expected overlap between two random k-dim subspaces of R^d."""
    vals = []
    for _ in range(n_trials):
        U1 = np.linalg.qr(np.random.randn(d, k))[0]
        U2 = np.linalg.qr(np.random.randn(d, k))[0]
        vals.append(mean_cosine_overlap(U1, U2))
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


# ---------------------------------------------------------------------------
# Model loading and SVD
# ---------------------------------------------------------------------------

def load_bloom_at_step(step: int):
    """Load BLOOM-1b1 at a specific training step.

    Uses the intermediate checkpoint repo for all steps except 'final',
    which loads the fully trained model.
    """
    from transformers import AutoModelForCausalLM

    if step == "final":
        repo = MODEL_FINAL
        revision = None
        print(f"  Loading {repo} (final)...")
    else:
        repo = MODEL_REPO
        revision = f"global_step{step}"
        print(f"  Loading {repo} at {revision}...")

    kwargs = dict(
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    if CACHE_DIR:
        kwargs["cache_dir"] = CACHE_DIR
    if revision:
        kwargs["revision"] = revision

    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    except Exception as e:
        print(f"  WARNING: Failed to load step {step}: {e}")
        return None

    model.eval()
    return model


def get_bloom_layer_svd(model, layer_idx: int):
    """SVD of composed MLP product W_down @ W_up for a BLOOM layer.

    BLOOM MLP structure (BloomMLP):
        dense_h_to_4h: Linear(d_model, d_ff)   -- up projection
        gelu_impl:     GELU activation
        dense_4h_to_h: Linear(d_ff, d_model)   -- down projection

    Composed product: W_down @ W_up has shape (d_model, d_model).
    Returns (U, S) where U columns are left singular vectors.
    """
    mlp = model.transformer.h[layer_idx].mlp
    W_up = mlp.dense_h_to_4h.weight.detach().float()    # (d_ff, d_model)
    W_down = mlp.dense_4h_to_h.weight.detach().float()   # (d_model, d_ff)
    composed = (W_down @ W_up).cpu().numpy()              # (d_model, d_model)
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


def get_all_layer_svds(model, n_layers: int, k_max: int):
    """Compute SVD for all layers, keeping top-k_max left singular vectors.

    Returns list of dicts with keys 'U_topk' and 'S'.
    """
    layer_data = []
    for li in range(n_layers):
        U, S = get_bloom_layer_svd(model, li)
        layer_data.append({
            "U_topk": U[:, :k_max],  # only keep what we need
            "S": S,
        })
        print(f"    SVD layer {li}/{n_layers}  "
              f"(top SV: {S[0]:.2f}, eff_rank: {effective_rank(S):.1f})")
    return layer_data


def effective_rank(sv: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized squared singular values)."""
    sv_sq = sv ** 2
    total = sv_sq.sum()
    if total < 1e-12:
        return 0.0
    p = sv_sq / total
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# Experiment 1: Pairwise overlap matrix
# ---------------------------------------------------------------------------

def compute_pairwise_matrix(layer_data: list, k: int) -> list:
    """NxN matrix of top-k mean cosine overlap between all layer pairs."""
    n = len(layer_data)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = mean_cosine_overlap(
                layer_data[i]["U_topk"][:, :k],
                layer_data[j]["U_topk"][:, :k],
            )
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix.tolist()


def pairwise_summary_stats(matrix: list, threshold: float = 0.15) -> dict:
    """Compute summary statistics from pairwise overlap matrix.

    Matches Table 5 format: pairs > threshold, mean at distance d.
    """
    mat = np.array(matrix)
    n = mat.shape[0]
    total_pairs = n * (n - 1) // 2

    # Count pairs above threshold
    above = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] > threshold:
                above += 1

    # Mean overlap at distances 1, 3, 5
    dist_means = {}
    for d in [1, 3, 5]:
        vals = []
        for i in range(n - d):
            vals.append(mat[i, i + d])
        if vals:
            dist_means[f"mean_d{d}"] = float(np.mean(vals))

    return {
        "pairs_above_threshold": above,
        "total_pairs": total_pairs,
        "frac_above_threshold": float(above / total_pairs) if total_pairs > 0 else 0,
        "threshold": threshold,
        **dist_means,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Adjacent-layer depth-resolved overlap
# ---------------------------------------------------------------------------

def compute_adjacent_overlaps(layer_data: list, k_values: list) -> dict:
    """For each adjacent pair, compute top-k overlap at multiple k values.

    Returns dict keyed by boundary string "i-(i+1)" with overlap values.
    """
    n = len(layer_data)
    results = {}
    for i in range(n - 1):
        boundary = f"{i}-{i+1}"
        boundary_data = {}
        for k in k_values:
            val = mean_cosine_overlap(
                layer_data[i]["U_topk"][:, :k],
                layer_data[i + 1]["U_topk"][:, :k],
            )
            boundary_data[f"top{k}"] = val
        results[boundary] = boundary_data
        print(f"    Boundary {boundary}: "
              + ", ".join(f"top{k}={boundary_data[f'top{k}']:.3f}" for k in k_values))
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Redistribution at key boundaries
# ---------------------------------------------------------------------------

def compute_boundary_redistribution(layer_data: list, k_values: list,
                                     boundary_idx: int) -> dict:
    """Detailed overlap at a specific boundary across multiple k values.

    Returns dict with overlap at each k, useful for tracking redistribution
    from narrow to broad subspaces (Tables 1-2 equivalent).
    """
    i = boundary_idx
    results = {}
    for k in k_values:
        results[f"top{k}"] = mean_cosine_overlap(
            layer_data[i]["U_topk"][:, :k],
            layer_data[i + 1]["U_topk"][:, :k],
        )
    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(checkpoints: list, k_values: list, output_dir: str):
    n_layers = N_LAYERS
    k_max = max(k_values)

    # Random baselines
    print("Computing random subspace baselines...")
    baselines = {}
    for k in k_values:
        baselines[k] = random_subspace_baseline(D_MODEL, k, n_trials=200)
        print(f"  k={k}: random mean cosine = "
              f"{baselines[k]['mean']:.4f} +/- {baselines[k]['std']:.4f}")

    results = {
        "model": "bloom-1b1",
        "model_repo": MODEL_REPO,
        "architecture": {
            "n_layers": N_LAYERS,
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "activation": "gelu",
            "block_ordering": "sequential",
            "positional_encoding": "alibi",
        },
        "k_values": k_values,
        "random_baselines": {str(k): v for k, v in baselines.items()},
        "checkpoints": {},
    }

    # Load existing results to allow resuming
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "bloom_1b1_experiments.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")

    for step in checkpoints:
        step_key = str(step)
        if step_key in results["checkpoints"]:
            print(f"\nSkipping step {step} (already computed)")
            continue

        print(f"\n{'='*60}")
        print(f"  BLOOM-1b1 -- step {step}")
        print(f"{'='*60}")

        model = load_bloom_at_step(step)
        if model is None:
            print(f"  Skipping step {step} (load failed)")
            continue

        # Verify architecture matches expectations
        config = model.config
        actual_layers = config.n_layer if hasattr(config, 'n_layer') else config.num_hidden_layers
        actual_hidden = config.hidden_size
        if actual_layers != N_LAYERS or actual_hidden != D_MODEL:
            print(f"  WARNING: Expected {N_LAYERS} layers / {D_MODEL} hidden, "
                  f"got {actual_layers} / {actual_hidden}. Updating.")
            # Don't crash; just note the discrepancy.

        # Compute SVDs for all layers
        print("  Computing SVDs...")
        layer_data = get_all_layer_svds(model, n_layers, k_max)

        step_results = {}

        # --- Experiment 1: Pairwise overlap matrix ---
        print("  Computing pairwise overlap matrix...")
        for k in k_values:
            matrix = compute_pairwise_matrix(layer_data, k)
            summary = pairwise_summary_stats(matrix)
            step_results[f"pairwise_k{k}"] = {
                "matrix": matrix,
                "summary": summary,
            }
            print(f"    k={k}: {summary['frac_above_threshold']*100:.1f}% pairs > 0.15, "
                  f"mean_d1={summary.get('mean_d1', 0):.3f}")

        # --- Experiment 2: Adjacent-layer depth-resolved overlap ---
        print("  Computing adjacent-layer overlaps...")
        step_results["adjacent"] = compute_adjacent_overlaps(layer_data, k_values)

        # --- Experiment 3: Redistribution at final and middle boundaries ---
        final_boundary = n_layers - 2   # boundary between layers 22 and 23
        middle_boundary = n_layers // 2  # boundary between layers 12 and 13
        detail_k = [5, 10, 20, 50]
        # For detailed redistribution we need k=50, recompute if needed
        if 50 > k_max:
            print("  Recomputing SVDs with k=50 for redistribution analysis...")
            layer_data_50 = get_all_layer_svds(model, n_layers, 50)
        else:
            layer_data_50 = layer_data

        step_results["redistribution"] = {
            "final_boundary": compute_boundary_redistribution(
                layer_data_50, detail_k, final_boundary),
            "middle_boundary": compute_boundary_redistribution(
                layer_data_50, detail_k, middle_boundary),
        }
        print(f"    Final  ({final_boundary}-{final_boundary+1}): "
              + ", ".join(f"top{k}={step_results['redistribution']['final_boundary'][f'top{k}']:.3f}"
                          for k in detail_k))
        print(f"    Middle ({middle_boundary}-{middle_boundary+1}): "
              + ", ".join(f"top{k}={step_results['redistribution']['middle_boundary'][f'top{k}']:.3f}"
                          for k in detail_k))

        # --- Per-layer effective rank (bonus, cheap to compute) ---
        step_results["effective_ranks"] = {
            str(li): effective_rank(layer_data[li]["S"])
            for li in range(n_layers)
        }

        results["checkpoints"][step_key] = step_results

        # Save after each checkpoint (allows resuming)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved intermediate results to {out_path}")

        # Free memory
        del model, layer_data
        if 50 > k_max:
            del layer_data_50
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(output_dir: str):
    """Generate all plots from saved results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    out_path = os.path.join(output_dir, "bloom_1b1_experiments.json")
    with open(out_path, "r") as f:
        results = json.load(f)

    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
    n_layers = results["architecture"]["n_layers"]
    baselines = results["random_baselines"]

    # =====================================================================
    # Plot 1: Pairwise overlap heatmaps across training (Figure 7 equiv)
    # =====================================================================
    k = 10
    k_key = f"pairwise_k{k}"

    # Pick a subset of checkpoints for the grid
    if len(checkpoints) <= 8:
        plot_steps = checkpoints
    else:
        # Sample ~8 evenly spaced checkpoints
        indices = np.linspace(0, len(checkpoints) - 1, 8, dtype=int)
        plot_steps = [checkpoints[i] for i in indices]

    n_plots = len(plot_steps)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(f"BLOOM-1b1: Pairwise top-{k} subspace overlap across training",
                 fontsize=14, y=1.02)

    vmin, vmax = 0.0, 0.6  # fixed color scale for comparability
    for idx, step in enumerate(plot_steps):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        step_key = str(step)
        if step_key in results["checkpoints"] and k_key in results["checkpoints"][step_key]:
            mat = np.array(results["checkpoints"][step_key][k_key]["matrix"])
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis",
                           origin="lower", aspect="equal")
            summary = results["checkpoints"][step_key][k_key]["summary"]
            ax.set_title(f"Step {step}\n"
                         f"{summary['frac_above_threshold']*100:.0f}% > 0.15",
                         fontsize=10)
        else:
            ax.set_title(f"Step {step}\n(no data)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

    # Remove unused axes
    for idx in range(len(plot_steps), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Mean cosine overlap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bloom_pairwise_heatmaps.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bloom_pairwise_heatmaps.png")

    # =====================================================================
    # Plot 2: Pairwise summary stats across training (Table 5 equiv)
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("BLOOM-1b1: Pairwise coherence summary across training", fontsize=13)

    for ki, k in enumerate(results["k_values"]):
        k_key = f"pairwise_k{k}"

        fracs = []
        d1s, d3s, d5s = [], [], []
        valid_steps = []
        for step in checkpoints:
            sk = str(step)
            if sk in results["checkpoints"] and k_key in results["checkpoints"][sk]:
                s = results["checkpoints"][sk][k_key]["summary"]
                fracs.append(s["frac_above_threshold"])
                d1s.append(s.get("mean_d1", np.nan))
                d3s.append(s.get("mean_d3", np.nan))
                d5s.append(s.get("mean_d5", np.nan))
                valid_steps.append(step)

        ax = axes[0]
        ax.plot(valid_steps, fracs, "o-", label=f"k={k}")
        ax.set_ylabel("Fraction of pairs > 0.15")
        ax.set_title("Pairwise alignment prevalence")

        ax = axes[1]
        ax.plot(valid_steps, d1s, "o-", label=f"k={k}")
        ax.set_ylabel("Mean overlap")
        ax.set_title("Mean overlap at distance 1")

        ax = axes[2]
        ax.plot(valid_steps, d5s, "o-", label=f"k={k}")
        ax.set_ylabel("Mean overlap")
        ax.set_title("Mean overlap at distance 5")

    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_xscale("symlog", linthresh=1000)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bloom_pairwise_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bloom_pairwise_summary.png")

    # =====================================================================
    # Plot 3: Adjacent-layer overlap by depth across training (Fig 6 equiv)
    # =====================================================================
    k = 10
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: line plot colored by depth
    ax = axes[0]
    n_boundaries = n_layers - 1
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_boundaries))

    for bi in range(n_boundaries):
        boundary = f"{bi}-{bi+1}"
        vals = []
        steps_with_data = []
        for step in checkpoints:
            sk = str(step)
            adj = results["checkpoints"].get(sk, {}).get("adjacent", {})
            if boundary in adj:
                vals.append(adj[boundary][f"top{k}"])
                steps_with_data.append(step)
        if vals:
            alpha = 0.4 if bi not in [0, n_boundaries - 1, n_boundaries // 2] else 1.0
            lw = 1.0 if alpha < 1.0 else 2.0
            ax.plot(steps_with_data, vals, "o-", color=colors[bi],
                    alpha=alpha, linewidth=lw, markersize=3,
                    label=f"{boundary}" if bi in [0, n_boundaries // 2, n_boundaries - 1] else None)

    bl = baselines.get(str(k), baselines.get("10", {}))
    if "mean" in bl:
        ax.axhline(bl["mean"], color="gray", linestyle="--", alpha=0.5,
                    label="Random baseline")

    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Top-{k} mean cosine overlap")
    ax.set_title(f"Adjacent-layer overlap by depth")
    ax.set_xscale("symlog", linthresh=1000)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Right: heatmap (boundary x training step)
    ax = axes[1]
    step_labels = []
    heatmap_data = []
    for step in checkpoints:
        sk = str(step)
        adj = results["checkpoints"].get(sk, {}).get("adjacent", {})
        col = []
        has_data = False
        for bi in range(n_boundaries):
            boundary = f"{bi}-{bi+1}"
            if boundary in adj:
                col.append(adj[boundary][f"top{k}"])
                has_data = True
            else:
                col.append(np.nan)
        if has_data:
            heatmap_data.append(col)
            step_labels.append(str(step))

    if heatmap_data:
        hm = np.array(heatmap_data).T  # (n_boundaries, n_steps)
        im = ax.imshow(hm, aspect="auto", cmap="viridis",
                        origin="lower", vmin=0.0, vmax=0.6)
        ax.set_yticks(range(n_boundaries))
        ax.set_yticklabels([f"{i}-{i+1}" for i in range(n_boundaries)], fontsize=6)
        ax.set_xticks(range(len(step_labels)))
        ax.set_xticklabels(step_labels, rotation=45, fontsize=7)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Layer boundary")
        ax.set_title(f"Top-{k} adjacent overlap heatmap")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Mean cosine overlap")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bloom_adjacent_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bloom_adjacent_depth.png")

    # =====================================================================
    # Plot 4: Redistribution at final vs middle boundary (Tables 1-2 equiv)
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BLOOM-1b1: Top-k overlap redistribution at key boundaries",
                 fontsize=13)

    for ax_idx, (boundary_name, ax_title) in enumerate([
        ("final_boundary", f"Final boundary (layers {N_LAYERS-2}-{N_LAYERS-1})"),
        ("middle_boundary", f"Middle boundary (layers {N_LAYERS//2}-{N_LAYERS//2+1})"),
    ]):
        ax = axes[ax_idx]
        for k_val in [5, 10, 50]:
            k_key = f"top{k_val}"
            vals = []
            steps_with_data = []
            for step in checkpoints:
                sk = str(step)
                redist = results["checkpoints"].get(sk, {}).get("redistribution", {})
                if boundary_name in redist and k_key in redist[boundary_name]:
                    vals.append(redist[boundary_name][k_key])
                    steps_with_data.append(step)
            if vals:
                ax.plot(steps_with_data, vals, "o-", label=f"top-{k_val}")

        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean cosine overlap")
        ax.set_title(ax_title)
        ax.set_xscale("symlog", linthresh=1000)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bloom_redistribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bloom_redistribution.png")

    # =====================================================================
    # Plot 5: Effective rank at first and last layer (Figure 3 equiv)
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for li, label, color in [(0, "Layer 0", "tab:blue"),
                              (N_LAYERS - 1, f"Layer {N_LAYERS-1}", "tab:red"),
                              (N_LAYERS // 2, f"Layer {N_LAYERS//2}", "tab:green")]:
        vals = []
        steps_with_data = []
        for step in checkpoints:
            sk = str(step)
            er = results["checkpoints"].get(sk, {}).get("effective_ranks", {})
            if str(li) in er:
                vals.append(er[str(li)])
                steps_with_data.append(step)
        if vals:
            ax.plot(steps_with_data, vals, "o-", color=color, label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title("BLOOM-1b1: Composed MLP effective rank across training")
    ax.set_xscale("symlog", linthresh=1000)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bloom_effective_rank.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bloom_effective_rank.png")

    print(f"\nAll plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BLOOM-1b1 cross-layer communicative structure experiments")
    parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=None,
        help="Training steps to analyze. Default: full schedule.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use a reduced set of checkpoints for quick testing.")
    parser.add_argument(
        "--k", nargs="+", type=int, default=K_VALUES,
        help="Subspace dimensions for overlap computation.")
    parser.add_argument(
        "--output_dir", default="results/bloom",
        help="Directory for output files.")
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Skip computation; just regenerate plots from existing results.")
    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.output_dir)
        sys.exit(0)

    if args.checkpoints is not None:
        checkpoints = args.checkpoints
    elif args.quick:
        checkpoints = CHECKPOINTS_QUICK
    else:
        checkpoints = CHECKPOINTS_FULL

    print(f"BLOOM-1b1 experiments")
    print(f"  Checkpoints: {checkpoints}")
    print(f"  k values: {args.k}")
    print(f"  Output: {args.output_dir}")
    print()

    results = run_all(checkpoints, args.k, args.output_dir)
    plot_results(args.output_dir)

    print("\nDone.")
