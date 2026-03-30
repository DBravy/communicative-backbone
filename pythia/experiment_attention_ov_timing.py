"""
Attention OV Cross-Layer Alignment Timing
==========================================

Tests whether attention heads develop cross-layer communicative structure
before, during, or after the MLP backbone documented in the paper.

For each layer, computes the combined OV matrix:

    OV_combined(l) = sum_h  W_O[l,h] @ W_V[l,h]

This is the attention analog of the composed MLP product W_down @ W_up:
it captures the total linear map from residual stream input to residual
stream output through the attention mechanism (ignoring the attention
pattern, which is data-dependent). Its top singular vectors identify the
directions that attention reads from and writes to the residual stream.

At each checkpoint, computes:
  1. SVD of OV_combined at each layer
  2. Adjacent-layer top-k subspace overlap (principal angles) for OV
  3. The same for the MLP composed product, for direct comparison
  4. Effective rank of both OV and MLP composed products

The critical output is a side-by-side comparison of OV vs MLP alignment
trajectories across training. If MLP alignment precedes OV alignment,
the "developmental priority" claim is strengthened. If they co-occur or
OV leads, the story needs revision.

Models: Pythia-410M, Pythia-1B, Pythia-1.4B
Checkpoints: 0, 128, 512, 2000, 8000, 32000, 64000, 143000

Usage:
    python experiment_attention_ov_timing.py [--models 410m 1b 1.4b]
"""

import argparse
import json
import os

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]

MODEL_CONFIGS = {
    "410m": {
        "name": "EleutherAI/pythia-410m",
        "d_model": 1024, "n_heads": 16, "d_head": 64,
        "d_ff": 4096, "n_layers": 24,
    },
    "1b": {
        "name": "EleutherAI/pythia-1b",
        "d_model": 2048, "n_heads": 16, "d_head": 128,
        "d_ff": 8192, "n_layers": 16,
    },
    "1.4b": {
        "name": "EleutherAI/pythia-1.4b",
        "d_model": 2048, "n_heads": 16, "d_head": 128,
        "d_ff": 8192, "n_layers": 24,
    },
}

K_VALUES = [5, 10, 50]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between two subspaces.

    Args:
        U1: (N, k) orthonormal columns spanning subspace 1
        U2: (N, k) orthonormal columns spanning subspace 2

    Returns:
        Array of min(k1, k2) cosines, sorted descending. In [0, 1].
    """
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def random_subspace_baseline(d: int, k: int, n_trials: int = 200) -> dict:
    """Expected overlap between two random k-dim subspaces of R^d."""
    cosines_all = []
    for _ in range(n_trials):
        U1 = np.linalg.qr(np.random.randn(d, k))[0]
        U2 = np.linalg.qr(np.random.randn(d, k))[0]
        cosines = principal_angles_cosines(U1, U2)
        cosines_all.append(np.mean(cosines))
    return {
        "mean": float(np.mean(cosines_all)),
        "std": float(np.std(cosines_all)),
    }


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


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def get_ov_combined(model, layer_idx: int, n_heads: int, d_head: int):
    """Compute the combined OV matrix for a Pythia (GPT-NeoX) layer.

    OV_combined = sum_h W_O[h] @ W_V[h]

    GPT-NeoX QKV layout:
        query_key_value weight: (3 * n_heads * d_head, d_model)
        Reshaped to (n_heads, 3 * d_head, d_model), then for each head:
            Q = [:d_head, :], K = [d_head:2*d_head, :], V = [2*d_head:, :]

    Output projection:
        dense weight: (d_model, n_heads * d_head)
        For head h: W_O[h] = dense[:, h*d_head:(h+1)*d_head]

    Returns:
        composed: (d_model, d_model) combined OV matrix
        U, S from SVD of composed
    """
    attn = model.gpt_neox.layers[layer_idx].attention
    qkv_w = attn.query_key_value.weight.detach().float()  # (3*d_model, d_model)
    dense_w = attn.dense.weight.detach().float()           # (d_model, d_model)

    d_model = dense_w.shape[0]

    ov_sum = torch.zeros(d_model, d_model, dtype=torch.float32)

    for h in range(n_heads):
        # V projection for head h: (d_head, d_model)
        v_start = h * 3 * d_head + 2 * d_head
        v_end = h * 3 * d_head + 3 * d_head
        W_V_h = qkv_w[v_start:v_end, :]

        # O projection for head h: (d_model, d_head)
        W_O_h = dense_w[:, h * d_head:(h + 1) * d_head]

        # OV for this head: (d_model, d_model)
        ov_sum += W_O_h @ W_V_h

    composed = ov_sum.cpu().numpy()
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


def get_mlp_composed(model, layer_idx: int):
    """SVD of composed MLP product W_down @ W_up for Pythia."""
    mlp = model.gpt_neox.layers[layer_idx].mlp
    W_up = mlp.dense_h_to_4h.weight.detach().float()    # (d_ff, d_model)
    W_down = mlp.dense_4h_to_h.weight.detach().float()   # (d_model, d_ff)
    composed = (W_down @ W_up).cpu().numpy()              # (d_model, d_model)
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_at_checkpoint(model_name: str, step: int):
    """Load a Pythia model at a specific training checkpoint."""
    from transformers import AutoModelForCausalLM
    revision = f"step{step}"
    print(f"  Loading {model_name} at {revision}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
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
    n_heads = cfg["n_heads"]
    d_head = cfg["d_head"]
    d_model = cfg["d_model"]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ov_timing_{model_key}.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        # Random baselines
        print("Computing random subspace baselines...")
        baselines = {}
        for k in K_VALUES:
            baselines[str(k)] = random_subspace_baseline(d_model, k)
            bl = baselines[str(k)]
            print(f"  k={k}: random mean_cosine = {bl['mean']:.4f} +/- {bl['std']:.4f}")

        results = {
            "model": model_key,
            "model_name": model_name,
            "model_family": "pythia",
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_head": d_head,
            "k_values": K_VALUES,
            "random_baselines": baselines,
            "checkpoints": {},
        }

    for step in CHECKPOINTS:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'=' * 60}")

        model = load_model_at_checkpoint(model_name, step)

        # Extract SVDs for all layers, both OV and MLP
        ov_data = []
        mlp_data = []
        for li in range(n_layers):
            U_ov, S_ov = get_ov_combined(model, li, n_heads, d_head)
            U_mlp, S_mlp = get_mlp_composed(model, li)
            ov_data.append({"U": U_ov, "S": S_ov})
            mlp_data.append({"U": U_mlp, "S": S_mlp})
            print(f"    Layer {li}: OV eff_rank={effective_rank(S_ov):.1f}, "
                  f"MLP eff_rank={effective_rank(S_mlp):.1f}")

        step_results = {
            "adjacent_overlap": {},
            "per_layer": {},
        }

        # Per-layer effective rank and top singular values
        for li in range(n_layers):
            step_results["per_layer"][str(li)] = {
                "ov_effective_rank": effective_rank(ov_data[li]["S"]),
                "mlp_effective_rank": effective_rank(mlp_data[li]["S"]),
                "ov_top10_sv": ov_data[li]["S"][:10].tolist(),
                "mlp_top10_sv": mlp_data[li]["S"][:10].tolist(),
            }

        # Adjacent-layer overlap for both OV and MLP
        for li in range(n_layers - 1):
            pair_key = f"{li}_{li + 1}"
            pair_result = {}

            for k in K_VALUES:
                # OV overlap
                U1_ov = ov_data[li]["U"][:, :k]
                U2_ov = ov_data[li + 1]["U"][:, :k]
                cos_ov = principal_angles_cosines(U1_ov, U2_ov)

                # MLP overlap
                U1_mlp = mlp_data[li]["U"][:, :k]
                U2_mlp = mlp_data[li + 1]["U"][:, :k]
                cos_mlp = principal_angles_cosines(U1_mlp, U2_mlp)

                pair_result[f"ov_top{k}"] = {
                    "mean_cosine": float(np.mean(cos_ov)),
                    "max_cosine": float(np.max(cos_ov)),
                    "min_cosine": float(np.min(cos_ov)),
                }
                pair_result[f"mlp_top{k}"] = {
                    "mean_cosine": float(np.mean(cos_mlp)),
                    "max_cosine": float(np.max(cos_mlp)),
                    "min_cosine": float(np.min(cos_mlp)),
                }

            step_results["adjacent_overlap"][pair_key] = pair_result

            # Print summary for k=10
            ov_mc = pair_result["ov_top10"]["mean_cosine"]
            mlp_mc = pair_result["mlp_top10"]["mean_cosine"]
            print(f"    Boundary {li}-{li+1}: OV={ov_mc:.4f}  MLP={mlp_mc:.4f}")

        # Summary: mean across all adjacent boundaries
        for k in K_VALUES:
            ov_vals = [step_results["adjacent_overlap"][pk][f"ov_top{k}"]["mean_cosine"]
                       for pk in step_results["adjacent_overlap"]]
            mlp_vals = [step_results["adjacent_overlap"][pk][f"mlp_top{k}"]["mean_cosine"]
                        for pk in step_results["adjacent_overlap"]]
            step_results[f"mean_ov_top{k}"] = float(np.mean(ov_vals))
            step_results[f"mean_mlp_top{k}"] = float(np.mean(mlp_vals))

        print(f"\n  Step {step} summary (top-10):")
        print(f"    Mean OV  overlap: {step_results['mean_ov_top10']:.4f}")
        print(f"    Mean MLP overlap: {step_results['mean_mlp_top10']:.4f}")

        results["checkpoints"][str(step)] = step_results

        del model, ov_data, mlp_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save after each checkpoint
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    model_key = results["model"]
    n_layers = results["n_layers"]
    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
    baselines = results["random_baselines"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Attention OV vs MLP Cross-Layer Alignment: Pythia-{model_key}",
        fontsize=14,
    )

    # ------------------------------------------------------------------
    # Panel 1: Mean adjacent overlap over training (OV vs MLP, k=10)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    k = 10
    ov_means = [results["checkpoints"][str(s)][f"mean_ov_top{k}"] for s in checkpoints]
    mlp_means = [results["checkpoints"][str(s)][f"mean_mlp_top{k}"] for s in checkpoints]

    ax.plot(checkpoints, mlp_means, "o-", color="tab:red", label="MLP (W_down W_up)", linewidth=2)
    ax.plot(checkpoints, ov_means, "s--", color="tab:blue", label="Attention (OV combined)", linewidth=2)

    bl = baselines[str(k)]
    ax.axhline(bl["mean"], color="gray", linestyle=":", alpha=0.7, label="Random baseline")
    ax.fill_between(
        checkpoints,
        bl["mean"] - 2 * bl["std"],
        bl["mean"] + 2 * bl["std"],
        color="gray", alpha=0.1,
    )
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine (adjacent layers)")
    ax.set_title(f"Mean adjacent-layer top-{k} overlap")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Panel 2: Delta from baseline (timing comparison)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    bl_val = bl["mean"]
    ov_delta = [v - bl_val for v in ov_means]
    mlp_delta = [v - bl_val for v in mlp_means]

    ax.plot(checkpoints, mlp_delta, "o-", color="tab:red", label="MLP above random", linewidth=2)
    ax.plot(checkpoints, ov_delta, "s--", color="tab:blue", label="OV above random", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine - random baseline")
    ax.set_title("Alignment above chance (timing)")
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 3: Multiple k values at key steps
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    compare_steps = [0, 512, 2000, 143000]
    compare_steps = [s for s in compare_steps if str(s) in results["checkpoints"]]
    x_pos = np.arange(len(K_VALUES))
    width = 0.8 / (2 * len(compare_steps))
    colors_mlp = plt.cm.Reds(np.linspace(0.3, 0.9, len(compare_steps)))
    colors_ov = plt.cm.Blues(np.linspace(0.3, 0.9, len(compare_steps)))

    for i, s in enumerate(compare_steps):
        ov_k = [results["checkpoints"][str(s)][f"mean_ov_top{kk}"] for kk in K_VALUES]
        mlp_k = [results["checkpoints"][str(s)][f"mean_mlp_top{kk}"] for kk in K_VALUES]
        offset_mlp = (i - len(compare_steps)) * width
        offset_ov = i * width
        ax.bar(x_pos + offset_mlp, mlp_k, width, color=colors_mlp[i],
               label=f"MLP s{s}" if i == 0 or i == len(compare_steps) - 1 else "")
        ax.bar(x_pos + offset_ov, ov_k, width, color=colors_ov[i],
               label=f"OV s{s}" if i == 0 or i == len(compare_steps) - 1 else "")

    # Random baselines
    bl_vals = [baselines[str(kk)]["mean"] for kk in K_VALUES]
    ax.plot(x_pos, bl_vals, "k--", alpha=0.5, label="Random")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(kk) for kk in K_VALUES])
    ax.set_xlabel("k (subspace dimension)")
    ax.set_ylabel("Mean cosine")
    ax.set_title("OV vs MLP by subspace size")
    ax.legend(fontsize=7, ncol=2)

    # ------------------------------------------------------------------
    # Panel 4: Per-boundary OV overlap (depth-resolved, k=10)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    for si, step in enumerate(checkpoints):
        if step in [0, 512, 2000, 8000, 143000]:
            pairs = results["checkpoints"][str(step)]["adjacent_overlap"]
            boundaries = sorted(pairs.keys(), key=lambda x: int(x.split("_")[0]))
            ov_vals = [pairs[b][f"ov_top{10}"]["mean_cosine"] for b in boundaries]
            color = cmap(si / max(len(checkpoints) - 1, 1))
            ax.plot(range(len(ov_vals)), ov_vals, "o-", color=color,
                    label=f"Step {step}", markersize=4)

    ax.set_xlabel("Layer boundary")
    ax.set_ylabel("OV top-10 mean cosine")
    ax.set_title("OV overlap by depth across training")
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel 5: Per-boundary MLP overlap (depth-resolved, k=10) 
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    for si, step in enumerate(checkpoints):
        if step in [0, 512, 2000, 8000, 143000]:
            pairs = results["checkpoints"][str(step)]["adjacent_overlap"]
            boundaries = sorted(pairs.keys(), key=lambda x: int(x.split("_")[0]))
            mlp_vals = [pairs[b][f"mlp_top{10}"]["mean_cosine"] for b in boundaries]
            color = cmap(si / max(len(checkpoints) - 1, 1))
            ax.plot(range(len(mlp_vals)), mlp_vals, "o-", color=color,
                    label=f"Step {step}", markersize=4)

    ax.set_xlabel("Layer boundary")
    ax.set_ylabel("MLP top-10 mean cosine")
    ax.set_title("MLP overlap by depth across training")
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel 6: Effective rank trajectories (OV vs MLP, layer 0 + final)
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    final_layer = str(n_layers - 1)

    ov_rank_0 = [results["checkpoints"][str(s)]["per_layer"]["0"]["ov_effective_rank"]
                 for s in checkpoints]
    mlp_rank_0 = [results["checkpoints"][str(s)]["per_layer"]["0"]["mlp_effective_rank"]
                  for s in checkpoints]
    ov_rank_f = [results["checkpoints"][str(s)]["per_layer"][final_layer]["ov_effective_rank"]
                 for s in checkpoints]
    mlp_rank_f = [results["checkpoints"][str(s)]["per_layer"][final_layer]["mlp_effective_rank"]
                  for s in checkpoints]

    ax.plot(checkpoints, mlp_rank_0, "o-", color="tab:red", label="MLP layer 0", linewidth=1.5)
    ax.plot(checkpoints, mlp_rank_f, "o-", color="darkred", label=f"MLP layer {final_layer}", linewidth=1.5)
    ax.plot(checkpoints, ov_rank_0, "s--", color="tab:blue", label="OV layer 0", linewidth=1.5)
    ax.plot(checkpoints, ov_rank_f, "s--", color="darkblue", label=f"OV layer {final_layer}", linewidth=1.5)

    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title("Spectral concentration: OV vs MLP")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"ov_timing_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention OV vs MLP cross-layer alignment timing")
    parser.add_argument("--models", nargs="+", default=["410m", "1b", "1.4b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output_dir", default="results/ov_timing")
    args = parser.parse_args()

    for model_key in args.models:
        results = run_experiment(model_key, args.output_dir)
        plot_results(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
