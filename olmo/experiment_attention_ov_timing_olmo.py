"""
Attention OV Cross-Layer Alignment Timing (OLMo-2-1B)
======================================================

OLMo port of experiment_attention_ov_timing.py (originally for Pythia).

Tests whether attention heads develop cross-layer communicative structure
before, during, or after the MLP backbone in OLMo-2-1B. The Pythia results
showed that MLP alignment leads (detectable at step 128), OV alignment
transiently surges during backbone formation (steps 512-2000), and then
OV decohereres while MLP holds. This experiment tests whether the same
pattern holds in a SwiGLU model where MLP weight-level alignment itself
eventually dissolves.

For each layer, computes the combined OV matrix:

    OV_combined(l) = sum_h  W_O[l,h] @ W_V[l,h]

This is the attention analog of the composed MLP product W_down @ W_up.

OLMo-2 uses the standard HF Llama attention layout with separate
q_proj, k_proj, v_proj, o_proj matrices. If the model uses grouped
query attention (n_kv_heads < n_heads), each V head is shared across
its group of Q heads, and the OV sum accounts for this.

Checkpoints span early and late training:
  Early (0-10000): from allenai/OLMo-2-0425-1B-early-training
  Late (100000, 1000000): from allenai/OLMo-2-0425-1B

Usage:
    python experiment_attention_ov_timing_olmo.py [--output_dir results/ov_timing_olmo]
"""

import argparse
import json
import math
import os

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

MODEL_NAME = "allenai/OLMo-2-0425-1B"
D_MODEL = 2048
D_FF = 8192
N_LAYERS = 16

# Dense early coverage + late checkpoints to see decoherence
CHECKPOINTS = [0, 1000, 2000, 3000, 5000, 10000, 100000, 1000000]

K_VALUES = [5, 10, 50]


# ---------------------------------------------------------------------------
# Checkpoint helpers (shared pattern with other OLMo experiments)
# ---------------------------------------------------------------------------

def step_to_revision(step: int) -> str:
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


def load_model_at_checkpoint(step: int):
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else MODEL_NAME
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
# Core computation
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between two subspaces."""
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

def get_attention_config(model):
    """Read attention head configuration from the model config.

    Returns (n_heads, n_kv_heads, d_head).
    Handles both MHA (n_kv_heads == n_heads) and GQA.
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    # num_key_value_heads may be absent (defaults to n_heads = MHA)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    d_head = cfg.hidden_size // n_heads
    return n_heads, n_kv_heads, d_head


def get_ov_combined(model, layer_idx: int,
                    n_heads: int, n_kv_heads: int, d_head: int):
    """Compute the combined OV matrix for an OLMo-2 (Llama-style) layer.

    OV_combined = sum over all Q-heads h of:  W_O[h] @ W_V[kv_group(h)]

    OLMo-2 HF layout (separate projections):
        v_proj.weight: (n_kv_heads * d_head, d_model)
        o_proj.weight: (d_model, n_heads * d_head)

    For head h:
        W_V[kv(h)] = v_proj[kv(h)*d_head : (kv(h)+1)*d_head, :]
        W_O[h]     = o_proj[:, h*d_head : (h+1)*d_head]

    Returns:
        U, S from SVD of the combined OV matrix (d_model, d_model)
    """
    attn = model.model.layers[layer_idx].self_attn
    W_V = attn.v_proj.weight.detach().float()   # (n_kv_heads * d_head, d_model)
    W_O = attn.o_proj.weight.detach().float()    # (d_model, n_heads * d_head)

    d_model = W_O.shape[0]
    heads_per_kv_group = n_heads // n_kv_heads

    ov_sum = torch.zeros(d_model, d_model, dtype=torch.float32)

    for h in range(n_heads):
        kv_idx = h // heads_per_kv_group

        # V slice for this head's KV group: (d_head, d_model)
        W_V_h = W_V[kv_idx * d_head:(kv_idx + 1) * d_head, :]

        # O slice for this head: (d_model, d_head)
        W_O_h = W_O[:, h * d_head:(h + 1) * d_head]

        ov_sum += W_O_h @ W_V_h

    composed = ov_sum.cpu().numpy()
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


def get_mlp_composed(model, layer_idx: int):
    """SVD of composed MLP product W_down @ W_up for OLMo-2."""
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()      # (d_ff, d_model)
    W_down = mlp.down_proj.weight.detach().float()   # (d_model, d_ff)
    composed = (W_down @ W_up).cpu().numpy()          # (d_model, d_model)
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ov_timing_olmo_1b.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        # Random baselines
        print("Computing random subspace baselines...")
        baselines = {}
        for k in K_VALUES:
            baselines[str(k)] = random_subspace_baseline(D_MODEL, k)
            bl = baselines[str(k)]
            print(f"  k={k}: random mean_cosine = {bl['mean']:.4f} +/- {bl['std']:.4f}")

        results = {
            "model": "1b",
            "model_name": MODEL_NAME,
            "model_family": "olmo",
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "k_values": K_VALUES,
            "random_baselines": baselines,
            "checkpoints": {},
        }

    for step in CHECKPOINTS:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  OLMo-2-1B -- step {step}")
        print(f"{'=' * 60}")

        model = load_model_at_checkpoint(step)

        # Read attention config from model (handles GQA automatically)
        n_heads, n_kv_heads, d_head = get_attention_config(model)
        if step == CHECKPOINTS[0]:
            print(f"  Attention config: n_heads={n_heads}, "
                  f"n_kv_heads={n_kv_heads}, d_head={d_head}")
            # Store in results for reference
            results["n_heads"] = n_heads
            results["n_kv_heads"] = n_kv_heads
            results["d_head"] = d_head

        # Extract SVDs for all layers
        ov_data = []
        mlp_data = []
        for li in range(N_LAYERS):
            U_ov, S_ov = get_ov_combined(model, li, n_heads, n_kv_heads, d_head)
            U_mlp, S_mlp = get_mlp_composed(model, li)
            ov_data.append({"U": U_ov, "S": S_ov})
            mlp_data.append({"U": U_mlp, "S": S_mlp})
            print(f"    Layer {li:2d}: OV eff_rank={effective_rank(S_ov):.1f}, "
                  f"MLP eff_rank={effective_rank(S_mlp):.1f}")

        step_results = {
            "adjacent_overlap": {},
            "per_layer": {},
        }

        # Per-layer effective rank and top singular values
        for li in range(N_LAYERS):
            step_results["per_layer"][str(li)] = {
                "ov_effective_rank": effective_rank(ov_data[li]["S"]),
                "mlp_effective_rank": effective_rank(mlp_data[li]["S"]),
                "ov_top10_sv": ov_data[li]["S"][:10].tolist(),
                "mlp_top10_sv": mlp_data[li]["S"][:10].tolist(),
            }

        # Adjacent-layer overlap for both OV and MLP
        for li in range(N_LAYERS - 1):
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
        print(f"  Saved to {out_path}")

    print(f"\nDone. Results saved to {out_path}")
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

    n_layers = results["n_layers"]
    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
    baselines = results["random_baselines"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Attention OV vs MLP Cross-Layer Alignment: OLMo-2-1B", fontsize=14)

    # ------------------------------------------------------------------
    # Panel 1: Mean adjacent overlap over training (OV vs MLP, k=10)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    k = 10
    ov_means = [results["checkpoints"][str(s)][f"mean_ov_top{k}"] for s in checkpoints]
    mlp_means = [results["checkpoints"][str(s)][f"mean_mlp_top{k}"] for s in checkpoints]

    ax.plot(checkpoints, mlp_means, "o-", color="tab:red",
            label="MLP (W_down W_up)", linewidth=2)
    ax.plot(checkpoints, ov_means, "s--", color="tab:blue",
            label="Attention (OV combined)", linewidth=2)

    bl = baselines[str(k)]
    ax.axhline(bl["mean"], color="gray", linestyle=":", alpha=0.7, label="Random baseline")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine (adjacent layers)")
    ax.set_title(f"Mean adjacent-layer top-{k} overlap")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Panel 2: Delta from baseline
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    bl_val = bl["mean"]
    ov_delta = [v - bl_val for v in ov_means]
    mlp_delta = [v - bl_val for v in mlp_means]

    ax.plot(checkpoints, mlp_delta, "o-", color="tab:red",
            label="MLP above random", linewidth=2)
    ax.plot(checkpoints, ov_delta, "s--", color="tab:blue",
            label="OV above random", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine - random baseline")
    ax.set_title("Alignment above chance")
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 3: Multiple k values at key steps
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    compare_steps = [0, 3000, 10000, 1000000]
    compare_steps = [s for s in compare_steps if str(s) in results["checkpoints"]]
    x_pos = np.arange(len(K_VALUES))
    width = 0.8 / (2 * max(len(compare_steps), 1))
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

    bl_vals = [baselines[str(kk)]["mean"] for kk in K_VALUES]
    ax.plot(x_pos, bl_vals, "k--", alpha=0.5, label="Random")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(kk) for kk in K_VALUES])
    ax.set_xlabel("k (subspace dimension)")
    ax.set_ylabel("Mean cosine")
    ax.set_title("OV vs MLP by subspace size")
    ax.legend(fontsize=7, ncol=2)

    # ------------------------------------------------------------------
    # Panel 4: Per-boundary OV overlap, depth-resolved (k=10)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    show_steps = [s for s in [0, 3000, 5000, 10000, 100000, 1000000]
                  if str(s) in results["checkpoints"]]
    for si, step in enumerate(show_steps):
        pairs = results["checkpoints"][str(step)]["adjacent_overlap"]
        boundaries = sorted(pairs.keys(), key=lambda x: int(x.split("_")[0]))
        ov_vals = [pairs[b]["ov_top10"]["mean_cosine"] for b in boundaries]
        color = cmap(si / max(len(show_steps) - 1, 1))
        ax.plot(range(len(ov_vals)), ov_vals, "o-", color=color,
                label=f"Step {step:,}", markersize=4)

    ax.set_xlabel("Layer boundary")
    ax.set_ylabel("OV top-10 mean cosine")
    ax.set_title("OV overlap by depth across training")
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel 5: Per-boundary MLP overlap, depth-resolved (k=10)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    for si, step in enumerate(show_steps):
        pairs = results["checkpoints"][str(step)]["adjacent_overlap"]
        boundaries = sorted(pairs.keys(), key=lambda x: int(x.split("_")[0]))
        mlp_vals = [pairs[b]["mlp_top10"]["mean_cosine"] for b in boundaries]
        color = cmap(si / max(len(show_steps) - 1, 1))
        ax.plot(range(len(mlp_vals)), mlp_vals, "o-", color=color,
                label=f"Step {step:,}", markersize=4)

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

    ax.plot(checkpoints, mlp_rank_0, "o-", color="tab:red",
            label="MLP layer 0", linewidth=1.5)
    ax.plot(checkpoints, mlp_rank_f, "o-", color="darkred",
            label=f"MLP layer {final_layer}", linewidth=1.5)
    ax.plot(checkpoints, ov_rank_0, "s--", color="tab:blue",
            label="OV layer 0", linewidth=1.5)
    ax.plot(checkpoints, ov_rank_f, "s--", color="darkblue",
            label=f"OV layer {final_layer}", linewidth=1.5)

    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title("Spectral concentration: OV vs MLP")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ov_timing_olmo_1b.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention OV vs MLP alignment timing (OLMo-2-1B)")
    parser.add_argument("--output_dir", default="results/ov_timing_olmo")
    args = parser.parse_args()

    results = run_experiment(args.output_dir)
    plot_results(results, args.output_dir)

    print("\nDone.")
