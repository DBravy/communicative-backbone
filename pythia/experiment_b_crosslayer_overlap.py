"""
Experiment B: Cross-Layer Subspace Coherence (Bulk vs Tail)
============================================================

Tests whether training selectively aligns the *structural* (top-k) singular
subspaces across layers while leaving the *flexible* (tail-k) subspaces
incoherent. This is the missing link between:
  - per-layer alignment (Experiment A / alignment tracking): local rank compression
  - global trajectory smoothness (Experiment C / DCT): network-wide manifold

The prediction: top-k subspaces of composed MLP products become more coherent
across adjacent layers during training, while bottom-k subspaces remain
near-random. This would show that training builds a network-wide manifold by
aligning local structural directions.

Method:
  At each checkpoint, for each pair of adjacent layers (l, l+1):
    1. Compute SVD of composed W_down @ W_up at each layer
    2. Extract top-k left singular vectors (structural subspace)
    3. Extract bottom-k left singular vectors (flexible subspace)
    4. Compute principal angles between the two layers' subspaces
    5. Report mean cosine of principal angles (1 = identical, 0 = orthogonal)

Models: Pythia-70M, Pythia-410M
Checkpoints: 0, 128, 512, 2000, 8000, 32000, 64000, 143000

Usage:
    python experiment_b_crosslayer_overlap.py [--models 70m 410m] [--device cuda]

PATCH (read/write side):
  The original recorded only top{k}/bot{k}: the mean top-/bottom-k overlap of the
  LEFT singular vectors (U) between layers. This patch keeps those keys unchanged
  and adds, per pair, the full set of four block relations under top{k}_rel and
  bot{k}_rel:
      UiUj : left(early)  vs left(late)    (== legacy top/bot, "write vs write")
      ViVj : right(early) vs right(late)   ("read vs read")
      UiVj : left(early)  vs right(late)   ("earlier writes -> later reads")
      ViUj : right(early) vs left(late)    ("later writes -> earlier reads")
  Keys are named by singular-vector block, not by interpretation. The write/read
  reading was verified numerically (composed @ V[:,i] = S[i]*U[:,i], so V is the
  input/read side and U the output/write side). If that convention is ever found
  to be transposed, the labels swap but every recorded number stands.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

_ORICO_CACHE = "/Volumes/ORICO/huggingface_cache"
CACHE_DIR = _ORICO_CACHE if os.path.isdir("/Volumes/ORICO") else None

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

# Subspace dimensions to compare. We test several k values.
# These should match or overlap with TOP_K_VALUES in Experiment A.
K_VALUES = [5, 10, 20, 50]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Compute cosines of principal angles between two subspaces.
    
    Args:
        U1: (N, k1) orthonormal columns spanning subspace 1
        U2: (N, k2) orthonormal columns spanning subspace 2
    
    Returns:
        Array of min(k1, k2) cosines of principal angles, sorted descending.
        Values in [0, 1]. 1 = directions coincide, 0 = orthogonal.
    """
    # The singular values of U1^T @ U2 are the cosines of the principal angles
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    # Clip to [0, 1] for numerical safety
    cosines = np.clip(cosines, 0.0, 1.0)
    return cosines


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> dict:
    """Compute summary statistics of subspace overlap.
    
    Returns dict with:
        mean_cosine: mean of principal angle cosines (overall alignment)
        max_cosine: max cosine (best-aligned direction pair)
        min_cosine: min cosine (worst-aligned direction pair)
        grassmann_distance: geodesic distance on Grassmannian
        overlap_score: fraction of "well-aligned" directions (cosine > 0.5)
    """
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


def _band_slice(M: np.ndarray, k: int, band: str) -> np.ndarray:
    """Leading (top) or trailing (bot) k columns of an SVD factor."""
    if band == "top":
        return M[:, :k]
    if band == "bot":
        return M[:, -k:]
    raise ValueError(f"unknown band: {band!r}")


def relations_for_pair(svd_early: dict, svd_late: dict, k: int, band: str) -> dict:
    """All four cross-layer subspace relations between an earlier and a later layer.

    Each key names the exact singular-vector blocks compared; the first letter is
    the earlier layer, the second the later layer:
        UiUj : left(early)  vs left(late)
        ViVj : right(early) vs right(late)
        UiVj : left(early)  vs right(late)
        ViUj : right(early) vs left(late)

    Under the column-vector convention (out = composed @ r, composed = U S V^T,
    verified numerically), U are output/write directions and V input/read
    directions, so UiVj is the directed "earlier writes, later reads" channel and
    ViUj its reverse. Labels follow that convention; the numbers do not depend on
    it.
    """
    Ue = _band_slice(svd_early["U"], k, band)
    Ve = _band_slice(svd_early["V"], k, band)
    Ul = _band_slice(svd_late["U"], k, band)
    Vl = _band_slice(svd_late["V"], k, band)
    return {
        "UiUj": subspace_overlap(Ue, Ul),
        "ViVj": subspace_overlap(Ve, Vl),
        "UiVj": subspace_overlap(Ue, Vl),
        "ViUj": subspace_overlap(Ve, Ul),
    }


def random_subspace_baseline(d: int, k: int, n_trials: int = 100) -> dict:
    """Expected overlap between two random k-dimensional subspaces of R^d.
    
    This provides the null hypothesis: what overlap would we see if the
    subspaces were unrelated?
    """
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
    """Get full SVD of composed MLP product at given layer.
    
    Returns (U, S, Vt) where W_down @ W_up = U @ diag(S) @ Vt.
    U columns are left singular vectors in d_model space.
    """
    layer = model.gpt_neox.layers[layer_idx].mlp
    W_up = layer.dense_h_to_4h.weight.detach().float()    # (d_ff, d_model)
    W_down = layer.dense_4h_to_h.weight.detach().float()   # (d_model, d_ff)
    composed = (W_down @ W_up).cpu().numpy()                # (d_model, d_model)
    U, S, Vt = np.linalg.svd(composed, full_matrices=True)
    return U, S, Vt


def load_model_at_checkpoint(model_name: str, step: int):
    """Load a Pythia model at a specific training checkpoint."""
    from transformers import AutoModelForCausalLM

    revision = f"step{step}"
    print(f"  Loading {model_name} at {revision}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
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

    # Compute random baselines once
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
        "d_model": d_model,
        "n_layers": n_layers,
        "k_values": K_VALUES,
        "random_baselines": {str(k): v for k, v in baselines.items()},
        "checkpoints": {},
    }

    for step in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model = load_model_at_checkpoint(model_name, step)

        # Extract SVDs for all layers
        layer_svds = []
        for li in range(n_layers):
            U, S, Vt = get_layer_svd(model, li)
            # Keep both factors: U (left/write) and V = Vt.T (right/read).
            # Memory roughly doubles vs the original (two d x d factors per layer).
            layer_svds.append({"U": U, "V": Vt.T, "S": S})

        step_results = {"adjacent_pairs": {}, "non_adjacent": {}}

        # --- Adjacent layer comparisons ---
        for li in range(n_layers - 1):
            pair_key = f"{li}_{li+1}"
            pair_data = {}

            for k in K_VALUES:
                # All four block relations, for the top-k and bottom-k bands.
                top_rel = relations_for_pair(layer_svds[li], layer_svds[li + 1], k, "top")
                bot_rel = relations_for_pair(layer_svds[li], layer_svds[li + 1], k, "bot")

                # Legacy keys preserved exactly (UiUj == original left-vs-left
                # overlap), so existing figures reproduce unchanged.
                pair_data[f"top{k}"] = top_rel["UiUj"]
                pair_data[f"bot{k}"] = bot_rel["UiUj"]

                # Full relation set (UiUj, ViVj, UiVj, ViUj).
                pair_data[f"top{k}_rel"] = top_rel
                pair_data[f"bot{k}_rel"] = bot_rel

                print(f"    Layers {li}-{li+1}, k={k:2d}: "
                      f"UiUj={top_rel['UiUj']['mean_cosine']:.4f}  "
                      f"ViVj={top_rel['ViVj']['mean_cosine']:.4f}  "
                      f"UiVj={top_rel['UiVj']['mean_cosine']:.4f}  "
                      f"ViUj={top_rel['ViUj']['mean_cosine']:.4f}  "
                      f"(rand={baselines[k]['mean_cosine_mean']:.4f})")

            step_results["adjacent_pairs"][pair_key] = pair_data

        # --- Non-adjacent comparisons (first-to-last, first-to-mid) ---
        # These test whether the structural subspace is globally coherent,
        # not just locally (between neighbors).
        global_pairs = [
            (0, n_layers - 1, "first_last"),
            (0, n_layers // 2, "first_mid"),
            (n_layers // 2, n_layers - 1, "mid_last"),
        ]

        for l1, l2, label in global_pairs:
            # global_pairs are all ordered l1 < l2 (earlier, later).
            pair_data = {}
            for k in K_VALUES:
                top_rel = relations_for_pair(layer_svds[l1], layer_svds[l2], k, "top")
                bot_rel = relations_for_pair(layer_svds[l1], layer_svds[l2], k, "bot")

                pair_data[f"top{k}"] = top_rel["UiUj"]
                pair_data[f"bot{k}"] = bot_rel["UiUj"]
                pair_data[f"top{k}_rel"] = top_rel
                pair_data[f"bot{k}_rel"] = bot_rel

            step_results["non_adjacent"][label] = pair_data
            r10 = pair_data["top10_rel"]
            print(f"    Global {label} (layers {l1}-{l2}), k=10: "
                  f"UiUj={r10['UiUj']['mean_cosine']:.4f}  "
                  f"ViVj={r10['ViVj']['mean_cosine']:.4f}  "
                  f"UiVj={r10['UiVj']['mean_cosine']:.4f}  "
                  f"ViUj={r10['ViUj']['mean_cosine']:.4f}")

        results["checkpoints"][str(step)] = step_results

        del model, layer_svds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"crosslayer_overlap_{model_key}.json")
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

    # We'll focus on k=10 for the main plots
    k = 10

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Layer Subspace Coherence: Pythia-{model_key} (k={k})",
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
               label=f'Random baseline', alpha=0.7)
    ax.fill_between(
        checkpoints,
        bl["mean_cosine_mean"] - 2 * bl["mean_cosine_std"],
        bl["mean_cosine_mean"] + 2 * bl["mean_cosine_std"],
        color='gray', alpha=0.15
    )
    ax.set_xscale('symlog', linthresh=100)
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
    ax.set_xscale('symlog', linthresh=100)
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
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([pk.replace("_", "-") for pk in pair_keys],
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
    plot_path = os.path.join(output_dir, f"crosslayer_overlap_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_relations(results: dict, output_dir: str):
    """Plot all four cross-layer relations (adjacent-pair mean) across training.

    Purely descriptive: every relation is drawn against the random baseline with
    none privileged, so you can see whether the directed channels (UiVj, ViUj)
    track, lead, or diverge from the left-vs-left overlap (UiUj) the original
    figures were built on. The right panel shows the directed asymmetry
    UiVj - ViUj; a value far from zero indicates a genuine direction to the flow.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping relation plots.")
        return

    model_key = results["model"]
    checkpoints = sorted(int(s) for s in results["checkpoints"].keys())
    baselines = results["random_baselines"]
    k = 10
    rel_keys = ["UiUj", "ViVj", "UiVj", "ViUj"]
    colors = {"UiUj": "tab:red", "ViVj": "tab:green",
              "UiVj": "tab:purple", "ViUj": "tab:orange"}

    means = {rk: [] for rk in rel_keys}
    for step in checkpoints:
        pairs = results["checkpoints"][str(step)]["adjacent_pairs"]
        for rk in rel_keys:
            vals = [pairs[pk][f"top{k}_rel"][rk]["mean_cosine"] for pk in pairs]
            means[rk].append(float(np.mean(vals)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cross-layer relations: Pythia-{model_key} "
                 f"(top-{k}, adjacent-pair mean)", fontsize=13)

    for rk in rel_keys:
        ax1.plot(checkpoints, means[rk], "o-", color=colors[rk], label=rk)
    bl = baselines[str(k)]
    ax1.axhline(bl["mean_cosine_mean"], color="gray", linestyle="--",
                alpha=0.7, label="random")
    ax1.fill_between(checkpoints,
                     bl["mean_cosine_mean"] - 2 * bl["mean_cosine_std"],
                     bl["mean_cosine_mean"] + 2 * bl["mean_cosine_std"],
                     color="gray", alpha=0.15)
    ax1.set_xscale("symlog", linthresh=100)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean cosine (adjacent layers)")
    ax1.set_title("All four block relations")
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    asym = [means["UiVj"][i] - means["ViUj"][i] for i in range(len(checkpoints))]
    ax2.plot(checkpoints, asym, "o-", color="black")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xscale("symlog", linthresh=100)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("UiVj - ViUj (mean cosine)")
    ax2.set_title("Directed asymmetry (>0: earlier-writes / later-reads stronger)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"crosslayer_relations_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Relation plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment B: Cross-layer subspace coherence")
    parser.add_argument("--models", nargs="+", default=["70m", "410m"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output_dir", default="results/experiment_b")
    args = parser.parse_args()

    for model_key in args.models:
        results = run_experiment(model_key, args.output_dir)
        plot_results(results, args.output_dir)
        plot_relations(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
