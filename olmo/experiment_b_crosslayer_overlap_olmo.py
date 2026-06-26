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
  reading was verified numerically (composed @ V[:,i] = S[i]*U[:,i]). OLMo is
  SwiGLU, so composed = W_down @ W_up still excludes the gate (gate_proj), exactly
  as before; that is a separate question from this patch.

PATCH (disk): each checkpoint is deleted from the HF cache once its data has been
  gathered (default on; use --keep_checkpoints to retain downloads).
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


def _has_relations(step_result: dict) -> bool:
    """True if a stored checkpoint already carries the four-relation data."""
    pairs = step_result.get("adjacent_pairs", {})
    if not pairs:
        return False
    any_pair = next(iter(pairs.values()))
    return any(key.endswith("_rel") for key in any_pair)


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
    """Load an OLMo 2 model at a specific training checkpoint.

    Returns (model, repo_id, revision) so the caller can free exactly the
    checkpoint that was downloaded once its data has been gathered.
    """
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
    return model, repo, revision


def _ref_matches(rev, revision):
    """True if a cached revision corresponds to the loaded ref or commit."""
    if revision == rev.commit_hash:
        return True
    return any(r == revision or r.endswith("/" + revision) for r in rev.refs)


def free_checkpoint(repo_id, revision, cache_dir=None):
    """Delete a single downloaded checkpoint (model revision) from the HF cache.

    Frees disk once a checkpoint's data has been gathered. Best-effort and safe:
    it matches the exact repo_id and the ref/commit that was loaded, deletes only
    those, and never raises into the sweep (a failure prints a warning and the run
    continues). A miss is a no-op. This matters here because the OLMo sweep pulls
    many large checkpoints across two repos (early-training and main).
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(cache_dir=cache_dir)
    except Exception as e:
        print(f"  [cache] skip free for {repo_id}@{revision}: cannot scan cache ({e})")
        return
    commit_hashes = [
        rev.commit_hash
        for repo in info.repos if repo.repo_id == repo_id
        for rev in repo.revisions if _ref_matches(rev, revision)
    ]
    if not commit_hashes:
        print(f"  [cache] nothing to free for {repo_id}@{revision} (not in cache)")
        return
    try:
        strategy = info.delete_revisions(*commit_hashes)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        print(f"  [cache] freed {freed}: deleted {repo_id}@{revision}")
    except Exception as e:
        print(f"  [cache] could not delete {repo_id}@{revision}: {e}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(model_key: str, output_dir: str, free_checkpoints: bool = True):
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
        existing = results["checkpoints"].get(str(step))
        if existing is not None and _has_relations(existing):
            print(f"\n  Skipping step {step} (already computed, relations present)")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model, repo, revision = load_model_at_checkpoint(model_name, step)

        # Extract SVDs for all layers
        layer_svds = []
        for li in range(n_layers):
            U, S, Vt = get_layer_svd(model, li)
            # Keep both factors: U (left/write) and V = Vt.T (right/read).
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

        # --- Non-adjacent comparisons ---
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
        if free_checkpoints:
            free_checkpoint(repo, revision, CACHE_DIR)

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


def plot_relations(results: dict, output_dir: str):
    """Plot all four cross-layer relations (adjacent-pair mean) across training.

    Purely descriptive: every relation is drawn against the random baseline with
    none privileged, so you can see whether the directed channels (UiVj, ViUj)
    track, lead, or diverge from the left-vs-left overlap (UiUj) the original
    figures were built on. The right panel shows the directed asymmetry
    UiVj - ViUj; a value far from zero indicates a genuine direction to the flow.
    For the SwiGLU models this is the place to watch: if UU falls while VV or UV
    stay elevated, the alignment is relocating rather than dissolving.
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
    fig.suptitle(f"Cross-layer relations: OLMo-{model_key.upper()} "
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
    ax1.set_xscale("symlog", linthresh=1000)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean cosine (adjacent layers)")
    ax1.set_title("All four block relations")
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    asym = [means["UiVj"][i] - means["ViUj"][i] for i in range(len(checkpoints))]
    ax2.plot(checkpoints, asym, "o-", color="black")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xscale("symlog", linthresh=1000)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("UiVj - ViUj (mean cosine)")
    ax2.set_title("Directed asymmetry (>0: earlier-writes / later-reads stronger)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"crosslayer_relations_olmo_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Relation plot saved to {plot_path}")
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
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep downloaded checkpoints in the HF cache. Default: "
                             "delete each checkpoint from the cache once its data has "
                             "been gathered, to save disk during long sweeps.")
    args = parser.parse_args()

    for model_key in args.models:
        results = run_experiment(model_key, args.output_dir,
                                 free_checkpoints=not args.keep_checkpoints)
        plot_results(results, args.output_dir)
        plot_relations(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
