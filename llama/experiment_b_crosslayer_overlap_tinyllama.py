"""
Experiment B (TinyLlama): Cross-Layer Subspace Coherence (Bulk vs Tail)
========================================================================

TinyLlama port of experiment_b_crosslayer_overlap_olmo.py.

Tests whether training selectively aligns the *structural* (top-k) singular
subspaces across layers while leaving the *flexible* (tail-k) subspaces
incoherent.

Architecture note:
    TinyLlama uses SwiGLU (gate_proj, up_proj, down_proj), identical MLP
    structure to OLMo-2. The composed product W_down @ W_up is the primary
    analysis target. Key architectural differences from OLMo-2-1B:
      - 22 layers (vs OLMo's 16)
      - d_ff = 5632 (vs OLMo's 8192)
      - d_model = 2048 (same)
    The extra depth is the critical variable: if the coherence valley
    location scales sensibly with network depth (e.g., the valley appears
    proportionally later in TinyLlama's 22-layer stack than in OLMo's 16),
    that is a strong structural prediction confirmed.

Critical questions:
    1. Does TinyLlama show the same pattern as late-training OLMo: moderate
       coherence at early boundaries (maintained by gate sparsity), declining
       coherence at mid-to-late boundaries, and a transition zone where
       coherence is weakest?
    2. Does the valley location scale sensibly with network depth (22 layers
       vs OLMo's 16)?

Model: TinyLlama-1.1B
Checkpoints: 50K, 240K, 480K, 715K, 955K, 1195K, 1431K steps
    (published as separate HuggingFace repos)

Usage:
    python experiment_b_crosslayer_overlap_tinyllama.py

PATCH (read/write side): keeps the original top{k}/bot{k} (LEFT-vs-LEFT overlap)
  and adds, per pair, top{k}_rel and bot{k}_rel with all four block relations:
      UiUj : left(early)  vs left(late)    (== legacy top/bot, "write vs write")
      ViVj : right(early) vs right(late)   ("read vs read")
      UiVj : left(early)  vs right(late)   ("earlier writes -> later reads")
      ViUj : right(early) vs left(late)    ("later writes -> earlier reads")
  Convention verified numerically (composed @ V[:,i] = S[i]*U[:,i]). TinyLlama is
  SwiGLU, so composed = W_down @ W_up still excludes the gate, exactly as before.
PATCH (disk): each checkpoint repo is deleted from the HF cache once its data has
  been gathered (default on; --keep_checkpoints to retain).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TinyLlama architecture constants
D_MODEL = 2048
D_FF = 5632
N_LAYERS = 22

K_VALUES = [5, 10, 20, 50]

# Checkpoints as separate HF repos
CHECKPOINT_REPOS = [
    (50_000,    "105B",  "PY007/TinyLlama-1.1B-step-50K-105b"),
    (240_000,   "503B",  "PY007/TinyLlama-1.1B-intermediate-step-240k-503b"),
    (480_000,   "1T",    "PY007/TinyLlama-1.1B-intermediate-step-480k-1T"),
    (955_000,   "2T",    "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"),
    (1_195_000, "2.5T",  "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"),
    (1_431_000, "3T",    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
]


# ---------------------------------------------------------------------------
# Core computation (identical to OLMo/Pythia version)
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

    Keys name the blocks compared (first = earlier, second = later):
        UiUj : left(early)  vs left(late)
        ViVj : right(early) vs right(late)
        UiVj : left(early)  vs right(late)
        ViUj : right(early) vs left(late)

    Under the column-vector convention (out = composed @ r, composed = U S V^T,
    verified numerically), U are output/write and V input/read directions, so UiVj
    is the directed "earlier writes, later reads" channel and ViUj its reverse.
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


def load_model_at_checkpoint(repo_id: str):
    """Load a TinyLlama checkpoint from a HuggingFace repo.

    Returns (model, repo_id, revision) so the caller can free exactly this
    checkpoint once its data has been gathered. Each checkpoint is its own repo,
    so revision is the default branch ("main").
    """
    print(f"  Loading {repo_id}...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    return model, repo_id, "main"


def _ref_matches(rev, revision):
    """True if a cached revision corresponds to the loaded ref or commit."""
    if revision == rev.commit_hash:
        return True
    return any(r == revision or r.endswith("/" + revision) for r in rev.refs)


def free_checkpoint(repo_id, revision, cache_dir=None):
    """Delete a single downloaded checkpoint from the HF cache.

    Frees disk once a checkpoint's data has been gathered. Best-effort and safe:
    matches the exact repo_id and the ref/commit loaded, deletes only those, and
    never raises into the sweep. A miss is a no-op. Each TinyLlama checkpoint is a
    full separate repo (multiple GB), so these accumulate fast without cleanup.
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

def run_experiment(output_dir: str, free_checkpoints: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "crosslayer_overlap_tinyllama_1b.json")

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
            baselines[k] = random_subspace_baseline(D_MODEL, k, n_trials=200)
            print(f"  k={k}: random mean_cosine = "
                  f"{baselines[k]['mean_cosine_mean']:.4f} +/- "
                  f"{baselines[k]['mean_cosine_std']:.4f}")

        results = {
            "model": "tinyllama-1.1b",
            "model_family": "tinyllama",
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "k_values": K_VALUES,
            "random_baselines": {str(k): v for k, v in baselines.items()},
            "checkpoint_repos": {
                str(step): {"tokens": tok, "repo": repo}
                for step, tok, repo in CHECKPOINT_REPOS
            },
            "checkpoints": {},
        }

    for step, tokens, repo_id in CHECKPOINT_REPOS:
        step_key = str(step)
        existing = results["checkpoints"].get(step_key)
        if existing is not None and _has_relations(existing):
            print(f"\n  Skipping step {step:,} (already computed, relations present)")
            continue

        print(f"\n{'='*60}")
        print(f"  TinyLlama -- step {step:,} ({tokens} tokens)")
        print(f"{'='*60}")

        model, repo, revision = load_model_at_checkpoint(repo_id)

        # Extract SVDs for all layers
        layer_svds = []
        for li in range(N_LAYERS):
            U, S, Vt = get_layer_svd(model, li)
            # Keep both factors: U (left/write) and V = Vt.T (right/read).
            layer_svds.append({"U": U, "V": Vt.T, "S": S})

        step_results = {"adjacent_pairs": {}, "non_adjacent": {}}

        # --- Adjacent layer comparisons ---
        for li in range(N_LAYERS - 1):
            pair_key = f"{li}_{li+1}"
            pair_data = {}

            for k in K_VALUES:
                # All four block relations, for the top-k and bottom-k bands.
                top_rel = relations_for_pair(layer_svds[li], layer_svds[li + 1], k, "top")
                bot_rel = relations_for_pair(layer_svds[li], layer_svds[li + 1], k, "bot")

                # Legacy keys preserved exactly (UiUj == original left-vs-left).
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
        # With 22 layers, we check first-last, first-mid, mid-last,
        # plus first-quarter and third-quarter boundaries
        global_pairs = [
            (0, N_LAYERS - 1, "first_last"),
            (0, N_LAYERS // 2, "first_mid"),
            (N_LAYERS // 2, N_LAYERS - 1, "mid_last"),
            (0, N_LAYERS // 4, "first_quarter"),
            (N_LAYERS // 4, N_LAYERS // 2, "second_quarter"),
            (N_LAYERS // 2, 3 * N_LAYERS // 4, "third_quarter"),
            (3 * N_LAYERS // 4, N_LAYERS - 1, "fourth_quarter"),
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

        results["checkpoints"][step_key] = step_results

        del model, layer_svds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if free_checkpoints:
            free_checkpoint(repo, revision, CACHE_DIR)

        # Save after each checkpoint
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to {out_path}")

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

    n_layers = results["n_layers"]
    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
    baselines = results["random_baselines"]

    k = 10

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Layer Subspace Coherence: TinyLlama-1.1B (k={k})",
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
    ax.set_xscale('symlog', linthresh=100000)
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
    ax.set_xscale('symlog', linthresh=100000)
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
    # Label every Nth pair for readability
    n_pairs = len(pair_keys)
    tick_stride = max(1, n_pairs // 10)
    ax.set_xticks([i for i in range(0, n_pairs, tick_stride)])
    ax.set_xticklabels([pair_keys[i].replace("_", "-")
                        for i in range(0, n_pairs, tick_stride)],
                       rotation=45, fontsize=7)
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Mean cosine")
    ax.set_title(f"Per-pair overlap (step {checkpoints[-1]:,})")
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
    ax.set_title(f"Overlap by subspace size (step {checkpoints[-1]:,})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "crosslayer_overlap_tinyllama_1b.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_relations(results: dict, output_dir: str):
    """Plot all four cross-layer relations (adjacent-pair mean) across training.

    Purely descriptive: every relation is drawn against the random baseline with
    none privileged, so you can see whether the directed channels (UiVj, ViUj)
    track, lead, or diverge from the left-vs-left overlap (UiUj). The right panel
    shows the directed asymmetry UiVj - ViUj. For this SwiGLU model, watch whether
    UU falls while VV or UV hold (relocation) versus all of them dissolving.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping relation plots.")
        return

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
    fig.suptitle(f"Cross-layer relations: TinyLlama-1.1B "
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
    ax1.set_xscale("symlog", linthresh=100000)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean cosine (adjacent layers)")
    ax1.set_title("All four block relations")
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    asym = [means["UiVj"][i] - means["ViUj"][i] for i in range(len(checkpoints))]
    ax2.plot(checkpoints, asym, "o-", color="black")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xscale("symlog", linthresh=100000)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("UiVj - ViUj (mean cosine)")
    ax2.set_title("Directed asymmetry (>0: earlier-writes / later-reads stronger)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "crosslayer_relations_tinyllama_1b.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Relation plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment B (TinyLlama): Cross-layer subspace coherence")
    parser.add_argument("--output_dir", default="results/experiment_b_tinyllama")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep downloaded checkpoints in the HF cache. Default: "
                             "delete each checkpoint from the cache once its data has "
                             "been gathered, to save disk during long sweeps.")
    args = parser.parse_args()

    results = run_experiment(args.output_dir,
                             free_checkpoints=not args.keep_checkpoints)
    plot_results(results, args.output_dir)
    plot_relations(results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
