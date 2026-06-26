"""
Pairwise Subspace Overlap Across Training (TinyLlama-1.1B)
==========================================================

Computes the top-k subspace overlap between ALL pairs of layers at each
checkpoint, producing an n_layers x n_layers overlap matrix per step.

This extends the adjacent-layer analysis to capture long-range structure:
do distant layers share subspace alignment, and how does this evolve?

Output format matches the existing pairwise_overlap_olmo_1b.json and
pairwise_overlap_1b.json (Pythia) for direct comparison.

Architecture: TinyLlama uses SwiGLU (gate_proj, up_proj, down_proj).
The composed product W_down @ W_up is used as the analysis target.

Checkpoints: 50000, 240000, 480000, 715000, 955000, 1195000, 1431000

Usage:
    python experiment_pairwise_overlap_tinyllama.py

PATCH (read/write side): keeps the original symmetric overlap_matrix (as
  overlap_matrix and UU) and adds, per checkpoint, UU/VV/UV matrices:
      UU : symmetric, (i,j) = left_i  vs left_j   ("write vs write", == original)
      VV : symmetric, (i,j) = right_i vs right_j  ("read vs read")
      UV : full,      (i,j) = left_i  vs right_j  (directed: i writes -> j reads)
  UV is intentionally non-symmetric (UV[i][j] vs UV[j][i] encode the two flow
  directions). Convention verified numerically (composed @ V[:,i] = S[i]*U[:,i]).
  TinyLlama is SwiGLU, so composed = W_down @ W_up still excludes the gate.
PATCH (disk): each checkpoint repo is deleted from the HF cache once its data has
  been gathered (default on; --keep_checkpoints to retain).
"""

import json
import os

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D_MODEL = 2048
D_FF = 5632
N_LAYERS = 22
K = 10

CHECKPOINT_REPOS = {
    50000: "PY007/TinyLlama-1.1B-step-50K-105b",
    240000: "PY007/TinyLlama-1.1B-intermediate-step-240k-503b",
    480000: "PY007/TinyLlama-1.1B-intermediate-step-480k-1T",
    715000: "PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T",
    955000: "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T",
    1195000: "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
    1431000: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}


# ---------------------------------------------------------------------------
# Subspace overlap
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between two subspaces."""
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return cosines


def mean_subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """Mean cosine of principal angles between two k-dim subspaces."""
    cosines = principal_angles_cosines(U1, U2)
    return float(np.mean(cosines))


# ---------------------------------------------------------------------------
# Model loading and SVD extraction
# ---------------------------------------------------------------------------

def load_model_at_checkpoint(step: int):
    repo = CHECKPOINT_REPOS[step]
    print(f"  Loading {repo}...")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    # Each checkpoint is its own repo; revision is the default branch.
    return model, repo, "main"


def _ref_matches(rev, revision):
    """True if a cached revision corresponds to the loaded ref or commit."""
    if revision == rev.commit_hash:
        return True
    return any(r == revision or r.endswith("/" + revision) for r in rev.refs)


def free_checkpoint(repo_id, revision, cache_dir=None):
    """Delete a single downloaded checkpoint from the HF cache.

    Frees disk once a checkpoint's data has been gathered. Best-effort and safe:
    matches the exact repo_id and the ref/commit loaded, deletes only those, and
    never raises into the sweep. A miss is a no-op.
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


def get_layer_svd_uv(model, layer_idx: int, k: int):
    """Top-k left (write) and right (read) singular vectors of W_down @ W_up.

    Returns (U_topk, V_topk), each (d_model, k). TinyLlama is SwiGLU; this
    composed product excludes the gate (gate_proj), matching the other static
    measurements.
    """
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()
    W_down = mlp.down_proj.weight.detach().float()

    composed = (W_down @ W_up).cpu().numpy()
    U, S, Vt = np.linalg.svd(composed, full_matrices=False)
    return U[:, :k], Vt[:k].T


def _symmetric_matrix(blocks):
    """Symmetric NxN matrix of mean principal-angle cosines, diagonal = 1."""
    n = len(blocks)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = mean_subspace_overlap(blocks[i], blocks[j])
            M[i, j] = v
            M[j, i] = v
    return M.tolist()


def _directed_matrix(row_blocks, col_blocks):
    """Full (non-symmetric) NxN: entry [i, j] = overlap(row_blocks[i], col_blocks[j])."""
    n = len(row_blocks)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = mean_subspace_overlap(row_blocks[i], col_blocks[j])
    return M.tolist()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str, free_checkpoints: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pairwise_overlap_tinyllama_1b.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        results = {
            "model": "tinyllama-1.1b",
            "model_family": "tinyllama",
            "n_layers": N_LAYERS,
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "k": K,
            "checkpoint_repos": {str(s): r for s, r in CHECKPOINT_REPOS.items()},
            "checkpoints": {},
        }

    steps = sorted(CHECKPOINT_REPOS.keys())

    for step in steps:
        existing = results["checkpoints"].get(str(step))
        if existing is not None and all(key in existing for key in ("UU", "VV", "UV")):
            print(f"\n  Skipping step {step} (already computed, all relations present)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Step {step}")
        print(f"{'=' * 60}")

        model, repo, revision = load_model_at_checkpoint(step)

        # Extract top-k left (write) and right (read) singular vectors per layer
        print(f"  Computing SVDs for {N_LAYERS} layers...")
        layer_U, layer_V = [], []
        for li in range(N_LAYERS):
            U_k, V_k = get_layer_svd_uv(model, li, K)
            layer_U.append(U_k)
            layer_V.append(V_k)

        # Pairwise overlap matrices for all block relations
        print(f"  Computing pairwise overlaps (UU, VV, UV)...")
        UU = _symmetric_matrix(layer_U)
        VV = _symmetric_matrix(layer_V)
        UV = _directed_matrix(layer_U, layer_V)
        overlap_matrix = np.array(UU)  # summary stats on left-vs-left, as before

        # Print summary (on UU, matching the original metric)
        n_above = int(np.sum(overlap_matrix[np.triu_indices(N_LAYERS, k=1)] > 0.15))
        n_total = N_LAYERS * (N_LAYERS - 1) // 2
        mean_d1 = float(np.mean([overlap_matrix[i, i + 1] for i in range(N_LAYERS - 1)]))
        mean_d3 = float(np.mean([overlap_matrix[i, i + 3] for i in range(N_LAYERS - 3)]))
        mean_d5 = float(np.mean([overlap_matrix[i, i + 5] for i in range(N_LAYERS - 5)]))

        print(f"  Pairs > 0.15: {n_above}/{n_total} ({100 * n_above / n_total:.0f}%)")
        print(f"  Mean d=1: {mean_d1:.3f}  d=3: {mean_d3:.3f}  d=5: {mean_d5:.3f}")

        results["checkpoints"][str(step)] = {
            "overlap_matrix": UU,  # backward-compatible alias (left-vs-left)
            "UU": UU,
            "VV": VV,
            "UV": UV,
        }

        del model, layer_U, layer_V
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if free_checkpoints:
            free_checkpoint(repo, revision, CACHE_DIR)

        # Save after each checkpoint
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

    print(f"\nDone. Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pairwise Subspace Overlap (TinyLlama-1.1B)")
    parser.add_argument("--output_dir", default="results/pairwise_overlap")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep downloaded checkpoints in the HF cache. Default: "
                             "delete each checkpoint from the cache once its data has "
                             "been gathered, to save disk during long sweeps.")
    args = parser.parse_args()

    run_experiment(args.output_dir, free_checkpoints=not args.keep_checkpoints)
