"""
Compute full pairwise subspace overlap matrices (OLMo).
========================================================

OLMo port of compute_pairwise_overlap.py (originally for Pythia).

For each checkpoint, computes the NxN matrix where entry (i, j) is the
top-k mean cosine overlap between layers i and j.

Models: OLMo-1B, OLMo-7B (v1, with intermediate checkpoints)
Checkpoints: sampled across training

Checkpoint loading:
    Requires: pip install ai2-olmo
    Models: allenai/OLMo-1B, allenai/OLMo-7B
    Revisions: "step{N}-tokens{T}B"

Usage:
    python compute_pairwise_overlap_olmo.py [--models 1b 7b] [--k 10]

PATCH (read/write side):
  The original stored a single symmetric overlap_matrix: the top-k overlap of the
  LEFT singular vectors (U) between every layer pair. This patch keeps that matrix
  (as overlap_matrix and as UU) and adds two more per checkpoint:
      UU : symmetric, (i,j) = left_i  vs left_j   ("write vs write", == original)
      VV : symmetric, (i,j) = right_i vs right_j  ("read vs read")
      UV : full,      (i,j) = left_i  vs right_j  (directed: i writes -> j reads)
  UV is intentionally NON-symmetric: UV[i][j] and UV[j][i] encode the two flow
  directions, and VU would just be UV transposed. Keys are named by singular-vector
  block; the write/read reading was verified numerically (composed @ V[:,i] =
  S[i]*U[:,i]). The composed product is W_down @ W_up and still excludes the SwiGLU
  gate (gate_proj), exactly as before; that is a separate question from this patch.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

CHECKPOINTS_1B = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 100000, 500000, 1000000]

MODEL_CONFIGS = {
    "1b": {
        "name": "allenai/OLMo-2-0425-1B",
        "d_model": 2048, "d_ff": 8192, "n_layers": 16,
        "checkpoints": CHECKPOINTS_1B,
    },
}


def step_to_revision(step: int) -> str:
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


def principal_angles_cosines(U1, U2):
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def load_model_at_checkpoint(model_name, step):
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
    # Return repo + revision so the caller can free exactly this checkpoint.
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
    continues). A miss is a no-op. This is important here because the OLMo sweep
    pulls many large checkpoints across two repos (early-training and main).
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


def get_layer_svd(model, layer_idx):
    """SVD of composed MLP product W_down @ W_up.

    Returns (U, S, Vt) with composed = U @ diag(S) @ Vt.
    NOTE: for SwiGLU models (up_proj / down_proj) this composed product does NOT
    include the gate matrix (gate_proj). That omission matches the original
    measurement and is intentionally left unchanged here; the read/write patch is
    orthogonal to it.
    """
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()
    W_down = mlp.down_proj.weight.detach().float()

    composed = (W_down @ W_up).cpu().numpy()
    U, S, Vt = np.linalg.svd(composed, full_matrices=True)
    return U, S, Vt


def _symmetric_matrix(blocks):
    """Symmetric NxN matrix of mean principal-angle cosines, diagonal = 1."""
    n = len(blocks)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = float(np.mean(principal_angles_cosines(blocks[i], blocks[j])))
            M[i, j] = v
            M[j, i] = v
    return M.tolist()


def _directed_matrix(row_blocks, col_blocks):
    """Full (non-symmetric) NxN: entry [i, j] = overlap(row_blocks[i], col_blocks[j])."""
    n = len(row_blocks)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = float(np.mean(principal_angles_cosines(row_blocks[i], col_blocks[j])))
    return M.tolist()


def compute_pairwise_matrices(model, n_layers, k):
    """Compute pairwise top-k overlap matrices for all block relations.

    Returns dict of NxN matrices (as nested lists):
        UU : symmetric, (i,j) = left_i  vs left_j   (== original overlap_matrix)
        VV : symmetric, (i,j) = right_i vs right_j
        UV : full,      (i,j) = left_i  vs right_j

    Keys are by singular-vector block. Under the column-vector convention
    (composed = U S V^T, out = composed @ r, verified numerically), U are
    output/write and V input/read directions, so UV[i][j] is the directed
    "layer i writes, layer j reads" channel and UV[j][i] its reverse. VU is just
    UV transposed and is not stored separately.
    """
    layer_Us, layer_Vs = [], []
    for li in range(n_layers):
        U, _, Vt = get_layer_svd(model, li)
        layer_Us.append(U[:, :k])
        layer_Vs.append(Vt[:k].T)
        print(f"    SVD layer {li}/{n_layers}")

    return {
        "UU": _symmetric_matrix(layer_Us),
        "VV": _symmetric_matrix(layer_Vs),
        "UV": _directed_matrix(layer_Us, layer_Vs),
    }


def run(model_key, k, output_dir, free_checkpoints=True):
    cfg = MODEL_CONFIGS[model_key]
    checkpoints = cfg["checkpoints"]

    # Load existing results if available (to skip already-computed checkpoints)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"pairwise_overlap_olmo_{model_key}.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_path} "
              f"({len(results['checkpoints'])} checkpoints already computed)")
    else:
        results = {
            "model": model_key,
            "model_name": cfg["name"],
            "model_family": "olmo",
            "n_layers": cfg["n_layers"],
            "d_model": cfg["d_model"],
            "k": k,
            "checkpoints": {},
        }

    for step in checkpoints:
        existing = results["checkpoints"].get(str(step))
        if existing is not None and all(key in existing for key in ("UU", "VV", "UV")):
            print(f"\n  Skipping step {step} (already computed, all relations present)")
            continue

        print(f"\n  {model_key} step {step}")
        model, repo, revision = load_model_at_checkpoint(cfg["name"], step)
        mats = compute_pairwise_matrices(model, cfg["n_layers"], k)
        results["checkpoints"][str(step)] = {
            "overlap_matrix": mats["UU"],  # backward-compatible alias (left-vs-left)
            "UU": mats["UU"],
            "VV": mats["VV"],
            "UV": mats["UV"],
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if free_checkpoints:
            free_checkpoint(repo, revision, CACHE_DIR)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["1b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_dir", default="results/experiment_b_olmo")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep downloaded checkpoints in the HF cache. Default: "
                             "delete each checkpoint from the cache once its data has "
                             "been gathered, to save disk during long sweeps.")
    args = parser.parse_args()

    for m in args.models:
        run(m, args.k, args.output_dir, free_checkpoints=not args.keep_checkpoints)
    print("\nDone.")
