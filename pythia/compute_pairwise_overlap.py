"""
Compute full pairwise subspace overlap matrices.
=================================================

Extends experiment_b: for each checkpoint, computes the NxN matrix where
entry (i, j) is the top-k mean cosine overlap between layers i and j.
Saves results to results/experiment_b/pairwise_overlap_{model}.json.

Usage:
    python compute_pairwise_overlap.py [--models 410m 1b 1.4b] [--k 10]

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
  S[i]*U[:,i]). Pythia is GELU (two-matrix MLP), so composed = W_down @ W_up is the
  full linear map and there is no gate to exclude.

PATCH (disk): each checkpoint is deleted from the HF cache once its data has been
  gathered (default on; use --keep_checkpoints to retain downloads).
"""

import argparse
import json
import os

import numpy as np
import torch

CACHE_DIR = os.environ.get("HF_HOME", None)

CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]

MODEL_CONFIGS = {
    "410m":  {"name": "EleutherAI/pythia-410m",    "d_model": 1024, "n_layers": 24, "d_ff": 4096},
    "1b":    {"name": "EleutherAI/pythia-1b",      "d_model": 2048, "n_layers": 16, "d_ff": 8192},
    "1.4b":  {"name": "EleutherAI/pythia-1.4b",    "d_model": 2048, "n_layers": 24, "d_ff": 8192},
}


def principal_angles_cosines(U1, U2):
    """Cosines of principal angles between two subspaces."""
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def load_model_at_checkpoint(model_name, step):
    from transformers import AutoModelForCausalLM
    revision = f"step{step}"
    print(f"  Loading {model_name} at {revision}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    # Return repo + revision so the caller can free exactly this checkpoint.
    return model, model_name, revision


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
    continues). A miss is a no-op.
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

    Returns (U, S, Vt) with composed = U @ diag(S) @ Vt. Pythia is GELU
    (dense_h_to_4h / dense_4h_to_h), so this composed product is the full linear
    map of the MLP; there is no separate gate matrix to exclude.
    """
    layer = model.gpt_neox.layers[layer_idx].mlp
    W_up = layer.dense_h_to_4h.weight.detach().float()
    W_down = layer.dense_4h_to_h.weight.detach().float()
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
    results = {
        "model": model_key,
        "model_name": cfg["name"],
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "k": k,
        "checkpoints": {},
    }

    for step in CHECKPOINTS:
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

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"pairwise_overlap_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["410m", "1b", "1.4b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_dir", default="results/experiment_b")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep downloaded checkpoints in the HF cache. Default: "
                             "delete each checkpoint from the cache once its data has "
                             "been gathered, to save disk during long sweeps.")
    args = parser.parse_args()

    for m in args.models:
        run(m, args.k, args.output_dir, free_checkpoints=not args.keep_checkpoints)
    print("\nDone.")
