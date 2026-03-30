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
    return model


def get_layer_svd_top_k(model, layer_idx: int, k: int) -> np.ndarray:
    """
    Get top-k left singular vectors of composed MLP product W_down @ W_up.
    Returns U_topk of shape (d_model, k).
    """
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()
    W_down = mlp.down_proj.weight.detach().float()

    composed = (W_down @ W_up).cpu().numpy()
    U, S, Vt = np.linalg.svd(composed, full_matrices=False)
    return U[:, :k]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str):
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
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Step {step}")
        print(f"{'=' * 60}")

        model = load_model_at_checkpoint(step)

        # Extract top-k SVDs for all layers
        print(f"  Computing SVDs for {N_LAYERS} layers...")
        layer_U = []
        for li in range(N_LAYERS):
            U_k = get_layer_svd_top_k(model, li, K)
            layer_U.append(U_k)

        # Compute full pairwise overlap matrix
        print(f"  Computing pairwise overlaps ({N_LAYERS * (N_LAYERS - 1) // 2} pairs)...")
        overlap_matrix = np.zeros((N_LAYERS, N_LAYERS))

        for i in range(N_LAYERS):
            for j in range(N_LAYERS):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                elif j > i:
                    ov = mean_subspace_overlap(layer_U[i], layer_U[j])
                    overlap_matrix[i, j] = ov
                    overlap_matrix[j, i] = ov

        # Print summary
        n_above = np.sum(overlap_matrix[np.triu_indices(N_LAYERS, k=1)] > 0.15)
        n_total = N_LAYERS * (N_LAYERS - 1) // 2
        mean_d1 = np.mean([overlap_matrix[i, i + 1] for i in range(N_LAYERS - 1)])

        d3_pairs = [(i, i + 3) for i in range(N_LAYERS - 3)]
        mean_d3 = np.mean([overlap_matrix[i, j] for i, j in d3_pairs])

        d5_pairs = [(i, i + 5) for i in range(N_LAYERS - 5)]
        mean_d5 = np.mean([overlap_matrix[i, j] for i, j in d5_pairs])

        print(f"  Pairs > 0.15: {n_above}/{n_total} ({100 * n_above / n_total:.0f}%)")
        print(f"  Mean d=1: {mean_d1:.3f}  d=3: {mean_d3:.3f}  d=5: {mean_d5:.3f}")

        results["checkpoints"][str(step)] = {
            "overlap_matrix": overlap_matrix.tolist(),
        }

        del model, layer_U
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    args = parser.parse_args()

    run_experiment(args.output_dir)
