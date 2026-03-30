"""
Compute full pairwise subspace overlap matrices.
=================================================

Extends experiment_b: for each checkpoint, computes the NxN matrix where
entry (i, j) is the top-k mean cosine overlap between layers i and j.
Saves results to results/experiment_b/pairwise_overlap_{model}.json.

Usage:
    python compute_pairwise_overlap.py [--models 410m 1b 1.4b] [--k 10]
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
    print(f"  Loading {model_name} at step{step}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=f"step{step}",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    return model


def get_layer_svd(model, layer_idx):
    """SVD of composed MLP product W_down @ W_up."""
    layer = model.gpt_neox.layers[layer_idx].mlp
    W_up = layer.dense_h_to_4h.weight.detach().float()
    W_down = layer.dense_4h_to_h.weight.detach().float()
    composed = (W_down @ W_up).cpu().numpy()
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


def compute_pairwise_matrix(model, n_layers, k):
    """Compute NxN top-k mean cosine overlap matrix."""
    # Extract SVDs for all layers
    layer_Us = []
    for li in range(n_layers):
        U, _ = get_layer_svd(model, li)
        layer_Us.append(U[:, :k])
        print(f"    SVD layer {li}/{n_layers}")

    # Full pairwise overlap
    matrix = np.ones((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            cosines = principal_angles_cosines(layer_Us[i], layer_Us[j])
            val = float(np.mean(cosines))
            matrix[i, j] = val
            matrix[j, i] = val

    return matrix.tolist()


def run(model_key, k, output_dir):
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
        model = load_model_at_checkpoint(cfg["name"], step)
        matrix = compute_pairwise_matrix(model, cfg["n_layers"], k)
        results["checkpoints"][str(step)] = {"overlap_matrix": matrix}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    args = parser.parse_args()

    for m in args.models:
        run(m, args.k, args.output_dir)
    print("\nDone.")
