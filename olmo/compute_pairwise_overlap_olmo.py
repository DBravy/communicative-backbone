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

CHECKPOINTS_1B = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

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
    return model


def get_layer_svd(model, layer_idx):
    """SVD of composed MLP product W_down @ W_up."""
    mlp = model.model.layers[layer_idx].mlp
    W_up = mlp.up_proj.weight.detach().float()
    W_down = mlp.down_proj.weight.detach().float()

    composed = (W_down @ W_up).cpu().numpy()
    U, S, _ = np.linalg.svd(composed, full_matrices=True)
    return U, S


def compute_pairwise_matrix(model, n_layers, k):
    """Compute NxN top-k mean cosine overlap matrix."""
    layer_Us = []
    for li in range(n_layers):
        U, _ = get_layer_svd(model, li)
        layer_Us.append(U[:, :k])
        print(f"    SVD layer {li}/{n_layers}")

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
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n  {model_key} step {step}")
        model = load_model_at_checkpoint(cfg["name"], step)
        matrix = compute_pairwise_matrix(model, cfg["n_layers"], k)
        results["checkpoints"][str(step)] = {"overlap_matrix": matrix}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["1b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_dir", default="results/experiment_b_olmo")
    args = parser.parse_args()

    for m in args.models:
        run(m, args.k, args.output_dir)
    print("\nDone.")
