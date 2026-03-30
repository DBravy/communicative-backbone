"""
Compute per-layer MLP cross-covariance spectrum for Pythia models.
==================================================================

For each layer, collects MLP input/output pairs during a forward pass,
computes the cross-covariance matrix, and takes the SVD to get
effective rank and total variance.

Usage:
    python compute_crosscov_pythia.py [--models 410m 1b 1.4b] [--n_tokens 500] [--device mps]
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = os.environ.get("HF_HOME", None)

MODEL_CONFIGS = {
    "410m":  {"name": "EleutherAI/pythia-410m",  "d_model": 1024, "n_layers": 24, "d_ff": 4096},
    "1b":    {"name": "EleutherAI/pythia-1b",    "d_model": 2048, "n_layers": 16, "d_ff": 8192},
    "1.4b":  {"name": "EleutherAI/pythia-1.4b",  "d_model": 2048, "n_layers": 24, "d_ff": 8192},
}

SAMPLE_TEXTS = [
    "The development of quantum computing has accelerated in recent years, with several companies demonstrating systems that can perform calculations beyond the reach of classical supercomputers. These advances raise important questions about cryptography and security.",
    "In evolutionary biology, the concept of fitness landscapes provides a powerful framework for understanding how populations navigate the space of possible genotypes. Peaks in this landscape correspond to well-adapted organisms.",
    "The global supply chain disruptions of the early 2020s revealed deep vulnerabilities in just-in-time manufacturing. Companies that had optimized for efficiency found themselves unable to cope with sudden shifts in demand and supply.",
    "Neural network training involves navigating a high-dimensional loss landscape. The geometry of this landscape, including the presence of saddle points and flat minima, has significant implications for generalization.",
    "Archaeological evidence from the Indus Valley civilization suggests a remarkably sophisticated urban planning system, with standardized brick sizes, elaborate drainage systems, and evidence of long-distance trade networks.",
    "The interaction between the gut microbiome and the central nervous system, often called the gut-brain axis, has emerged as a major area of research in both neuroscience and gastroenterology.",
    "Monetary policy in the post-2008 era has been characterized by historically low interest rates and unconventional tools such as quantitative easing. The long-term consequences of these policies remain debated.",
    "The study of turbulence remains one of the great unsolved problems in classical physics. Despite centuries of investigation, we lack a complete theoretical framework for predicting turbulent flows.",
    "Recent advances in single-cell RNA sequencing have transformed our understanding of cellular heterogeneity within tissues previously thought to be homogeneous.",
    "The philosophy of language has grappled with the relationship between meaning and reference since Frege's distinction between sense and reference in the late nineteenth century.",
]


def load_model(model_name, device, dtype, step=None):
    """Load a Pythia model, optionally at a specific training step."""
    revision = f"step{step}" if step is not None else None
    step_str = f" at step{step}" if step else " (final)"
    print(f"Loading {model_name}{step_str} on {device} at {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
    )
    return model, tokenizer


def collect_mlp_io(model, tokenizer, texts, device, n_tokens, n_layers):
    """
    Run forward passes and collect MLP input/output pairs at each layer.
    Returns dict: layer_idx -> (inputs, outputs), each shape (n_collected, d_model).
    """
    mlp_inputs = {i: [] for i in range(n_layers)}
    mlp_outputs = {i: [] for i in range(n_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float().cpu()
            y = out.detach().float().cpu()
            x = x.reshape(-1, x.shape[-1])
            y = y.reshape(-1, y.shape[-1])
            mlp_inputs[layer_idx].append(x)
            mlp_outputs[layer_idx].append(y)
        return hook_fn

    # Pythia uses gpt_neox.layers[i].mlp
    for i in range(n_layers):
        h = model.gpt_neox.layers[i].mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    total_collected = 0
    for text in texts:
        if total_collected >= n_tokens:
            break
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            model(**tokens)
        seq_len = tokens["input_ids"].shape[1]
        total_collected += seq_len
        print(f"  Collected {total_collected} token positions so far...")

    for h in hooks:
        h.remove()

    result = {}
    for i in range(n_layers):
        inp = torch.cat(mlp_inputs[i], dim=0)[:n_tokens].numpy()
        out = torch.cat(mlp_outputs[i], dim=0)[:n_tokens].numpy()
        result[i] = (inp, out)
        print(f"  Layer {i}: collected {inp.shape[0]} positions, d={inp.shape[1]}")

    return result


def compute_crosscov_spectrum(inputs, outputs):
    """
    Compute the cross-covariance matrix between inputs and outputs,
    take SVD, return effective rank and total variance.
    """
    n = inputs.shape[0]

    inputs_c = inputs - inputs.mean(axis=0, keepdims=True)
    outputs_c = outputs - outputs.mean(axis=0, keepdims=True)

    C = (outputs_c.T @ inputs_c) / (n - 1)

    _, S, _ = np.linalg.svd(C, full_matrices=False)

    total_var = float(np.sum(S ** 2))

    s2 = S ** 2
    s2_norm = s2 / s2.sum()
    s2_norm = s2_norm[s2_norm > 1e-12]
    entropy = -np.sum(s2_norm * np.log(s2_norm))
    eff_rank = float(np.exp(entropy))

    top_sv = S[:20].tolist()

    return eff_rank, total_var, top_sv


def run(model_key, n_tokens, device, dtype, step=None):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg["n_layers"]

    model, tokenizer = load_model(cfg["name"], device, dtype, step=step)
    print(f"\nCollecting MLP input/output pairs ({n_tokens} token positions)...\n")
    mlp_io = collect_mlp_io(model, tokenizer, SAMPLE_TEXTS, device, n_tokens, n_layers)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\nComputing cross-covariance spectra...\n")
    step_label = str(step) if step is not None else "final"
    results = {
        "model": model_key,
        "model_name": cfg["name"],
        "checkpoint": step_label,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "d_model": cfg["d_model"],
        "layers": {},
    }

    for i in range(n_layers):
        inp, out = mlp_io[i]
        eff_rank, total_var, top_sv = compute_crosscov_spectrum(inp, out)
        results["layers"][str(i)] = {
            "effective_rank": round(eff_rank, 2),
            "total_variance": total_var,
            "top_20_singular_values": [round(s, 4) for s in top_sv],
        }
        print(f"  Layer {i:2d}: eff_rank={eff_rank:7.1f}  total_var={total_var:.3e}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/crosscov_pythia_{model_key}_{step_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: Pythia-{model_key} (step {step_label})")
    print("=" * 60)
    print(f"{'Layer':>6} {'Eff. Rank':>10} {'Total Var':>12}")
    print("-" * 60)
    for i in range(n_layers):
        d = results["layers"][str(i)]
        print(f"{i:>6} {d['effective_rank']:>10.1f} {d['total_variance']:>12.3e}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["410m", "1b", "1.4b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n_tokens", type=int, default=500)
    parser.add_argument("--step", type=int, default=None,
                        help="Training step (default: final checkpoint)")
    parser.add_argument("--device", default="auto",
                        help="Device: 'mps', 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--fp32", action="store_true",
                        help="Use float32 instead of float16")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype = torch.float32 if args.fp32 else torch.float16

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Token positions: {args.n_tokens}")

    for m in args.models:
        run(m, args.n_tokens, device, dtype, step=args.step)

    print("\nDone.")
