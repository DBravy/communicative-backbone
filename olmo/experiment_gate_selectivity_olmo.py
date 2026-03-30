"""
Gate Selectivity Across Training (OLMo-2-1B)
=============================================

Tracks how the SwiGLU gate's modulation behavior changes over training.

For each checkpoint, runs a batch of diverse text through the model,
captures the MLP input at each layer, and computes the gate activation
g(x) = SiLU(W_gate @ x). From these gate vectors, we derive:

1. Mean magnitude: average absolute gate value (should rise from init)
2. Sparsity: fraction of gate values below a threshold (does the gate
   learn to shut off dimensions?)
3. Kurtosis: peakedness of the gate distribution per input (high kurtosis
   means a few dimensions dominate)
4. Within-input variance: variance of gate values across hidden dims for
   a single token, averaged over tokens. Measures how selective the gate
   is for any given input.
5. Across-input variance: variance of gate values for a single hidden dim
   across tokens, averaged over dims. HIGH VALUES = the gate is making
   input-dependent decisions. This is the KEY metric: when this rises,
   the gate is "waking up" as a computational mechanism.
6. Effective rank: exponential of the entropy of the normalized squared
   gate activations across the input batch. Low rank means the gate
   concentrates modulation in a small number of patterns.

Checkpoints: 0, 1000, 3000, 5000, 10000, 100000, 1000000

Usage:
    python experiment_gate_selectivity_olmo.py [--n_samples 256]
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from scipy import stats as sp_stats

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

MODEL_NAME = "allenai/OLMo-2-0425-1B"
D_MODEL = 2048
D_FF = 8192
N_LAYERS = 16

CHECKPOINTS = [0, 1000, 3000, 5000, 10000, 100000, 1000000]

SPARSITY_THRESHOLDS = [0.01, 0.05, 0.1]

# ---------------------------------------------------------------------------
# Sample text for computing activations
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine and ushered in the era of antibiotics. Before this breakthrough, even minor infections could prove fatal, and surgical procedures carried enormous risk. Fleming noticed that a mold called Penicillium notatum had contaminated one of his petri dishes and was killing the surrounding bacteria. This accidental observation led to one of the most important medical advances in history.",
    "In quantum mechanics, the uncertainty principle states that certain pairs of physical properties cannot both be known to arbitrary precision simultaneously. The more precisely one property is measured, the less precisely the other can be controlled. This fundamental limit is not a statement about the inadequacy of measurement instruments, but rather a reflection of the intrinsic nature of quantum systems themselves.",
    "The Amazon rainforest spans approximately 5.5 million square kilometers across nine countries in South America. It contains roughly 10 percent of all species on Earth and produces about 20 percent of the world's oxygen. The forest plays a critical role in regulating global climate patterns by absorbing carbon dioxide and releasing water vapor through transpiration.",
    "Machine learning models trained on large datasets can exhibit unexpected emergent behaviors that were not explicitly programmed or anticipated by their designers. These capabilities often appear suddenly as model scale increases, rather than improving gradually. Understanding when and why emergence occurs remains one of the central open questions in artificial intelligence research today.",
    "The construction of the Panama Canal, completed in 1914, required the excavation of over 200 million cubic yards of earth and rock. The project employed tens of thousands of workers and took over a decade to complete. Engineering challenges included managing tropical diseases, designing a lock system to handle elevation changes, and controlling the unpredictable Chagres River during flood season.",
    "Photosynthesis converts light energy into chemical energy through a series of reactions that take place in the chloroplasts of plant cells. The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH. The Calvin cycle then uses these energy carriers to fix carbon dioxide into organic molecules that the plant uses for growth and energy.",
    "The Renaissance period in European history, spanning roughly from the 14th to the 17th century, marked a profound cultural transformation characterized by renewed interest in classical antiquity. Advances in art, science, and philosophy reshaped European intellectual life. Florence, under the patronage of the Medici family, became a particularly important center of artistic and intellectual activity during this era.",
    "Neural networks process information through layers of interconnected nodes, each applying a nonlinear transformation to its inputs. The universal approximation theorem establishes that sufficiently wide networks can represent any continuous function, but says nothing about whether gradient-based training will find such representations. The gap between representational capacity and learnability remains a fundamental tension in deep learning.",
    "The human genome contains approximately 3.2 billion base pairs of DNA, encoding roughly 20,000 protein-coding genes. However, protein-coding sequences account for only about 1.5 percent of the total genome. The remaining noncoding regions, once dismissed as junk DNA, are now understood to play critical roles in gene regulation, chromosome structure, and evolutionary adaptation.",
    "Ocean currents transport vast quantities of heat around the globe, fundamentally shaping regional climates and weather patterns. The Atlantic meridional overturning circulation, for example, carries warm water northward from the tropics, giving Western Europe a milder climate than its latitude would otherwise suggest. Disruptions to these circulation patterns have been linked to abrupt climate shifts in the geological record.",
]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def step_to_revision(step: int) -> str:
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


def load_model_at_checkpoint(step: int):
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else MODEL_NAME
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


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def get_sample_input_ids(tokenizer, seq_len=128):
    """Tokenize sample texts into chunks of seq_len."""
    all_ids = []
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

    sequences = []
    for i in range(0, len(all_ids) - seq_len + 1, seq_len):
        sequences.append(all_ids[i:i + seq_len])

    if not sequences:
        sequences.append(all_ids[:seq_len])

    return torch.tensor(sequences, dtype=torch.long)


def collect_gate_activations(model, input_ids, n_samples=256):
    """
    Run input through the model, capture MLP inputs at each layer,
    and compute gate activations g(x) = SiLU(W_gate @ x).

    Returns dict: layer_idx -> gate_values array of shape (n_samples, d_ff)
    """
    mlp_inputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            mlp_inputs[layer_idx] = args[0].detach().cpu()
        return hook_fn

    hooks = []
    for i in range(N_LAYERS):
        h = model.model.layers[i].mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx in range(input_ids.shape[0]):
            ids = input_ids[batch_idx:batch_idx + 1].to(device)
            model(ids)

            if batch_idx == 0:
                for li in range(N_LAYERS):
                    # Initialize storage
                    pass

    # Collect all mlp inputs across batches
    # Re-run more carefully, accumulating token-level inputs
    all_mlp_inputs = {i: [] for i in range(N_LAYERS)}

    for batch_idx in range(input_ids.shape[0]):
        ids = input_ids[batch_idx:batch_idx + 1].to(device)

        with torch.no_grad():
            model(ids)

        for li in range(N_LAYERS):
            # mlp_inputs[li] shape: (1, seq_len, d_model)
            all_mlp_inputs[li].append(mlp_inputs[li].squeeze(0))

    for h in hooks:
        h.remove()

    # Concatenate and subsample
    gate_activations = {}
    for li in range(N_LAYERS):
        x = torch.cat(all_mlp_inputs[li], dim=0)  # (total_tokens, d_model)
        if x.shape[0] > n_samples:
            indices = torch.randperm(x.shape[0])[:n_samples]
            x = x[indices]

        # Compute gate activations: SiLU(W_gate @ x)
        W_gate = model.model.layers[li].mlp.gate_proj.weight.detach().cpu().float()
        x_float = x.float()
        gate_pre = x_float @ W_gate.T  # (n_samples, d_ff)
        gate_vals = torch.nn.functional.silu(gate_pre)

        gate_activations[li] = gate_vals.numpy()
        print(f"    Layer {li}: collected {gate_vals.shape[0]} gate vectors")

    return gate_activations


# ---------------------------------------------------------------------------
# Gate statistics
# ---------------------------------------------------------------------------

def effective_rank(matrix: np.ndarray) -> float:
    """Effective rank via entropy of normalized squared singular values."""
    S = np.linalg.svd(matrix, compute_uv=False)
    S_sq = S ** 2
    total = S_sq.sum()
    if total < 1e-12:
        return 0.0
    p = S_sq / total
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def compute_gate_statistics(gate_vals: np.ndarray) -> dict:
    """
    Compute statistics of gate activations.
    gate_vals: (n_samples, d_ff)
    """
    n_samples, d_ff = gate_vals.shape
    abs_vals = np.abs(gate_vals)

    # 1. Mean magnitude
    mean_magnitude = float(np.mean(abs_vals))

    # 2. Sparsity at various thresholds
    sparsity = {}
    for thresh in SPARSITY_THRESHOLDS:
        frac_below = float(np.mean(abs_vals < thresh))
        sparsity[str(thresh)] = frac_below

    # 3. Kurtosis per input, averaged
    # Kurtosis of gate values across d_ff for each input
    per_input_kurtosis = []
    for i in range(n_samples):
        k = float(sp_stats.kurtosis(gate_vals[i], fisher=True))
        per_input_kurtosis.append(k)
    mean_kurtosis = float(np.mean(per_input_kurtosis))
    std_kurtosis = float(np.std(per_input_kurtosis))

    # 4. Within-input variance: var across d_ff per token, averaged
    within_input_var = float(np.mean(np.var(gate_vals, axis=1)))

    # 5. Across-input variance: var across tokens per dim, averaged
    # THIS IS THE KEY METRIC
    across_input_var = float(np.mean(np.var(gate_vals, axis=0)))

    # Also compute: per-dimension across-input std for distribution analysis
    per_dim_std = np.std(gate_vals, axis=0)  # (d_ff,)
    across_input_var_median = float(np.median(np.var(gate_vals, axis=0)))
    across_input_var_max = float(np.max(np.var(gate_vals, axis=0)))

    # 6. Effective rank of the gate activation matrix
    # Center the matrix first (subtract mean across samples)
    centered = gate_vals - gate_vals.mean(axis=0, keepdims=True)
    eff_rank = effective_rank(centered)

    # 7. Fraction of variance explained by top-k components
    S = np.linalg.svd(centered, compute_uv=False)
    S_sq = S ** 2
    total_var = S_sq.sum()
    if total_var > 1e-12:
        top10_energy = float(S_sq[:10].sum() / total_var)
        top50_energy = float(S_sq[:50].sum() / total_var)
    else:
        top10_energy = 0.0
        top50_energy = 0.0

    return {
        "mean_magnitude": mean_magnitude,
        "sparsity": sparsity,
        "kurtosis_mean": mean_kurtosis,
        "kurtosis_std": std_kurtosis,
        "within_input_variance": within_input_var,
        "across_input_variance_mean": across_input_var,
        "across_input_variance_median": across_input_var_median,
        "across_input_variance_max": across_input_var_max,
        "effective_rank": eff_rank,
        "top10_energy": top10_energy,
        "top50_energy": top50_energy,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_samples: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gate_selectivity_olmo_1b.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        results = {
            "model": "1b",
            "model_name": MODEL_NAME,
            "model_family": "olmo",
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "n_samples": n_samples,
            "sparsity_thresholds": SPARSITY_THRESHOLDS,
            "checkpoints": {},
        }

    tokenizer = get_tokenizer()
    input_ids = get_sample_input_ids(tokenizer)
    print(f"Input: {input_ids.shape[0]} sequences of length {input_ids.shape[1]} "
          f"({input_ids.shape[0] * input_ids.shape[1]} total tokens)")

    for step in CHECKPOINTS:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Step {step}")
        print(f"{'=' * 60}")

        model = load_model_at_checkpoint(step)
        gate_activations = collect_gate_activations(model, input_ids, n_samples)

        step_results = {"layers": {}}
        for li in range(N_LAYERS):
            print(f"  Computing statistics for layer {li}...")
            stats = compute_gate_statistics(gate_activations[li])
            step_results["layers"][str(li)] = stats

        # Summary: mean across all layers
        all_across_var = [step_results["layers"][str(li)]["across_input_variance_mean"]
                          for li in range(N_LAYERS)]
        all_within_var = [step_results["layers"][str(li)]["within_input_variance"]
                          for li in range(N_LAYERS)]
        all_eff_rank = [step_results["layers"][str(li)]["effective_rank"]
                        for li in range(N_LAYERS)]

        step_results["summary"] = {
            "mean_across_input_variance": float(np.mean(all_across_var)),
            "mean_within_input_variance": float(np.mean(all_within_var)),
            "mean_effective_rank": float(np.mean(all_eff_rank)),
        }

        results["checkpoints"][str(step)] = step_results

        del model, gate_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save after each checkpoint
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to {out_path}")

    print(f"\nDone. Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gate Selectivity Across Training (OLMo-2-1B)")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="Number of token positions to sample per checkpoint")
    parser.add_argument("--output_dir", default="results/gate_selectivity")
    args = parser.parse_args()

    run_experiment(args.n_samples, args.output_dir)
