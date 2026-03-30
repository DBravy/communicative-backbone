"""
Jacobian Cross-Layer Subspace Overlap (OLMo-2-1B)
==================================================

Computes the input-dependent Jacobian of the full SwiGLU MLP (including
the gate) at each layer, then measures cross-layer subspace overlap on
the Jacobian's singular vectors rather than the static weight matrices.

This is the definitive test of whether effective cross-layer coherence
is maintained even as weight-level coherence declines. If the SwiGLU gate
is absorbing communicative responsibility from the weights, then:

- Weight-level cross-layer overlap should decline (already observed)
- Jacobian-level cross-layer overlap should hold steady or rise
- The divergence between the two should grow over training

Jacobian derivation:
    The SwiGLU MLP computes f(x) = W_down @ (SiLU(W_gate @ x) * (W_up @ x))

    Let g = W_gate @ x, u = W_up @ x, sigma = SiLU.
    Then f(x) = W_down @ (sigma(g) * u)

    The Jacobian J = df/dx is:
        J = W_down @ (diag(sigma'(g) * u) @ W_gate + diag(sigma(g)) @ W_up)

    where sigma'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
                    = sigmoid(z) * (1 + z * (1 - sigmoid(z)))

    For each input x, this yields a (d_model, d_model) matrix.

The weight-level analysis (W_down @ W_up) is also computed at the same
checkpoints for direct comparison.

Checkpoints: 0, 1000, 3000, 5000, 10000, 100000, 1000000

Usage:
    python experiment_jacobian_crosslayer_olmo.py [--n_samples 128]
"""

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch

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

CHECKPOINTS = [0, 10000, 1000000]

K_VALUES = [5, 10, 20, 50]

# ---------------------------------------------------------------------------
# Sample text
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
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    return model


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def get_sample_input_ids(tokenizer, seq_len=128):
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


def collect_mlp_inputs(model, input_ids, n_samples=128):
    """
    Run input through model, capture the input tensor to each MLP layer.
    Returns dict: layer_idx -> np.ndarray of shape (n_samples, d_model)
    """
    mlp_inputs_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            mlp_inputs_store[layer_idx] = args[0].detach().cpu()
        return hook_fn

    hooks = []
    for i in range(N_LAYERS):
        h = model.model.layers[i].mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    device = next(model.parameters()).device
    all_mlp_inputs = {i: [] for i in range(N_LAYERS)}

    for batch_idx in range(input_ids.shape[0]):
        ids = input_ids[batch_idx:batch_idx + 1].to(device)
        with torch.no_grad():
            model(ids)
        for li in range(N_LAYERS):
            all_mlp_inputs[li].append(mlp_inputs_store[li].squeeze(0))

    for h in hooks:
        h.remove()

    result = {}
    rng = np.random.RandomState(42)
    for li in range(N_LAYERS):
        x = torch.cat(all_mlp_inputs[li], dim=0).float().numpy()
        if x.shape[0] > n_samples:
            indices = rng.choice(x.shape[0], n_samples, replace=False)
            x = x[indices]
        result[li] = x

    return result


def extract_layer_weights(model, layer_idx):
    """Extract one layer's MLP weights as fp32 numpy arrays."""
    mlp = model.model.layers[layer_idx].mlp
    return {
        "W_gate": mlp.gate_proj.weight.detach().cpu().float().numpy(),
        "W_up": mlp.up_proj.weight.detach().cpu().float().numpy(),
        "W_down": mlp.down_proj.weight.detach().cpu().float().numpy(),
    }


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return x * sig


def silu_derivative(x):
    """Derivative of SiLU: sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return sig * (1.0 + x * (1.0 - sig))


def compute_jacobian(x_vec, W_gate, W_up, W_down):
    """
    Compute the Jacobian of the SwiGLU MLP at input x_vec.

    f(x) = W_down @ (SiLU(W_gate @ x) * (W_up @ x))

    J = W_down @ (diag(SiLU'(g) * u) @ W_gate + diag(SiLU(g)) @ W_up)

    Args:
        x_vec: (d_model,) input vector
        W_gate: (d_ff, d_model) gate projection weights
        W_up: (d_ff, d_model) up projection weights
        W_down: (d_model, d_ff) down projection weights

    Returns:
        J: (d_model, d_model) Jacobian matrix
    """
    g = W_gate @ x_vec            # (d_ff,)
    u = W_up @ x_vec              # (d_ff,)

    sigma_g = silu(g)             # (d_ff,)
    sigma_prime_g = silu_derivative(g)  # (d_ff,)

    # Inner: diag(sigma'(g) * u) @ W_gate + diag(sigma(g)) @ W_up
    # = (sigma'(g) * u)[:, None] * W_gate + sigma(g)[:, None] * W_up
    d1 = sigma_prime_g * u        # (d_ff,)
    d2 = sigma_g                  # (d_ff,)

    inner = d1[:, None] * W_gate + d2[:, None] * W_up  # (d_ff, d_model)

    J = W_down @ inner  # (d_model, d_model)
    return J


# ---------------------------------------------------------------------------
# Subspace overlap (from reference script)
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return cosines


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> dict:
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


def random_subspace_baseline(d: int, k: int, n_trials: int = 200) -> dict:
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


# ---------------------------------------------------------------------------
# Per-layer Jacobian SVD computation
# ---------------------------------------------------------------------------

def compute_layer_jacobian_svds(mlp_inputs, layer_weights, max_k=50):
    """
    Compute the Jacobian SVD for each input at a given layer.

    Returns list of dicts, one per input, each containing:
        - U_topk: (d_model, max_k) left singular vectors
        - S_topk: (max_k,) singular values
    """
    W_gate = layer_weights["W_gate"]
    W_up = layer_weights["W_up"]
    W_down = layer_weights["W_down"]

    n_samples = mlp_inputs.shape[0]
    svd_results = []

    for i in range(n_samples):
        J = compute_jacobian(mlp_inputs[i], W_gate, W_up, W_down)
        # Full SVD — keep all singular values for spectral profiling,
        # but only the top max_k left singular vectors for overlap analysis.
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        svd_results.append({
            "U_topk": U[:, :max_k],
            "S_topk": S[:max_k],
            "S_full": S,
        })

    return svd_results


def compute_layer_weight_svd(layer_weights, max_k=50):
    """
    Compute weight-level SVD of W_down @ W_up (for comparison).
    Returns U_topk: (d_model, max_k), S_topk: (max_k,)
    """
    W_up = layer_weights["W_up"]
    W_down = layer_weights["W_down"]

    composed = W_down @ W_up
    U, S, Vt = np.linalg.svd(composed, full_matrices=False)
    return U[:, :max_k], S[:max_k]


# ---------------------------------------------------------------------------
# Cross-layer overlap computation
# ---------------------------------------------------------------------------

def compute_crosslayer_jacobian_overlap(svds_layer_a, svds_layer_b, k_values):
    """
    For each input, compute subspace overlap between Jacobian top-k
    singular vectors at two layers. Average over inputs.

    Returns dict with per-k statistics.
    """
    n_samples = len(svds_layer_a)
    results = {}

    for k in k_values:
        per_input_overlaps = []
        for i in range(n_samples):
            U_a = svds_layer_a[i]["U_topk"][:, :k]
            U_b = svds_layer_b[i]["U_topk"][:, :k]
            overlap = subspace_overlap(U_a, U_b)
            per_input_overlaps.append(overlap)

        # Aggregate across inputs
        mean_cosines = [o["mean_cosine"] for o in per_input_overlaps]
        grassmann_dists = [o["grassmann_distance"] for o in per_input_overlaps]

        results[f"top{k}"] = {
            "mean_cosine_mean": float(np.mean(mean_cosines)),
            "mean_cosine_std": float(np.std(mean_cosines)),
            "mean_cosine_median": float(np.median(mean_cosines)),
            "grassmann_mean": float(np.mean(grassmann_dists)),
            "grassmann_std": float(np.std(grassmann_dists)),
        }

    return results


def compute_crosslayer_weight_overlap(U_a, U_b, k_values):
    """Weight-level subspace overlap for comparison."""
    results = {}
    for k in k_values:
        overlap = subspace_overlap(U_a[:, :k], U_b[:, :k])
        results[f"top{k}"] = overlap
    return results


# ---------------------------------------------------------------------------
# Jacobian spectral summary per layer
# ---------------------------------------------------------------------------

def compute_jacobian_spectral_summary(svd_results, k_values):
    """
    Aggregate spectral statistics from per-input Jacobian SVDs.
    Uses S_full (all singular values) for accurate effective rank,
    norm, and energy concentration.
    """
    n_samples = len(svd_results)

    # Collect top singular values for spectrum reporting
    all_S_top = np.array([r["S_topk"] for r in svd_results])
    mean_S = np.mean(all_S_top, axis=0)
    std_S = np.std(all_S_top, axis=0)

    # Full-spectrum effective rank and norm per input
    eff_ranks = []
    norms = []
    for r in svd_results:
        S = r["S_full"]
        S_sq = S ** 2
        total = S_sq.sum()
        norms.append(float(np.sqrt(total)))
        if total < 1e-12:
            eff_ranks.append(0.0)
            continue
        p = S_sq / total
        p = p[p > 1e-12]
        eff_ranks.append(float(np.exp(-np.sum(p * np.log(p)))))

    # Top-k energy fraction (numerator = top-k, denominator = full spectrum)
    energy = {}
    for k in k_values:
        per_input = []
        for r in svd_results:
            S = r["S_full"]
            S_sq = S ** 2
            total = S_sq.sum()
            if total < 1e-12:
                per_input.append(0.0)
            else:
                per_input.append(float(S_sq[:k].sum() / total))
        energy[f"top{k}"] = {
            "mean": float(np.mean(per_input)),
            "std": float(np.std(per_input)),
        }

    return {
        "mean_singular_values_top10": mean_S[:10].tolist(),
        "std_singular_values_top10": std_S[:10].tolist(),
        "effective_rank_mean": float(np.mean(eff_ranks)),
        "effective_rank_std": float(np.std(eff_ranks)),
        "effective_rank_median": float(np.median(eff_ranks)),
        "effective_rank_percentiles": {
            "p10": float(np.percentile(eff_ranks, 10)),
            "p25": float(np.percentile(eff_ranks, 25)),
            "p50": float(np.percentile(eff_ranks, 50)),
            "p75": float(np.percentile(eff_ranks, 75)),
            "p90": float(np.percentile(eff_ranks, 90)),
        },
        "jacobian_mean_norm": float(np.mean(norms)),
        "jacobian_std_norm": float(np.std(norms)),
        "n_samples": n_samples,
        "energy_concentration": energy,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_samples: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "jacobian_crosslayer_olmo_1b.json")

    max_k = max(K_VALUES)

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        print("Computing random subspace baselines...")
        baselines = {}
        for k in K_VALUES:
            baselines[k] = random_subspace_baseline(D_MODEL, k, n_trials=200)
            print(f"  k={k}: random mean_cosine = "
                  f"{baselines[k]['mean_cosine_mean']:.4f} +/- "
                  f"{baselines[k]['mean_cosine_std']:.4f}")

        results = {
            "model": "1b",
            "model_name": MODEL_NAME,
            "model_family": "olmo",
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "n_samples": n_samples,
            "k_values": K_VALUES,
            "random_baselines": {str(k): v for k, v in baselines.items()},
            "checkpoints": {},
        }

    tokenizer = get_tokenizer()
    input_ids = get_sample_input_ids(tokenizer)
    print(f"Input: {input_ids.shape[0]} sequences of length {input_ids.shape[1]}")

    for step in CHECKPOINTS:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Step {step}")
        print(f"{'=' * 60}")

        model = load_model_at_checkpoint(step)
        mlp_inputs = collect_mlp_inputs(model, input_ids, n_samples)

        n_actual = mlp_inputs[0].shape[0]
        print(f"  Collected {n_actual} MLP input vectors per layer")

        # --- Compute Jacobian and weight SVDs one layer at a time ---
        # Only one layer's fp32 numpy weights (~192MB) exist at a time
        # on top of the fp16 model (~2.4GB).
        print(f"  Computing Jacobian SVDs ({n_actual} inputs x {N_LAYERS} layers)...")
        t0 = time.time()

        jacobian_svds = {}
        weight_svds = {}
        for li in range(N_LAYERS):
            layer_t0 = time.time()
            lw = extract_layer_weights(model, li)
            jacobian_svds[li] = compute_layer_jacobian_svds(
                mlp_inputs[li], lw, max_k=max_k
            )
            U, S = compute_layer_weight_svd(lw, max_k=max_k)
            weight_svds[li] = {"U": U, "S": S}
            del lw
            elapsed = time.time() - layer_t0
            print(f"    Layer {li}: {elapsed:.1f}s")

        total_time = time.time() - t0
        print(f"  Total SVD time: {total_time:.1f}s")

        # Free the model now that all SVDs are computed
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Cross-layer overlaps ---
        step_results = {
            "jacobian": {
                "adjacent_pairs": {},
                "non_adjacent": {},
                "spectral": {},
            },
            "weights": {
                "adjacent_pairs": {},
                "non_adjacent": {},
            },
        }

        # Jacobian spectral summary per layer
        for li in range(N_LAYERS):
            summary = compute_jacobian_spectral_summary(jacobian_svds[li], K_VALUES)
            step_results["jacobian"]["spectral"][str(li)] = summary

        # Adjacent pairs
        for li in range(N_LAYERS - 1):
            pair_key = f"{li}_{li + 1}"

            # Jacobian-level
            jac_overlap = compute_crosslayer_jacobian_overlap(
                jacobian_svds[li], jacobian_svds[li + 1], K_VALUES
            )
            step_results["jacobian"]["adjacent_pairs"][pair_key] = jac_overlap

            # Weight-level
            wt_overlap = compute_crosslayer_weight_overlap(
                weight_svds[li]["U"], weight_svds[li + 1]["U"], K_VALUES
            )
            step_results["weights"]["adjacent_pairs"][pair_key] = wt_overlap

            jac_cos = jac_overlap["top10"]["mean_cosine_mean"]
            wt_cos = wt_overlap["top10"]["mean_cosine"]
            print(f"    Layers {li}-{li+1}: "
                  f"jacobian={jac_cos:.4f}  weight={wt_cos:.4f}  "
                  f"delta={jac_cos - wt_cos:+.4f}")

        # Non-adjacent pairs
        global_pairs = [
            (0, N_LAYERS - 1, "first_last"),
            (0, N_LAYERS // 2, "first_mid"),
            (N_LAYERS // 2, N_LAYERS - 1, "mid_last"),
        ]
        for l1, l2, label in global_pairs:
            jac_overlap = compute_crosslayer_jacobian_overlap(
                jacobian_svds[l1], jacobian_svds[l2], K_VALUES
            )
            step_results["jacobian"]["non_adjacent"][label] = jac_overlap

            wt_overlap = compute_crosslayer_weight_overlap(
                weight_svds[l1]["U"], weight_svds[l2]["U"], K_VALUES
            )
            step_results["weights"]["non_adjacent"][label] = wt_overlap

            print(f"    Global {label}: "
                  f"jacobian={jac_overlap['top10']['mean_cosine_mean']:.4f}  "
                  f"weight={wt_overlap['top10']['mean_cosine']:.4f}")

        # Summary: mean adjacent overlap at k=10
        jac_adj = [
            step_results["jacobian"]["adjacent_pairs"][f"{li}_{li+1}"]["top10"]["mean_cosine_mean"]
            for li in range(N_LAYERS - 1)
        ]
        wt_adj = [
            step_results["weights"]["adjacent_pairs"][f"{li}_{li+1}"]["top10"]["mean_cosine"]
            for li in range(N_LAYERS - 1)
        ]
        step_results["summary"] = {
            "mean_adjacent_jacobian_top10": float(np.mean(jac_adj)),
            "mean_adjacent_weight_top10": float(np.mean(wt_adj)),
            "delta_jacobian_minus_weight": float(np.mean(jac_adj) - np.mean(wt_adj)),
            "computation_time_seconds": total_time,
        }

        results["checkpoints"][str(step)] = step_results

        del mlp_inputs, jacobian_svds, weight_svds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        description="Jacobian Cross-Layer Subspace Overlap (OLMo-2-1B)")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of token positions to sample per checkpoint")
    parser.add_argument("--output_dir", default="results/jacobian_crosslayer")
    args = parser.parse_args()

    run_experiment(args.n_samples, args.output_dir)