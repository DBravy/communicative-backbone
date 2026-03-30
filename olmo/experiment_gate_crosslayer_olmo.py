"""
Cross-Layer Gate Coherence Across Training (OLMo-2-1B)
======================================================

For each checkpoint, runs text through the model and captures the gate
activation vector g(x) = SiLU(W_gate @ x) at each layer. Then measures
whether adjacent layers produce similar gate vectors for the same input.

This is the gate-level analog of the weight-level cross-layer subspace
overlap measured in experiment_b. If the SwiGLU gate is taking over
communicative coordination from the static weights, we predict:

- Gate cross-layer coherence should RISE over training
- The onset of rising gate coherence should roughly coincide with the
  onset of FALLING weight-level coherence
- The "handoff" should be visible if both are plotted together

Metrics per layer pair per checkpoint:
- Mean cosine similarity of gate vectors between adjacent layers
  (averaged over all input tokens)
- Pearson correlation of gate vectors (captures linear relationship
  even if magnitudes differ)
- Gate pattern overlap: binarize gate vectors (open/closed), compute
  Jaccard similarity

Also computes non-adjacent pairs (first-last, first-mid, mid-last).

Checkpoints: 0, 1000, 3000, 5000, 10000, 100000, 1000000

Usage:
    python experiment_gate_crosslayer_olmo.py [--n_samples 256]
"""

import argparse
import json
import math
import os

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

CHECKPOINTS = [0, 1000, 3000, 5000, 10000, 100000, 1000000]

# Threshold for binarizing gate activations (open vs closed)
GATE_OPEN_THRESHOLD = 0.1

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


def collect_gate_vectors(model, input_ids, n_samples=256):
    """
    Run input through model, collect gate activation vectors at each layer.
    Returns dict: layer_idx -> np.ndarray of shape (n_samples, d_ff)
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

    all_mlp_inputs = {i: [] for i in range(N_LAYERS)}

    for batch_idx in range(input_ids.shape[0]):
        ids = input_ids[batch_idx:batch_idx + 1].to(device)
        with torch.no_grad():
            model(ids)
        for li in range(N_LAYERS):
            all_mlp_inputs[li].append(mlp_inputs[li].squeeze(0))

    for h in hooks:
        h.remove()

    gate_vectors = {}
    for li in range(N_LAYERS):
        x = torch.cat(all_mlp_inputs[li], dim=0)  # (total_tokens, d_model)
        if x.shape[0] > n_samples:
            # Use fixed seed for reproducibility across layers/checkpoints
            rng = np.random.RandomState(42)
            indices = rng.choice(x.shape[0], n_samples, replace=False)
            x = x[indices]

        W_gate = model.model.layers[li].mlp.gate_proj.weight.detach().cpu().float()
        gate_pre = x.float() @ W_gate.T
        gate_vals = torch.nn.functional.silu(gate_pre)
        gate_vectors[li] = gate_vals.numpy()

    return gate_vectors


# ---------------------------------------------------------------------------
# Cross-layer coherence metrics
# ---------------------------------------------------------------------------

def cosine_similarity_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity for each row of A and B. Returns (n_samples,)."""
    dot = np.sum(A * B, axis=1)
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    denom = norm_a * norm_b
    denom = np.maximum(denom, 1e-12)
    return dot / denom


def pearson_correlation_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation for each row of A and B. Returns (n_samples,)."""
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)
    return cosine_similarity_batch(A_centered, B_centered)


def jaccard_similarity_batch(A: np.ndarray, B: np.ndarray,
                              threshold: float) -> np.ndarray:
    """Jaccard similarity of binarized gate patterns. Returns (n_samples,)."""
    A_open = np.abs(A) > threshold
    B_open = np.abs(B) > threshold
    intersection = np.sum(A_open & B_open, axis=1).astype(float)
    union = np.sum(A_open | B_open, axis=1).astype(float)
    union = np.maximum(union, 1.0)
    return intersection / union


def compute_pair_coherence(gate_a: np.ndarray, gate_b: np.ndarray) -> dict:
    """
    Compute all coherence metrics between gate vectors at two layers.
    gate_a, gate_b: (n_samples, d_ff)
    """
    cos_sims = cosine_similarity_batch(gate_a, gate_b)
    pearson = pearson_correlation_batch(gate_a, gate_b)
    jaccard = jaccard_similarity_batch(gate_a, gate_b, GATE_OPEN_THRESHOLD)

    return {
        "cosine_similarity": {
            "mean": float(np.mean(cos_sims)),
            "std": float(np.std(cos_sims)),
            "median": float(np.median(cos_sims)),
            "min": float(np.min(cos_sims)),
            "max": float(np.max(cos_sims)),
        },
        "pearson_correlation": {
            "mean": float(np.mean(pearson)),
            "std": float(np.std(pearson)),
            "median": float(np.median(pearson)),
        },
        "jaccard_similarity": {
            "mean": float(np.mean(jaccard)),
            "std": float(np.std(jaccard)),
            "median": float(np.median(jaccard)),
        },
    }


def random_gate_baseline(d_ff: int, n_samples: int = 256,
                          n_trials: int = 50) -> dict:
    """Expected coherence between random gate-like vectors."""
    cos_vals = []
    for _ in range(n_trials):
        A = np.random.randn(n_samples, d_ff)
        B = np.random.randn(n_samples, d_ff)
        cos_vals.append(float(np.mean(cosine_similarity_batch(A, B))))
    return {
        "cosine_mean": float(np.mean(cos_vals)),
        "cosine_std": float(np.std(cos_vals)),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_samples: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gate_crosslayer_olmo_1b.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        print("Computing random baseline...")
        baseline = random_gate_baseline(D_FF, n_samples)
        print(f"  Random cosine similarity: {baseline['cosine_mean']:.6f} "
              f"+/- {baseline['cosine_std']:.6f}")

        results = {
            "model": "1b",
            "model_name": MODEL_NAME,
            "model_family": "olmo",
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "n_samples": n_samples,
            "gate_open_threshold": GATE_OPEN_THRESHOLD,
            "random_baseline": baseline,
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
        gate_vectors = collect_gate_vectors(model, input_ids, n_samples)

        step_results = {"adjacent_pairs": {}, "non_adjacent": {}}

        # Adjacent layer pairs
        for li in range(N_LAYERS - 1):
            pair_key = f"{li}_{li + 1}"
            coherence = compute_pair_coherence(
                gate_vectors[li], gate_vectors[li + 1]
            )
            step_results["adjacent_pairs"][pair_key] = coherence
            cos_mean = coherence["cosine_similarity"]["mean"]
            print(f"    Layers {li}-{li+1}: "
                  f"cosine={cos_mean:.4f}  "
                  f"pearson={coherence['pearson_correlation']['mean']:.4f}  "
                  f"jaccard={coherence['jaccard_similarity']['mean']:.4f}")

        # Non-adjacent pairs
        global_pairs = [
            (0, N_LAYERS - 1, "first_last"),
            (0, N_LAYERS // 2, "first_mid"),
            (N_LAYERS // 2, N_LAYERS - 1, "mid_last"),
        ]
        for l1, l2, label in global_pairs:
            coherence = compute_pair_coherence(gate_vectors[l1], gate_vectors[l2])
            step_results["non_adjacent"][label] = coherence
            print(f"    Global {label} ({l1}-{l2}): "
                  f"cosine={coherence['cosine_similarity']['mean']:.4f}")

        # Summary: mean adjacent cosine similarity
        adj_cosines = [
            step_results["adjacent_pairs"][f"{li}_{li+1}"]["cosine_similarity"]["mean"]
            for li in range(N_LAYERS - 1)
        ]
        step_results["summary"] = {
            "mean_adjacent_cosine": float(np.mean(adj_cosines)),
            "std_adjacent_cosine": float(np.std(adj_cosines)),
            "min_adjacent_cosine": float(np.min(adj_cosines)),
            "max_adjacent_cosine": float(np.max(adj_cosines)),
        }

        results["checkpoints"][str(step)] = step_results

        del model, gate_vectors
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
        description="Cross-Layer Gate Coherence (OLMo-2-1B)")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="Number of token positions to sample per checkpoint")
    parser.add_argument("--output_dir", default="results/gate_crosslayer")
    args = parser.parse_args()

    run_experiment(args.n_samples, args.output_dir)
