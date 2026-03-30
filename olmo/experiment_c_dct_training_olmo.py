"""
Experiment C (OLMo): DCT Energy Concentration Across Training
===============================================================

OLMo port of experiment_c_dct_training.py (originally for Pythia).

Tracks how the frequency-domain structure of hidden-state trajectories evolves
during OLMo training. At each checkpoint, runs a batch of tokens through the
model, extracts the hidden-state trajectory (residual stream at each layer),
applies DCT decomposition, and measures energy concentration in low frequencies.

Models: OLMo-1B, OLMo-7B (v1, with intermediate checkpoints)
Checkpoints: sampled across training (every 1000 steps available)
Data: wikitext-103 validation (fallback: random tokens)

Checkpoint loading:
    Requires original (non-HF) model IDs for intermediate checkpoints:
        pip install ai2-olmo
    Models: allenai/OLMo-1B, allenai/OLMo-7B

Usage:
    python experiment_c_dct_training_olmo.py [--models 1b 7b] [--device cuda]
                                              [--n_sequences 200] [--seq_len 128]
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.fftpack import dct

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

CHECKPOINTS_1B = [0, 1000, 5000, 10000, 20000, 100000, 500000, 1000000]

MODEL_CONFIGS = {
    "1b": {
        "name": "allenai/OLMo-2-0425-1B",
        "d_model": 2048, "n_layers": 16,
        "checkpoints": CHECKPOINTS_1B,
    },
}

DEFAULT_N_SEQUENCES = 200
DEFAULT_SEQ_LEN = 128
DEFAULT_POSITIONS_PER_SEQ = 5


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def step_to_revision(step: int) -> str:
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_trajectories(model, input_ids: torch.Tensor,
                                 positions: list) -> np.ndarray:
    """Run model and extract hidden-state trajectories at specified positions.

    Returns:
        trajectories: (n_positions, n_layers+1, d_model) array
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states

    trajectories = []
    for pos in positions:
        traj = np.stack([
            hidden_states[layer_idx][0, pos, :].cpu().float().numpy()
            for layer_idx in range(len(hidden_states))
        ])
        trajectories.append(traj)

    return np.array(trajectories)


# ---------------------------------------------------------------------------
# DCT analysis (identical to Pythia version)
# ---------------------------------------------------------------------------

def dct_energy_spectrum(trajectory: np.ndarray) -> np.ndarray:
    """Compute normalized DCT energy spectrum of a trajectory."""
    n_layers, d_model = trajectory.shape
    dct_coeffs = dct(trajectory, type=2, axis=0, norm='ortho')
    energy = np.sum(dct_coeffs ** 2, axis=1)
    total = energy.sum()
    if total > 1e-12:
        energy /= total
    return energy


def spectral_metrics(energy_spectrum: np.ndarray) -> dict:
    """Compute summary metrics from a normalized energy spectrum."""
    n = len(energy_spectrum)
    freqs = np.arange(n)

    n_low = max(1, n // 4)
    low_freq_ratio = float(energy_spectrum[:n_low].sum())
    dc_ratio = float(energy_spectrum[0])
    centroid = float(np.sum(freqs * energy_spectrum))

    eps = 1e-12
    p = energy_spectrum + eps
    p = p / p.sum()
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(n))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    top3_ratio = float(np.sort(energy_spectrum)[-3:].sum())

    return {
        "low_freq_ratio": low_freq_ratio,
        "dc_ratio": dc_ratio,
        "centroid": centroid,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top3_ratio": top3_ratio,
        "n_frequencies": n,
    }


# ---------------------------------------------------------------------------
# Model and data loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, step: int, device: str):
    """Load OLMo 2 model and tokenizer at a specific checkpoint."""
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else model_name
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo, cache_dir=CACHE_DIR,
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )

    model.eval()
    model.to(device)

    return model, tokenizer


def get_input_data(tokenizer, n_sequences: int, seq_len: int,
                   device: str) -> list:
    """Generate input sequences for analysis.

    Tries wikitext-103 validation, falls back to random tokens.
    """
    sequences = []

    try:
        from datasets import load_dataset
        print("  Loading dataset for input sequences...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1",
                               split="validation", trust_remote_code=True)

        texts = [t for t in dataset["text"] if len(t.strip()) > 50]
        collected = 0
        for text in texts:
            if collected >= n_sequences:
                break
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) >= seq_len:
                input_ids = torch.tensor([tokens[:seq_len]], device=device)
                sequences.append(input_ids)
                collected += 1

        if collected >= n_sequences:
            print(f"  Loaded {collected} sequences from wikitext-103")
            return sequences
    except Exception as e:
        print(f"  Could not load dataset: {e}")

    print(f"  Using random token sequences (n={n_sequences}, len={seq_len})")
    vocab_size = tokenizer.vocab_size
    for _ in range(n_sequences):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        sequences.append(input_ids)

    return sequences


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(model_key: str, device: str, output_dir: str,
                   n_sequences: int, seq_len: int, positions_per_seq: int):
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["name"]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    checkpoints = cfg["checkpoints"]

    results = {
        "model": model_key,
        "model_name": model_name,
        "model_family": "olmo",
        "d_model": d_model,
        "n_layers": n_layers,
        "n_sequences": n_sequences,
        "seq_len": seq_len,
        "positions_per_seq": positions_per_seq,
        "n_dct_frequencies": n_layers + 1,
        "checkpoints": {},
    }

    for step in checkpoints:
        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_name, step, device)
        sequences = get_input_data(tokenizer, n_sequences, seq_len, device)

        rng = np.random.RandomState(42)

        all_spectra = []
        all_metrics = []

        for seq_idx, input_ids in enumerate(sequences):
            valid_positions = list(range(2, seq_len))
            positions = sorted(rng.choice(valid_positions,
                                          size=min(positions_per_seq, len(valid_positions)),
                                          replace=False))

            trajectories = extract_hidden_trajectories(model, input_ids, positions)

            for traj in trajectories:
                spectrum = dct_energy_spectrum(traj)
                metrics = spectral_metrics(spectrum)
                all_spectra.append(spectrum)
                all_metrics.append(metrics)

            if (seq_idx + 1) % 50 == 0:
                print(f"    Processed {seq_idx + 1}/{len(sequences)} sequences...")

        all_spectra = np.array(all_spectra)
        mean_spectrum = all_spectra.mean(axis=0)
        std_spectrum = all_spectra.std(axis=0)

        # Cross-token cosine similarity
        n_tokens = len(all_spectra)
        if n_tokens > 1:
            n_pairs = min(1000, n_tokens * (n_tokens - 1) // 2)
            cos_sims = []
            for _ in range(n_pairs):
                i, j = rng.choice(n_tokens, size=2, replace=False)
                a, b = all_spectra[i], all_spectra[j]
                norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
                if norm_a > 1e-12 and norm_b > 1e-12:
                    cos_sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
            cross_token_similarity = {
                "mean": float(np.mean(cos_sims)),
                "std": float(np.std(cos_sims)),
                "min": float(np.min(cos_sims)),
                "n_pairs": len(cos_sims),
            }
        else:
            cross_token_similarity = {"mean": 1.0, "std": 0.0, "min": 1.0, "n_pairs": 0}

        agg_metrics = {}
        for key in all_metrics[0].keys():
            if key == "n_frequencies":
                agg_metrics[key] = all_metrics[0][key]
            else:
                vals = [m[key] for m in all_metrics]
                agg_metrics[f"{key}_mean"] = float(np.mean(vals))
                agg_metrics[f"{key}_std"] = float(np.std(vals))

        step_results = {
            "mean_spectrum": mean_spectrum.tolist(),
            "std_spectrum": std_spectrum.tolist(),
            "cross_token_similarity": cross_token_similarity,
            "metrics": agg_metrics,
            "n_tokens_analyzed": n_tokens,
        }

        results["checkpoints"][str(step)] = step_results

        print(f"    Tokens analyzed: {n_tokens}")
        print(f"    DC ratio: {agg_metrics['dc_ratio_mean']:.4f} +/- {agg_metrics['dc_ratio_std']:.4f}")
        print(f"    Low-freq ratio: {agg_metrics['low_freq_ratio_mean']:.4f}")
        print(f"    Centroid: {agg_metrics['centroid_mean']:.2f}")
        print(f"    Norm entropy: {agg_metrics['normalized_entropy_mean']:.4f}")
        print(f"    Cross-token similarity: {cross_token_similarity['mean']:.6f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dct_training_olmo_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_dir: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    model_key = results["model"]
    checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
    n_freq = results["n_dct_frequencies"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"DCT Energy Concentration During Training: OLMo-{model_key.upper()}",
                 fontsize=14)

    cmap = plt.cm.plasma
    norm = Normalize(vmin=0, vmax=len(checkpoints) - 1)

    # Panel 1: Mean energy spectra
    ax = axes[0, 0]
    for ci, step in enumerate(checkpoints):
        spectrum = results["checkpoints"][str(step)]["mean_spectrum"]
        color = cmap(norm(ci))
        ax.plot(range(len(spectrum)), spectrum, '-', color=color,
                alpha=0.8, label=f"Step {step}")
    ax.set_xlabel("DCT frequency index")
    ax.set_ylabel("Normalized energy")
    ax.set_title("Mean energy spectrum")
    ax.legend(fontsize=7, ncol=2)

    # Panel 2: DC ratio
    ax = axes[0, 1]
    dc_vals = [results["checkpoints"][str(s)]["metrics"]["dc_ratio_mean"]
               for s in checkpoints]
    dc_stds = [results["checkpoints"][str(s)]["metrics"]["dc_ratio_std"]
               for s in checkpoints]
    ax.errorbar(checkpoints, dc_vals, yerr=dc_stds, fmt='o-', capsize=3,
                color='tab:red')
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("DC component energy fraction")
    ax.set_title("DC (smoothest mode) concentration")

    # Panel 3: Low-frequency ratio
    ax = axes[0, 2]
    lf_vals = [results["checkpoints"][str(s)]["metrics"]["low_freq_ratio_mean"]
               for s in checkpoints]
    lf_stds = [results["checkpoints"][str(s)]["metrics"]["low_freq_ratio_std"]
               for s in checkpoints]
    ax.errorbar(checkpoints, lf_vals, yerr=lf_stds, fmt='o-', capsize=3,
                color='tab:blue')
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Energy in bottom-25% frequencies")
    ax.set_title("Low-frequency energy ratio")

    # Panel 4: Frequency centroid
    ax = axes[1, 0]
    cent_vals = [results["checkpoints"][str(s)]["metrics"]["centroid_mean"]
                 for s in checkpoints]
    cent_stds = [results["checkpoints"][str(s)]["metrics"]["centroid_std"]
                 for s in checkpoints]
    ax.errorbar(checkpoints, cent_vals, yerr=cent_stds, fmt='o-', capsize=3,
                color='tab:green')
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Frequency centroid")
    ax.set_title("Spectral centroid (lower = smoother)")
    ax.axhline((n_freq - 1) / 2, color='gray', linestyle='--', alpha=0.5,
               label='Uniform centroid')
    ax.legend(fontsize=9)

    # Panel 5: Normalized entropy
    ax = axes[1, 1]
    ent_vals = [results["checkpoints"][str(s)]["metrics"]["normalized_entropy_mean"]
                for s in checkpoints]
    ent_stds = [results["checkpoints"][str(s)]["metrics"]["normalized_entropy_std"]
                for s in checkpoints]
    ax.errorbar(checkpoints, ent_vals, yerr=ent_stds, fmt='o-', capsize=3,
                color='tab:purple')
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized spectral entropy")
    ax.set_title("Spectral entropy (1 = uniform, 0 = concentrated)")

    # Panel 6: Cross-token similarity
    ax = axes[1, 2]
    sim_vals = [results["checkpoints"][str(s)]["cross_token_similarity"]["mean"]
                for s in checkpoints]
    sim_stds = [results["checkpoints"][str(s)]["cross_token_similarity"]["std"]
                for s in checkpoints]
    ax.errorbar(checkpoints, sim_vals, yerr=sim_stds, fmt='o-', capsize=3,
                color='tab:orange')
    ax.set_xscale('symlog', linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Cross-token spectral envelope similarity")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"dct_training_olmo_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Cross-model comparison plot
# ---------------------------------------------------------------------------

def plot_cross_model(results_list: list, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DCT Concentration: OLMo Model Size Comparison", fontsize=14)

    colors = {'1b': 'tab:blue', '7b': 'tab:red'}

    for results in results_list:
        model_key = results["model"]
        checkpoints = sorted([int(s) for s in results["checkpoints"].keys()])
        color = colors.get(model_key, 'tab:gray')

        dc_vals = [results["checkpoints"][str(s)]["metrics"]["dc_ratio_mean"]
                   for s in checkpoints]
        cent_vals = [results["checkpoints"][str(s)]["metrics"]["centroid_mean"]
                     for s in checkpoints]
        ent_vals = [results["checkpoints"][str(s)]["metrics"]["normalized_entropy_mean"]
                    for s in checkpoints]

        axes[0].plot(checkpoints, dc_vals, 'o-', color=color,
                     label=f"OLMo-{model_key.upper()}")
        axes[1].plot(checkpoints, cent_vals, 'o-', color=color,
                     label=f"OLMo-{model_key.upper()}")
        axes[2].plot(checkpoints, ent_vals, 'o-', color=color,
                     label=f"OLMo-{model_key.upper()}")

    for ax in axes:
        ax.set_xscale('symlog', linthresh=1000)
        ax.set_xlabel("Training step")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("DC energy fraction")
    axes[0].set_title("DC concentration")
    axes[1].set_ylabel("Frequency centroid")
    axes[1].set_title("Spectral centroid")
    axes[2].set_ylabel("Normalized entropy")
    axes[2].set_title("Spectral entropy")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dct_training_olmo_cross_model.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Cross-model plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment C (OLMo): DCT energy concentration across training")
    parser.add_argument("--models", nargs="+", default=["1b"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results/experiment_c_olmo")
    parser.add_argument("--n_sequences", type=int, default=DEFAULT_N_SEQUENCES)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--positions_per_seq", type=int, default=DEFAULT_POSITIONS_PER_SEQ)
    args = parser.parse_args()

    all_results = []
    for model_key in args.models:
        results = run_experiment(
            model_key, args.device, args.output_dir,
            args.n_sequences, args.seq_len, args.positions_per_seq
        )
        plot_results(results, args.output_dir)
        all_results.append(results)

    if len(all_results) > 1:
        plot_cross_model(all_results, args.output_dir)

    print("\nDone. All results saved to", args.output_dir)
