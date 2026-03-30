"""
Experiment C: DCT Energy Concentration Across Training
=======================================================

Tracks how the frequency-domain structure of hidden-state trajectories evolves
during Pythia training. At each checkpoint, runs a batch of tokens through the
model, extracts the hidden-state trajectory (residual stream at each layer),
applies DCT decomposition, and measures how concentrated the energy is in
low-frequency components.

The prediction: low-frequency concentration increases with training, and the
rate/degree of concentration scales with model size. This turns the existing
static observation (70M is spectrally flat, 410M is peaked) into a
developmental trajectory that parallels SVD emergence (Experiment A) and
cross-layer alignment (Experiment B).

Key metrics at each checkpoint:
  1. Normalized energy spectrum across DCT frequencies (averaged over tokens)
  2. Low-frequency energy ratio (fraction of energy in bottom 25% of frequencies)
  3. Frequency centroid (energy-weighted mean frequency)
  4. Spectral entropy (how uniform the distribution is)
  5. Cross-token similarity of spectral envelopes (cosine similarity)

Models: Pythia-70M, Pythia-410M
Checkpoints: 0, 128, 512, 2000, 8000, 32000, 64000, 143000
Data: The Pile validation (or OpenWebText if Pile is unavailable)

Usage:
    python experiment_c_dct_training.py [--models 70m 410m] [--device cuda]
                                        [--n_sequences 200] [--seq_len 128]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.fftpack import dct

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]

MODEL_CONFIGS = {
    "70m":  {"name": "EleutherAI/pythia-70m",  "d_model": 512,  "n_layers": 6},
    "160m": {"name": "EleutherAI/pythia-160m", "d_model": 768,  "n_layers": 12},
    "410m": {"name": "EleutherAI/pythia-410m", "d_model": 1024, "n_layers": 24},
    "1b":   {"name": "EleutherAI/pythia-1b",   "d_model": 2048, "n_layers": 16},
    "1.4b": {"name": "EleutherAI/pythia-1.4b", "d_model": 2048, "n_layers": 24},
}

# Default data settings
DEFAULT_N_SEQUENCES = 200
DEFAULT_SEQ_LEN = 128
DEFAULT_POSITIONS_PER_SEQ = 5  # sample this many token positions per sequence


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_trajectories(model, input_ids: torch.Tensor,
                                 positions: list) -> np.ndarray:
    """Run model and extract hidden-state trajectories at specified positions.
    
    Args:
        model: Pythia model
        input_ids: (batch, seq_len) token ids
        positions: list of token positions to extract
    
    Returns:
        trajectories: (n_positions, n_layers+1, d_model) array
            The "+1" is because we include the embedding layer output (layer 0)
            plus all transformer layer outputs.
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
        )

    # outputs.hidden_states is a tuple of (n_layers+1) tensors,
    # each of shape (batch, seq_len, d_model).
    # Index 0 = embedding output, index i = output of transformer layer i-1.
    hidden_states = outputs.hidden_states

    trajectories = []
    for pos in positions:
        # Stack across layers: (n_layers+1, d_model)
        traj = np.stack([
            hidden_states[layer_idx][0, pos, :].cpu().float().numpy()
            for layer_idx in range(len(hidden_states))
        ])
        trajectories.append(traj)

    return np.array(trajectories)  # (n_positions, n_layers+1, d_model)


# ---------------------------------------------------------------------------
# DCT analysis
# ---------------------------------------------------------------------------

def dct_energy_spectrum(trajectory: np.ndarray) -> np.ndarray:
    """Compute normalized DCT energy spectrum of a trajectory.
    
    Args:
        trajectory: (n_layers, d_model) hidden state at each layer
    
    Returns:
        energy_spectrum: (n_layers,) normalized energy at each DCT frequency
    """
    n_layers, d_model = trajectory.shape
    
    # Apply DCT along the layer axis for each dimension
    # DCT-II (the standard "DCT") applied to each column
    dct_coeffs = dct(trajectory, type=2, axis=0, norm='ortho')  # (n_layers, d_model)
    
    # Energy at each frequency = sum of squared coefficients across dimensions
    energy = np.sum(dct_coeffs ** 2, axis=1)  # (n_layers,)
    
    # Normalize
    total = energy.sum()
    if total > 1e-12:
        energy /= total
    
    return energy


def spectral_metrics(energy_spectrum: np.ndarray) -> dict:
    """Compute summary metrics from a normalized energy spectrum."""
    n = len(energy_spectrum)
    freqs = np.arange(n)
    
    # Low-frequency energy ratio (bottom 25% of frequencies)
    n_low = max(1, n // 4)
    low_freq_ratio = float(energy_spectrum[:n_low].sum())
    
    # DC component (frequency 0) energy
    dc_ratio = float(energy_spectrum[0])
    
    # Frequency centroid (energy-weighted mean frequency)
    centroid = float(np.sum(freqs * energy_spectrum))
    
    # Spectral entropy (how uniform the distribution is)
    # Max entropy = log(n) for uniform distribution
    eps = 1e-12
    p = energy_spectrum + eps
    p = p / p.sum()
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(n))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Top-3 frequency ratio
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
    """Load Pythia model and tokenizer at a specific checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    revision = f"step{step}"
    print(f"  Loading {model_name} at {revision}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    model.to(device)
    
    return model, tokenizer


def get_input_data(tokenizer, n_sequences: int, seq_len: int,
                   device: str) -> list:
    """Generate input sequences for analysis.
    
    Tries to load from The Pile validation set. Falls back to generating
    random token sequences (which is fine for measuring spectral structure,
    since the DCT envelope is architecture-invariant per Source 3).
    """
    sequences = []
    
    # Strategy 1: Try to load a standard dataset
    try:
        from datasets import load_dataset
        print("  Loading dataset for input sequences...")
        # Use a small slice of a public dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1",
                               split="validation", trust_remote_code=True)
        
        # Concatenate text and tokenize
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
    
    # Strategy 2: Random tokens (spectral envelope is invariant to input)
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

    results = {
        "model": model_key,
        "model_name": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_sequences": n_sequences,
        "seq_len": seq_len,
        "positions_per_seq": positions_per_seq,
        "n_dct_frequencies": n_layers + 1,  # includes embedding layer
        "checkpoints": {},
    }

    for step in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"  {model_key} -- step {step}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_name, step, device)
        sequences = get_input_data(tokenizer, n_sequences, seq_len, device)

        # Sample positions (avoiding first few tokens for stability)
        rng = np.random.RandomState(42)
        
        all_spectra = []
        all_metrics = []

        for seq_idx, input_ids in enumerate(sequences):
            # Pick random positions (avoiding position 0 which is often special)
            valid_positions = list(range(2, seq_len))
            positions = sorted(rng.choice(valid_positions,
                                          size=min(positions_per_seq, len(valid_positions)),
                                          replace=False))

            trajectories = extract_hidden_trajectories(model, input_ids, positions)
            # trajectories: (n_positions, n_layers+1, d_model)

            for traj in trajectories:
                spectrum = dct_energy_spectrum(traj)
                metrics = spectral_metrics(spectrum)
                all_spectra.append(spectrum)
                all_metrics.append(metrics)

            if (seq_idx + 1) % 50 == 0:
                print(f"    Processed {seq_idx + 1}/{len(sequences)} sequences...")

        # Aggregate results
        all_spectra = np.array(all_spectra)  # (n_total_tokens, n_frequencies)
        mean_spectrum = all_spectra.mean(axis=0)
        std_spectrum = all_spectra.std(axis=0)

        # Cross-token cosine similarity of spectral envelopes
        # (tests the "architectural invariant" finding from Source 3)
        n_tokens = len(all_spectra)
        if n_tokens > 1:
            # Sample pairs for efficiency
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

        # Aggregate per-token metrics
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

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dct_training_{model_key}.json")
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
    fig.suptitle(f"DCT Energy Concentration During Training: Pythia-{model_key}",
                 fontsize=14)

    cmap = plt.cm.plasma
    norm = Normalize(vmin=0, vmax=len(checkpoints) - 1)

    # --- Panel 1: Mean energy spectra at each checkpoint ---
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

    # --- Panel 2: DC ratio over training ---
    ax = axes[0, 1]
    dc_vals = [results["checkpoints"][str(s)]["metrics"]["dc_ratio_mean"]
               for s in checkpoints]
    dc_stds = [results["checkpoints"][str(s)]["metrics"]["dc_ratio_std"]
               for s in checkpoints]
    ax.errorbar(checkpoints, dc_vals, yerr=dc_stds, fmt='o-', capsize=3,
                color='tab:red')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("DC component energy fraction")
    ax.set_title("DC (smoothest mode) concentration")

    # --- Panel 3: Low-frequency ratio over training ---
    ax = axes[0, 2]
    lf_vals = [results["checkpoints"][str(s)]["metrics"]["low_freq_ratio_mean"]
               for s in checkpoints]
    lf_stds = [results["checkpoints"][str(s)]["metrics"]["low_freq_ratio_std"]
               for s in checkpoints]
    ax.errorbar(checkpoints, lf_vals, yerr=lf_stds, fmt='o-', capsize=3,
                color='tab:blue')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Energy in bottom-25% frequencies")
    ax.set_title("Low-frequency energy ratio")

    # --- Panel 4: Frequency centroid over training ---
    ax = axes[1, 0]
    cent_vals = [results["checkpoints"][str(s)]["metrics"]["centroid_mean"]
                 for s in checkpoints]
    cent_stds = [results["checkpoints"][str(s)]["metrics"]["centroid_std"]
                 for s in checkpoints]
    ax.errorbar(checkpoints, cent_vals, yerr=cent_stds, fmt='o-', capsize=3,
                color='tab:green')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Frequency centroid")
    ax.set_title("Spectral centroid (lower = smoother)")
    # Add reference line for uniform distribution
    ax.axhline((n_freq - 1) / 2, color='gray', linestyle='--', alpha=0.5,
               label='Uniform centroid')
    ax.legend(fontsize=9)

    # --- Panel 5: Normalized entropy over training ---
    ax = axes[1, 1]
    ent_vals = [results["checkpoints"][str(s)]["metrics"]["normalized_entropy_mean"]
                for s in checkpoints]
    ent_stds = [results["checkpoints"][str(s)]["metrics"]["normalized_entropy_std"]
                for s in checkpoints]
    ax.errorbar(checkpoints, ent_vals, yerr=ent_stds, fmt='o-', capsize=3,
                color='tab:purple')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized spectral entropy")
    ax.set_title("Spectral entropy (1 = uniform, 0 = concentrated)")

    # --- Panel 6: Cross-token similarity over training ---
    ax = axes[1, 2]
    sim_vals = [results["checkpoints"][str(s)]["cross_token_similarity"]["mean"]
                for s in checkpoints]
    sim_stds = [results["checkpoints"][str(s)]["cross_token_similarity"]["std"]
                for s in checkpoints]
    ax.errorbar(checkpoints, sim_vals, yerr=sim_stds, fmt='o-', capsize=3,
                color='tab:orange')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Cross-token spectral envelope similarity")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"dct_training_{model_key}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Cross-model comparison plot
# ---------------------------------------------------------------------------

def plot_cross_model(results_list: list, output_dir: str):
    """Compare metrics across model sizes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DCT Concentration: Model Size Comparison", fontsize=14)

    colors = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:red', '1b': 'tab:green', '1.4b': 'tab:purple'}

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
                     label=f"Pythia-{model_key}")
        axes[1].plot(checkpoints, cent_vals, 'o-', color=color,
                     label=f"Pythia-{model_key}")
        axes[2].plot(checkpoints, ent_vals, 'o-', color=color,
                     label=f"Pythia-{model_key}")

    for ax in axes:
        ax.set_xscale('symlog', linthresh=100)
        ax.set_xlabel("Training step")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("DC energy fraction")
    axes[0].set_title("DC concentration")
    axes[1].set_ylabel("Frequency centroid")
    axes[1].set_title("Spectral centroid")
    axes[2].set_ylabel("Normalized entropy")
    axes[2].set_title("Spectral entropy")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dct_training_cross_model.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Cross-model plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment C: DCT energy concentration across training")
    parser.add_argument("--models", nargs="+", default=["70m", "410m"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results/experiment_c")
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
