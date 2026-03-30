# Communication Before Computation: The Development of Communicative Structure in Transformer Learning

Code and data for reproducing the experiments in:

> Delbert Bray Jr. "Communication Before Computation: The Development of Communicative Structure in Transformer Learning."

## Overview

This repository contains experiments that track the emergence of geometric structure in transformer MLP weights during training. We analyze how singular value gaps, cross-layer subspace alignment, gate selectivity, and frequency-domain trajectory structure develop across training checkpoints in four model families:

| Model Family | Sizes | Activation | Source |
|---|---|---|---|
| **Pythia** | 70M, 160M, 410M, 1B, 1.4B | GELU | EleutherAI |
| **OLMo** | 1B | SwiGLU | AI2 |
| **BLOOM** | 1.1B | GELU (sequential) | BigScience |
| **TinyLlama** | 1.1B | SwiGLU | Zhang et al. |

## Repository Structure

```
.
├── pythia/          # Pythia experiments (primary model family)
│   ├── experiment_a_svd_emergence.py        # SVD bulk-tail gap emergence
│   ├── experiment_b_crosslayer_overlap.py   # Cross-layer subspace coherence
│   ├── experiment_c_dct_training.py         # DCT energy concentration
│   ├── experiment_attention_ov_timing.py    # OV circuit timing
│   ├── compute_pairwise_overlap.py          # Full NxN overlap matrices
│   ├── compute_crosscov_pythia.py           # MLP cross-covariance spectra
│   ├── figures/                             # Figure generation scripts
│   └── results/                             # Pre-computed results (JSON)
│
├── olmo/                  # OLMo-1B experiments
│   ├── experiment_a_svd_emergence_olmo.py
│   ├── experiment_b_crosslayer_overlap_olmo.py
│   ├── experiment_c_dct_training_olmo.py
│   ├── experiment_gate_crosslayer_olmo.py   # Gate coherence analysis
│   ├── experiment_gate_selectivity_olmo.py  # Gate selectivity tracking
│   ├── experiment_attention_ov_timing_olmo.py
│   ├── experiment_jacobian_crosslayer_olmo.py
│   ├── compute_crosscov_olmo.py
│   ├── figures/                             # Figure generation scripts
│   └── *.json                               # Pre-computed results
│
├── llama/           # TinyLlama-1.1B experiments
│   ├── experiment_pairwise_overlap_tinyllama.py
│   ├── experiment_b_crosslayer_overlap_tinyllama.py
│   ├── experiment_gate_crosslayer_tinyllama.py
│   ├── experiment_gate_selectivity_tinyllama.py
│   ├── run_tinyllama_experiments.py         # Sequential runner
│   ├── figures/                             # Figure generation scripts
│   └── *.json                               # Pre-computed results
│
├── bloom/                 # BLOOM-1.1B experiments
│   ├── bloom_experiments.py
│   └── bloom_1b1_experiments.json
│
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

For OLMo experiments, also install:
```bash
pip install ai2-olmo
```

Optionally set `HF_HOME` to control where Hugging Face model checkpoints are cached:
```bash
export HF_HOME=/path/to/your/cache
```

## Reproducing Results

### Regenerate figures from included results

Each model directory contains a `figures/` subdirectory with plotting scripts. These read from the included JSON result files:

```bash
cd pythia_chain/figures
python fig_svd_spectrum.py
python fig_pairwise_heatmap.py
python fig_crosslayer_coherence.py
# etc.
```

### Re-run experiments from scratch

Each experiment script accepts command-line arguments. For example:

```bash
# Pythia SVD emergence (all sizes)
python pythia_chain/experiment_a_svd_emergence.py --models 70m 160m 410m 1b 1.4b

# Pythia cross-layer coherence
python pythia_chain/experiment_b_crosslayer_overlap.py --models 70m 410m

# Pythia DCT analysis
python pythia_chain/experiment_c_dct_training.py --models 70m 410m

# OLMo experiments (run all)
cd olmo && bash run_all.sh

# TinyLlama experiments (run all)
python crossmodels/run_tinyllama_experiments.py

# BLOOM experiments
python bloom/bloom_experiments.py
```

Scripts automatically download model checkpoints from Hugging Face Hub. GPU is recommended but MPS (Apple Silicon) and CPU are supported.

## Pre-computed Results

All JSON result files are included so figures can be regenerated without re-running experiments. These contain serialized numpy arrays of SVD spectra, overlap matrices, gate statistics, and DCT energy profiles across all training checkpoints.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@article{bray2026communication,
  title={Communication Before Computation: The Development of Communicative Structure in Transformer Learning},
  author={Bray, Delbert Jr.},
  year={2026}
}
```
