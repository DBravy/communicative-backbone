#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${1:-cuda}"

echo "=== OLMo 2 1B: Full experiment pipeline (device=$DEVICE) ==="
echo ""

echo "[1/4] Experiment A: SVD Bulk-Tail Emergence"
python "$DIR/experiment_a_svd_emergence_olmo.py" --models 1b --device "$DEVICE"
echo ""

echo "[2/4] Experiment B: Cross-Layer Subspace Coherence"
python "$DIR/experiment_b_crosslayer_overlap_olmo.py" --models 1b
echo ""

echo "[3/4] Experiment C: DCT Energy Concentration"
python "$DIR/experiment_c_dct_training_olmo.py" --models 1b --device "$DEVICE"
echo ""

echo "[4/4] Pairwise Overlap Matrices"
python "$DIR/compute_pairwise_overlap_olmo.py" --models 1b
echo ""

echo "=== All experiments complete ==="
