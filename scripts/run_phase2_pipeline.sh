#!/bin/bash
################################################################################
# Phase 2 Pipeline: Score → Compute Weights → Train QB v4 → Eval
# Runs entirely on one GPU, sequentially.
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_phase2_pipeline.sh > logs/phase2_pipeline.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"

echo "============================================"
echo "  Phase 2 Full Pipeline"
echo "  $(date)"
echo "============================================"

# Step 1: Compute audio utility scores (if not already done)
WEIGHTS_FILE="data/instruct/avqa_train_quick5k_weights.json"
if [ -f "${WEIGHTS_FILE}" ]; then
    echo ">>> Weights file already exists, skipping scoring..."
else
    echo ">>> Step 1: Running audio utility scoring..."
    bash scripts/compute_audio_utility.sh
fi

# Step 2: Train QB v4 with weighted SFT
echo ""
echo ">>> Step 2: Training QB v4 (dropout + weighted SFT)..."
bash scripts/train_quick_bridge_v4_weighted.sh

echo ""
echo "============================================"
echo "  Phase 2 Pipeline Complete! $(date)"
echo "============================================"
