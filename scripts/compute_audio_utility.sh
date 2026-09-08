#!/bin/bash
################################################################################
# Compute audio utility scores for Reward-Weighted SFT (Phase 2)
#
# Runs QB v2 inference on training data in both "real" and "silent" modes.
# Results are used to compute per-sample audio utility weights.
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 nohup bash scripts/compute_audio_utility.sh > logs/compute_audio_utility.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
CKPT="checkpoints/audio_quick_bridge_v2/stage3_instruction"
TRAIN_DATA="data/instruct/avqa_train_quick5k.jsonl"

echo "============================================"
echo "  Audio Utility Scoring (QB v2 on training data)"
echo "  $(date)"
echo "============================================"

echo ""
echo ">>> Scoring mode=real on $(wc -l < ${TRAIN_DATA}) samples..."
echo "    Start: $(date)"
$PYTHON scripts/eval_avqa_fast.py \
    --checkpoint "${CKPT}" \
    --eval_data "${TRAIN_DATA}" \
    --audio_mode real \
    --output eval_results/train_utility_real.json
echo "    Done: $(date)"

echo ""
echo ">>> Scoring mode=silent on $(wc -l < ${TRAIN_DATA}) samples..."
echo "    Start: $(date)"
$PYTHON scripts/eval_avqa_fast.py \
    --checkpoint "${CKPT}" \
    --eval_data "${TRAIN_DATA}" \
    --audio_mode silent \
    --output eval_results/train_utility_silent.json
echo "    Done: $(date)"

echo ""
echo ">>> Computing per-sample weights..."
$PYTHON scripts/compute_sample_weights.py \
    --real_results eval_results/train_utility_real.json \
    --silent_results eval_results/train_utility_silent.json \
    --output data/instruct/avqa_train_quick5k_weights.json
echo "    Done: $(date)"

echo ""
echo "============================================"
echo "  Audio Utility Scoring Complete! $(date)"
echo "============================================"
