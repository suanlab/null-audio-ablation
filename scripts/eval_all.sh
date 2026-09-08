#!/bin/bash
################################################################################
# Evaluation Script: Run VideoLLM on All Benchmarks
#
# Objective: Evaluate a trained VideoLLM checkpoint on all supported benchmarks:
#   - video_mme: Comprehensive video understanding (900 videos, 2700 QA pairs)
#   - mvbench: Multi-task video QA (20 tasks)
#   - avqa: Audio-Visual Question Answering
#   - tempcompass: Temporal reasoning evaluation
#
# Configuration:
#   - Batch size: 1 (per-sample inference for accuracy)
#   - Max new tokens: 256 (sufficient for multiple-choice answers)
#   - Device: cuda (GPU inference)
#   - Output: JSON results file with per-benchmark metrics
#
# Usage:
#   export MODEL_PATH="./checkpoints/videollm-7b-stage3"
#   bash scripts/eval_all.sh
#
# Or with custom data directory:
#   export MODEL_PATH="./checkpoints/videollm-7b-stage3"
#   export DATA_DIR="data/benchmarks"
#   bash scripts/eval_all.sh
################################################################################

set -euo pipefail

# Environment variables with defaults
MODEL_PATH="${MODEL_PATH:?ERROR: MODEL_PATH is required}"
DATA_DIR="${DATA_DIR:-data/benchmarks}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "VideoLLM Benchmark Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run evaluation on all benchmarks
python -m eval.run \
    --model_path "$MODEL_PATH" \
    --benchmark all \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --max_new_tokens 256 \
    --device cuda

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/results.json"
echo "=========================================="
