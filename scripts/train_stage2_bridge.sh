#!/bin/bash
################################################################################
# Stage 2: Audio-Visual Temporal Bridge Pre-training
#
# Objective: Train the temporal bridge module (e.g., STC connector) to learn
# spatiotemporal relationships between audio and visual features.
#
# Configuration:
#   - Freeze: vision encoder, audio encoder, LLM
#   - Train: MM projectors + temporal bridge
#   - LoRA: Disabled (bridge is already lightweight)
#   - Learning rate: 2e-5 (standard pre-training rate)
#   - Epochs: 1 (contrastive pre-training pass)
#   - Batch size: 4 per GPU
#   - Gradient accumulation: 4 steps
#   - Load from: Stage 1 checkpoint
#
# Usage:
#   export DATA_PATH="data/bridge/vggsound_train.jsonl"
#   export VIDEO_DIR="data/videos"
#   export STAGE1_CKPT="./checkpoints/videollm-7b-stage1"
#   bash scripts/train_stage2_bridge.sh
################################################################################

set -euo pipefail

# Environment variables with defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-7B-Instruct}"
DATA_PATH="${DATA_PATH:?ERROR: DATA_PATH is required}"
VIDEO_DIR="${VIDEO_DIR:-}"
STAGE1_CKPT="${STAGE1_CKPT:-./checkpoints/videollm-7b-stage1}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/videollm-7b-stage2}"
NUM_GPUS="${NUM_GPUS:-8}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Stage 2: Temporal Bridge Pre-training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Video Dir: $VIDEO_DIR"
echo "Stage 1 Checkpoint: $STAGE1_CKPT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo ""

deepspeed --num_gpus="$NUM_GPUS" \
    src/videollm/train.py \
    --config configs/model/train_7b.yaml \
    --model_path "$STAGE1_CKPT" \
    --data_path "$DATA_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --freeze_vision \
    --freeze_audio \
    --freeze_llm \
    --no-use_lora \
    --use_audio \
    --use_temporal_bridge \
    --mm_projector_type stc \
    --deepspeed configs/ds_zero2.json \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb

echo ""
echo "Stage 2 training complete. Checkpoint saved to: $OUTPUT_DIR"
