#!/bin/bash
################################################################################
# Stage 3: Instruction Tuning
#
# Objective: Fine-tune the model on instruction-following tasks using LoRA
# to adapt the LLM for video-audio understanding while keeping encoders frozen.
#
# Configuration:
#   - Freeze: vision encoder, audio encoder
#   - Train: MM projectors + temporal bridge + LLM (via LoRA)
#   - LoRA: Enabled (r=64, alpha=128)
#   - Learning rate: 2e-5 (standard instruction tuning rate)
#   - Epochs: 3 (multiple passes for convergence)
#   - Batch size: 4 per GPU
#   - Gradient accumulation: 4 steps
#   - Load from: Stage 2 checkpoint
#
# Usage:
#   export DATA_PATH="data/instruct/avqa_train.jsonl"
#   export VIDEO_DIR="data/videos"
#   export STAGE2_CKPT="./checkpoints/videollm-7b-stage2"
#   bash scripts/train_stage3_instruction.sh
################################################################################

set -euo pipefail

# Environment variables with defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-7B-Instruct}"
DATA_PATH="${DATA_PATH:?ERROR: DATA_PATH is required}"
VIDEO_DIR="${VIDEO_DIR:-}"
STAGE2_CKPT="${STAGE2_CKPT:-./checkpoints/videollm-7b-stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/videollm-7b-stage3}"
NUM_GPUS="${NUM_GPUS:-8}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Stage 3: Instruction Tuning (LoRA)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Video Dir: $VIDEO_DIR"
echo "Stage 2 Checkpoint: $STAGE2_CKPT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo ""

deepspeed --num_gpus="$NUM_GPUS" \
    src/videollm/train.py \
    --config configs/model/train_7b.yaml \
    --model_path "$STAGE2_CKPT" \
    --data_path "$DATA_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --freeze_vision \
    --freeze_audio \
    --no-freeze_llm \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --use_audio \
    --use_temporal_bridge \
    --mm_projector_type stc \
    --deepspeed configs/ds_zero2.json \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb

echo ""
echo "Stage 3 training complete. Checkpoint saved to: $OUTPUT_DIR"
