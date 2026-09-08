#!/bin/bash
################################################################################
# Stage 4: Direct Preference Optimization (DPO)
#
# Objective: Fine-tune the model using DPO to align model outputs with human
# preferences on video-audio understanding tasks.
#
# Configuration:
#   - Freeze: vision encoder, audio encoder
#   - Train: MM projectors + temporal bridge + LLM (via LoRA)
#   - LoRA: Enabled (r=64, alpha=128)
#   - DPO beta: 0.1 (configurable via DPO_BETA env var)
#   - Learning rate: 2e-5
#   - Epochs: 1 (DPO typically requires fewer epochs)
#   - Batch size: 4 per GPU
#   - Gradient accumulation: 4 steps
#   - Load from: Stage 3 checkpoint
#
# Data format (JSONL):
#   {"video": "path.mp4", "prompt": "<video>\n<audio>\nDescribe.", "chosen": "Good answer", "rejected": "Bad answer"}
#
# Usage:
#   export DATA_PATH="data/dpo/preferences.jsonl"
#   export VIDEO_DIR="data/videos"
#   export STAGE3_CKPT="./checkpoints/videollm-7b-stage3"
#   bash scripts/train_stage4_dpo.sh
################################################################################

set -euo pipefail

# Environment variables with defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-7B-Instruct}"
DATA_PATH="${DATA_PATH:?ERROR: DATA_PATH is required}"
VIDEO_DIR="${VIDEO_DIR:-}"
STAGE3_CKPT="${STAGE3_CKPT:-./checkpoints/videollm-7b-stage3}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/videollm-7b-stage4}"
NUM_GPUS="${NUM_GPUS:-8}"
DPO_BETA="${DPO_BETA:-0.1}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Stage 4: Direct Preference Optimization"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Video Dir: $VIDEO_DIR"
echo "Stage 3 Checkpoint: $STAGE3_CKPT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "DPO Beta: $DPO_BETA"
echo ""

deepspeed --num_gpus="$NUM_GPUS" \
    src/videollm/train.py \
    --config configs/model/train_7b.yaml \
    --model_path "$STAGE3_CKPT" \
    --data_path "$DATA_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --use_dpo \
    --dpo_beta "$DPO_BETA" \
    --dpo_loss_type sigmoid \
    --num_epochs 1 \
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
echo "Stage 4 DPO training complete. Checkpoint saved to: $OUTPUT_DIR"
