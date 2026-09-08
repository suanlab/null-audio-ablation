#!/bin/bash
################################################################################
# Stage 1: Audio-Visual Alignment
#
# Objective: Train ONLY the audio-visual projectors to align CLAP/SigLIP embeddings
# with the LLM input space.
#
# Configuration:
#   - Freeze: vision encoder, audio encoder, LLM
#   - Train: MM projectors only
#   - LoRA: Disabled (not needed for projector-only training)
#   - Learning rate: 1e-3 (higher for projector initialization)
#   - Epochs: 1 (quick alignment pass)
#   - Batch size: 4 per GPU
#   - Gradient accumulation: 4 steps
#
# Usage:
#   export DATA_PATH="data/alignment/audiocaps_train.jsonl"
#   export VIDEO_DIR="data/videos"
#   bash scripts/train_stage1_alignment.sh
################################################################################

set -euo pipefail

# Environment variables with defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-7B-Instruct}"
DATA_PATH="${DATA_PATH:?ERROR: DATA_PATH is required}"
VIDEO_DIR="${VIDEO_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/videollm-7b-stage1}"
NUM_GPUS="${NUM_GPUS:-8}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Stage 1: Audio-Visual Alignment Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Video Dir: $VIDEO_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo ""

deepspeed --num_gpus="$NUM_GPUS" \
    src/videollm/train.py \
    --config configs/model/train_7b.yaml \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --freeze_vision \
    --freeze_audio \
    --freeze_llm \
    --no-use_lora \
    --use_audio \
    --no-use_temporal_bridge \
    --mm_projector_type stc \
    --deepspeed configs/ds_zero2.json \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb

echo ""
echo "Stage 1 training complete. Checkpoint saved to: $OUTPUT_DIR"
