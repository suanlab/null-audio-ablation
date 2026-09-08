#!/bin/bash
################################################################################
# Quick Bridge Validation Experiment (GPU 2)
#
# Start from v2 Stage 1 projectors, retrain only Stage 2+3 with:
#   - bridge_lr=1e-3 (50x higher than before)
#   - Audio-only training examples  
#   - Smaller dataset for faster iteration (~2-4 hours total)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 bash scripts/train_quick_bridge_test.sh
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/audio_quick_bridge"
VIDEO_DIR="data/videos"

echo "============================================"
echo "  Quick Bridge Validation (GPU 2)"
echo "  $(date)"
echo "============================================"

# Stage 2: Bridge with high LR — 5000 samples, 5 epochs (~1-2 hours)
echo ""
echo ">>> Stage 2: Bridge (5K samples, 5 epochs, bridge_lr=1e-3)"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/bridge/vggsound_train_quick5k.jsonl \
    --output_dir "${BASE_DIR}/stage2_bridge" \
    --num_epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --bridge_lr 1e-3 \
    --warmup_ratio 0.05 \
    --freeze_vision \
    --freeze_audio \
    --freeze_llm \
    --no-use_lora \
    --use_audio \
    --use_temporal_bridge \
    --mm_projector_type stc \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 2 complete: $(date)"

# Stage 3: Instruction with LoRA — 5000 samples, 3 epochs (~1-2 hours)
echo ""
echo ">>> Stage 3: Instruction (5K samples, 3 epochs, bridge_lr=5e-4)"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_quick5k.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --bridge_lr 5e-4 \
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
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 3 complete: $(date)"
echo ""
echo "============================================"
echo "  Quick Bridge Test Complete! $(date)"
echo "  Checkpoints in: ${BASE_DIR}/"
echo "============================================"
