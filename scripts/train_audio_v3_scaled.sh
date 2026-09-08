#!/bin/bash
################################################################################
# Full 3-stage training with SCALED DATA (10x data scale-up)
#
# Data scale-up:
#   Stage 1: 10,094 samples (was 982) — VGGSound alignment
#   Stage 2: 20,188 samples (was 1,631) — VGGSound bridge
#   Stage 3: 10,384 samples (was 388) — AVQA + VGGSound instruction
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 bash scripts/train_audio_v3_scaled.sh
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/audio_v3"
VIDEO_DIR="data/videos"

echo "============================================"
echo "  AGTA v3: Scaled Training (10x data)"
echo "  $(date)"
echo "============================================"

# ------------------------------------------------------------------
# Stage 1: Alignment — projectors only, 10K samples, lr=1e-3, 3 epochs
# ------------------------------------------------------------------
echo ""
echo ">>> Stage 1: Alignment (VGGSound captions, 10094 samples, 3 epochs)"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/alignment/audiocaps_train_scaled.jsonl \
    --output_dir "${BASE_DIR}/stage1_alignment" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \
    --freeze_vision \
    --freeze_audio \
    --freeze_llm \
    --no-use_lora \
    --use_audio \
    --no-use_temporal_bridge \
    --mm_projector_type stc \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 1 complete: $(date)"

# ------------------------------------------------------------------
# Stage 2: Bridge — projectors + AGTA bridge, 20K samples, lr=2e-5, 3 epochs
# Reduced to 3 epochs (was 5) since data is 12x larger
# ------------------------------------------------------------------
echo ""
echo ">>> Stage 2: Bridge (VGGSound bridge, 20188 samples, 3 epochs)"
echo "    Bridge LR=1e-3, Projector LR=2e-5"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/bridge/vggsound_train_v3.jsonl \
    --output_dir "${BASE_DIR}/stage2_bridge" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bridge_lr 1e-3 \
    --warmup_ratio 0.03 \
    --freeze_vision \
    --freeze_audio \
    --freeze_llm \
    --no-use_lora \
    --use_audio \
    --use_temporal_bridge \
    --mm_projector_type stc \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 2 complete: $(date)"

# ------------------------------------------------------------------
# Stage 3: Instruction — + LoRA, AVQA+VGGSound QA, lr=2e-5, 3 epochs
# ------------------------------------------------------------------
echo ""
echo ">>> Stage 3: Instruction (AVQA+VGGSound QA, 10384 samples, 3 epochs)"
echo "    Bridge LR=5e-4, Base LR=2e-5"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_v3.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
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
    --logging_steps 20 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 3 complete: $(date)"
echo ""
echo "============================================"
echo "  All 3 stages complete! $(date)"
echo "  Checkpoints in: ${BASE_DIR}/"
echo "============================================"
