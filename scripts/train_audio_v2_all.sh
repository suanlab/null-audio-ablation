#!/bin/bash
################################################################################
# Full 3-stage training with REAL AUDIO (PyAV-based loading)
#
# This retrains all stages from scratch with actual audio waveforms
# (not silence) using the PyAV audio loading fix.
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 bash scripts/train_audio_v2_all.sh
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/audio_v2"
VIDEO_DIR="data/videos"

echo "============================================"
echo "  AGTA Full Retraining with Real Audio"
echo "  $(date)"
echo "============================================"

# ------------------------------------------------------------------
# Stage 1: Alignment — projectors only, AudioCaps, lr=1e-3, 3 epochs
# ------------------------------------------------------------------
echo ""
echo ">>> Stage 1: Alignment (AudioCaps, 982 samples, 3 epochs)"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/alignment/audiocaps_train.jsonl \
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
    --logging_steps 5 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 1 complete: $(date)"

# ------------------------------------------------------------------
# Stage 2: Bridge — projectors + AGTA bridge, VGGSound, lr=2e-5, 5 epochs
# ------------------------------------------------------------------
echo ""
echo ">>> Stage 2: Bridge (VGGSound, 1631 samples, 5 epochs)"
echo "    Start: $(date)"

# Copy stage1 checkpoint as starting point — train.py uses --model_path for LLM base
# We load stage1 weights via safetensors
$PYTHON src/videollm/train.py \
    --data_path data/bridge/vggsound_train.jsonl \
    --output_dir "${BASE_DIR}/stage2_bridge" \
    --num_epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
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
    --logging_steps 10 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 2 complete: $(date)"

# ------------------------------------------------------------------
# Stage 3: Instruction — + LoRA, AVQA, lr=2e-5, 3 epochs
# ------------------------------------------------------------------
echo ""
echo ">>> Stage 3: Instruction Tuning (AVQA, 388 samples, 3 epochs)"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
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
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 3 complete: $(date)"
echo ""
echo "============================================"
echo "  All 3 stages complete! $(date)"
echo "  Checkpoints in: ${BASE_DIR}/"
echo "============================================"
