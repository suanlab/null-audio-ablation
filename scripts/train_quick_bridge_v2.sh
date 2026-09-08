#!/bin/bash
################################################################################
# Quick Bridge v2: AGTA with zero-init residual design
#
# Key changes from v1:
#   - Zero-init residual bridge (alpha=0, out_proj=0 at init)
#   - No TemporalImportance (was constant 0.5)
#   - No GatedFusion (gate was saturated)
#   - No output LayerNorm (was creating 70x scale mismatch)
#   - Loads v3 Stage 1 projector weights (pre-trained STC + audio projector)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 nohup bash scripts/train_quick_bridge_v2.sh > logs/train_quick_bridge_v2.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/audio_quick_bridge_v2"
V3_S1="checkpoints/audio_v3/stage1_alignment"

echo "============================================"
echo "  Quick Bridge v2 (zero-init residual)"
echo "  $(date)"
echo "============================================"

# Stage 2: Bridge + projectors, load v3 S1 projectors as starting point
echo ""
echo ">>> Stage 2: Bridge (5K samples, 5 epochs, bridge_lr=1e-3)"
echo "    Loading pre-trained projectors from ${V3_S1}"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/bridge/vggsound_train_quick5k.jsonl \
    --output_dir "${BASE_DIR}/stage2_bridge" \
    --load_from "${V3_S1}" \
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

# Stage 3: Instruction with LoRA, load from our Stage 2
echo ""
echo ">>> Stage 3: Instruction (5K samples, 3 epochs, bridge_lr=5e-4)"
echo "    Loading from ${BASE_DIR}/stage2_bridge"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_quick5k.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --load_from "${BASE_DIR}/stage2_bridge" \
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
echo "  Quick Bridge v2 Complete! $(date)"
echo "  Checkpoints in: ${BASE_DIR}/"
echo "============================================"
