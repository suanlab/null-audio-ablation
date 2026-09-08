#!/bin/bash
################################################################################
# Quick Bridge v4: AGTA v2 + Audio Dropout + Reward-Weighted SFT
#
# Same architecture as QB v2/v3 (zero-init residual bridge), but with:
#   - audio_dropout_prob=0.3 (from Phase 1, same as QB v3)
#   - sample_weights from audio utility scoring (Phase 2)
#
# This teaches the model to:
#   1. Not over-rely on audio (dropout)
#   2. Focus learning on samples where audio genuinely helps (weighted SFT)
#
# Comparison targets:
#   QB v2: real=65.3%, noise=23.5%, silent=24.5% (no dropout, no weights)
#   QB v3: TBD (dropout only)
#   QB v4: dropout + weighted SFT (this run)
#
# Prerequisites:
#   - Run scripts/compute_audio_utility.sh first to generate weights
#   - Weights file: data/instruct/avqa_train_quick5k_weights.json
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 nohup bash scripts/train_quick_bridge_v4_weighted.sh > logs/train_quick_bridge_v4.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/audio_quick_bridge_v4"
QB_V2_S2="checkpoints/audio_quick_bridge_v2/stage2_bridge"
WEIGHTS_FILE="data/instruct/avqa_train_quick5k_weights.json"

# Verify weights file exists
if [ ! -f "${WEIGHTS_FILE}" ]; then
    echo "ERROR: Weights file not found: ${WEIGHTS_FILE}"
    echo "Run scripts/compute_audio_utility.sh first!"
    exit 1
fi

echo "============================================"
echo "  Quick Bridge v4 (dropout=0.3 + weighted SFT)"
echo "  $(date)"
echo "============================================"

# We skip Stage 2 — reuse QB v2's Stage 2 bridge checkpoint.
# Only Stage 3 (instruction) changes: audio dropout + sample weights

echo ""
echo ">>> Stage 3: Instruction (5K samples, 3 epochs, dropout=0.3, weighted)"
echo "    Loading bridge from ${QB_V2_S2}"
echo "    Weights from ${WEIGHTS_FILE}"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_quick5k.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --load_from "${QB_V2_S2}" \
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
    --report_to none \
    --audio_dropout_prob 0.3 \
    --sample_weights_path "${WEIGHTS_FILE}"

echo "    Stage 3 complete: $(date)"
echo ""
echo "============================================"
echo "  Quick Bridge v4 Complete! $(date)"
echo "  Checkpoints in: ${BASE_DIR}/"
echo "============================================"

# Auto-run 5-mode eval after training
echo ""
echo ">>> Running 5-mode AVQA evaluation..."
CKPT="${BASE_DIR}/stage3_instruction"
for mode in real shuffled shifted noise silent; do
    echo "  Evaluating mode=${mode}..."
    $PYTHON scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --audio_mode "${mode}" \
        --output "eval_results/qb_v4_${mode}.json"
done

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/qb_v4_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']
summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'audio_contribution_positive': scores['real'] > scores['noise'] + 5,
    'audio_dropout_prob': 0.3,
    'weighted_sft': True,
    'base_checkpoint': 'qb_v2_stage2',
    'weights_file': '${WEIGHTS_FILE}',
}
with open('eval_results/qb_v4_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
lines = [
    '# QB v4 Audio Ablation (dropout=0.3 + weighted SFT)',
    '',
]
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open('eval_results/qb_v4_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/qb_v4_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  QB v4 Eval Complete! $(date)"
echo "============================================"
