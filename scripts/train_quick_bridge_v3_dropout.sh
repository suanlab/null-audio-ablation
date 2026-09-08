#!/bin/bash
################################################################################
# Quick Bridge v3: AGTA v2 + Audio Dropout (30%)
#
# Same architecture as QB v2 (zero-init residual bridge), but with
# audio_dropout_prob=0.3 during Stage 3 instruction tuning.
# This teaches the model to not over-rely on audio.
#
# Comparison target: QB v2 (no dropout)
#   QB v2 results: real=65.3%, noise=23.5%, silent=24.5%
#   Problem: model collapses without audio (below random chance)
#   Goal: real stays high, noise/silent rise to reasonable baseline
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_quick_bridge_v3_dropout.sh > logs/train_quick_bridge_v3.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/audio_quick_bridge_v3"
QB_V2_S2="checkpoints/audio_quick_bridge_v2/stage2_bridge"

echo "============================================"
echo "  Quick Bridge v3 (audio dropout 30%)"
echo "  $(date)"
echo "============================================"

# We skip Stage 2 — reuse QB v2's Stage 2 bridge checkpoint.
# Only Stage 3 (instruction) changes: audio_dropout_prob=0.3

echo ""
echo ">>> Stage 3: Instruction (5K samples, 3 epochs, audio_dropout=0.3)"
echo "    Loading bridge from ${QB_V2_S2}"
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
    --audio_dropout_prob 0.3

echo "    Stage 3 complete: $(date)"
echo ""
echo "============================================"
echo "  Quick Bridge v3 Complete! $(date)"
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
        --output "eval_results/qb_v3_${mode}.json"
done

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/qb_v3_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']
summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'audio_contribution_positive': scores['real'] > scores['noise'] + 5,
    'audio_dropout_prob': 0.3,
    'base_checkpoint': 'qb_v2_stage2',
}
with open('eval_results/qb_v3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
lines = [
    '# QB v3 Audio Ablation (dropout=0.3)',
    '',
]
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open('eval_results/qb_v3_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/qb_v3_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  QB v3 Eval Complete! $(date)"
echo "============================================"
