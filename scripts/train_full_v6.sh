#!/bin/bash
################################################################################
# Full-Scale v6: AGTA v2 + Audio Dropout (25K dataset)
#
# Same architecture and hyperparameters as QB v4, but on the full 25K dataset.
# Uses audio_dropout_prob=0.3 (proven effective in QB v3/v4).
# No sample weights — too expensive to compute for 25K.
#
# Base: QB v2 Stage 2 bridge (zero-init residual, AGTA v2)
# Data: avqa_train_v3.jsonl (25,371 samples)
# Expected time: ~16-20 hours on 1x A100 80GB (3 epochs)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_full_v6.sh > logs/train_full_v6.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/full_v6"
QB_V2_S2="checkpoints/audio_quick_bridge_v2/stage2_bridge"

echo "============================================"
echo "  Full-Scale v6 (25K, dropout=0.3)"
echo "  $(date)"
echo "============================================"

echo ""
echo ">>> Stage 3: Instruction (25K samples, 3 epochs, dropout=0.3)"
echo "    Loading bridge from ${QB_V2_S2}"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_v3.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --load_from "${QB_V2_S2}" \
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
    --logging_steps 50 \
    --save_total_limit 2 \
    --dataloader_num_workers 0 \
    --report_to none \
    --audio_dropout_prob 0.3

echo "    Stage 3 complete: $(date)"

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
        --output "eval_results/full_v6_${mode}.json"
done

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/full_v6_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']
summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'audio_contribution_positive': scores['real'] > scores['noise'] + 5,
    'training_phase': 'full_scale_sft',
    'dataset': 'avqa_train_v3.jsonl (25K)',
    'audio_dropout_prob': 0.3,
    'base_checkpoint': 'qb_v2_stage2',
}
with open('eval_results/full_v6_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
lines = ['# Full v6 Results (25K dataset)', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open('eval_results/full_v6_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/full_v6_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  Full-Scale v6 Complete! $(date)"
echo "============================================"
