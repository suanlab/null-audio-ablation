#!/bin/bash
################################################################################
# Resume v6 Full Training from checkpoint-3172 (epoch 2)
# Remaining: ~1586 steps (epoch 3)
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/full_v6"
QB_V2_S2="checkpoints/audio_quick_bridge_v2/stage2_bridge"
RESUME_CKPT="${BASE_DIR}/stage3_instruction/checkpoint-3172"

echo "============================================"
echo "  v6 Resume from checkpoint-3172"
echo "  $(date)"
echo "============================================"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_v3.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --load_from "${QB_V2_S2}" \
    --resume_from_checkpoint "${RESUME_CKPT}" \
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

echo "    v6 Training complete: $(date)"

# Auto-run 5-mode eval
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
print('\n'.join(lines))
"

echo "============================================"
echo "  v6 Complete! $(date)"
echo "============================================"
