#!/bin/bash
################################################################################
# Quick Bridge v5: GRPO RL Training (Phase 3)
#
# Runs GRPO reinforcement learning on top of the best SFT checkpoint.
# Uses audio dropout during rollouts: half with real audio, half with silence.
# Reward function: correctness + format + audio robustness bonus.
#
# Prerequisites:
#   - Best SFT checkpoint (QB v4 or QB v3)
#   - trl >= 0.29.0 installed
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 nohup bash scripts/train_quick_bridge_v5_grpo.sh > logs/train_quick_bridge_v5_grpo.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"

# Use best available SFT checkpoint (QB v4 > QB v3 > QB v2)
if [ -f "checkpoints/audio_quick_bridge_v4/stage3_instruction/model.safetensors" ]; then
    SFT_CKPT="checkpoints/audio_quick_bridge_v4/stage3_instruction"
elif [ -f "checkpoints/audio_quick_bridge_v3/stage3_instruction/model.safetensors" ]; then
    SFT_CKPT="checkpoints/audio_quick_bridge_v3/stage3_instruction"
else
    SFT_CKPT="checkpoints/audio_quick_bridge_v2/stage3_instruction"
fi

echo "============================================"
echo "  Quick Bridge v5: GRPO RL (Phase 3)"
echo "  SFT checkpoint: ${SFT_CKPT}"
echo "  $(date)"
echo "============================================"

$PYTHON scripts/train_grpo.py \
    --model_path "${SFT_CKPT}" \
    --data_path data/instruct/avqa_train_clean.jsonl \
    --video_dir data/videos \
    --output_dir checkpoints/audio_quick_bridge_v5_grpo \
    --num_epochs 1 \
    --batch_size 1 \
    --num_generations 8 \
    --max_completion_length 32 \
    --learning_rate 1e-6 \
    --beta 0.0 \
    --epsilon 0.2 \
    --audio_dropout_ratio 0.5 \
    --use_lora \
    --bf16

echo "    GRPO training complete: $(date)"

# Auto-run eval on the final checkpoint
echo ""
echo ">>> Running 5-mode AVQA evaluation on GRPO model..."
CKPT="checkpoints/audio_quick_bridge_v5_grpo/final"
for mode in real shuffled shifted noise silent; do
    echo "  Evaluating mode=${mode}..."
    $PYTHON scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --audio_mode "${mode}" \
        --output "eval_results/qb_v5_grpo_${mode}.json"
done

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/qb_v5_grpo_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']
summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'audio_contribution_positive': scores['real'] > scores['noise'] + 5,
    'training_phase': 'grpo_rl',
    'sft_checkpoint': '${SFT_CKPT}',
}
with open('eval_results/qb_v5_grpo_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
lines = ['# QB v5 GRPO RL Results', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open('eval_results/qb_v5_grpo_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/qb_v5_grpo_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  QB v5 GRPO Complete! $(date)"
echo "============================================"
