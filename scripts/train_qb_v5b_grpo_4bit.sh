#!/bin/bash
################################################################################
# QB v5b: GRPO RL with 4-bit quantization (QLoRA mode)
#
# Uses 4-bit NF4 quantization for the LLM backbone to reduce memory from
# ~36 GiB to ~15 GiB, allowing training on contested GPUs.
#
# Hyperparameters same as v5b:
#   - beta=0.04 (KL penalty)
#   - lr=5e-7
#   - G=4 (reduced for memory)
#   - 4-bit NF4 quantization (QLoRA)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_qb_v5b_grpo_4bit.sh > logs/train_qb_v5b_grpo.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
SFT_CKPT="checkpoints/audio_quick_bridge_v4/stage3_instruction"

echo "============================================"
echo "  QB v5b: GRPO RL (4-bit QLoRA, beta=0.04)"
echo "  SFT checkpoint: ${SFT_CKPT}"
echo "  $(date)"
echo "============================================"

$PYTHON scripts/train_grpo.py \
    --model_path "${SFT_CKPT}" \
    --data_path data/instruct/avqa_train_clean.jsonl \
    --video_dir data/videos \
    --output_dir checkpoints/audio_quick_bridge_v5b_grpo \
    --num_epochs 1 \
    --batch_size 1 \
    --num_generations 4 \
    --max_completion_length 32 \
    --learning_rate 5e-7 \
    --beta 0.04 \
    --epsilon 0.2 \
    --audio_dropout_ratio 0.5 \
    --use_lora \
    --bf16 \
    --load_in_4bit

echo "    GRPO training complete: $(date)"

# Auto-run eval (eval uses full-precision model from checkpoint, not 4-bit)
echo ""
echo ">>> Running 5-mode AVQA evaluation..."
CKPT="checkpoints/audio_quick_bridge_v5b_grpo/final"
for mode in real shuffled shifted noise silent; do
    echo "  Evaluating mode=${mode}..."
    $PYTHON scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --audio_mode "${mode}" \
        --output "eval_results/qb_v5b_grpo_${mode}.json"
done

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/qb_v5b_grpo_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']
summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'audio_contribution_positive': scores['real'] > scores['noise'] + 5,
    'training_phase': 'grpo_rl_4bit_g4',
    'sft_checkpoint': '${SFT_CKPT}',
    'beta': 0.04,
    'lr': 5e-7,
    'num_generations': 4,
    'quantization': '4bit_nf4_double_quant',
}
with open('eval_results/qb_v5b_grpo_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
lines = ['# QB v5b GRPO RL Results (4-bit QLoRA, beta=0.04, lr=5e-7, G=4)', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open('eval_results/qb_v5b_grpo_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/qb_v5b_grpo_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  QB v5b GRPO Complete! $(date)"
echo "============================================"
