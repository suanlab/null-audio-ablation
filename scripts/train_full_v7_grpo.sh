#!/bin/bash
################################################################################
# Full-Scale v7: GRPO RL on v6 checkpoint (4-bit QLoRA)
#
# Key learnings from v5/v5b:
#   - GRPO on 290 samples = catastrophic forgetting (R-S gap → -2%)
#   - 4-bit QLoRA needed for GPU memory (saves ~20 GiB)
#   - beta=0.04 KL penalty not enough for tiny datasets
#
# v7 improvements:
#   - Uses 2000 samples (10x more than v5b) for more stable RL
#   - 4-bit NF4 quantization (fits alongside other GPU processes)
#   - Merged checkpoint for eval compatibility
#   - Uses fast logit eval (5-8x faster)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_full_v7_grpo.sh > logs/train_full_v7_grpo.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"

# Use full-scale v6 checkpoint if available, else fall back to QB v4
if [ -f "checkpoints/full_v6/stage3_instruction/model.safetensors" ]; then
    SFT_CKPT="checkpoints/full_v6/stage3_instruction"
elif [ -f "checkpoints/audio_quick_bridge_v4/stage3_instruction/model.safetensors" ]; then
    SFT_CKPT="checkpoints/audio_quick_bridge_v4/stage3_instruction"
else
    echo "ERROR: No SFT checkpoint found"
    exit 1
fi

# Use first 2000 samples from full dataset for GRPO (25K is too much for RL)
GRPO_DATA="data/instruct/avqa_train_v3.jsonl"
if [ ! -f "${GRPO_DATA}" ]; then
    GRPO_DATA="data/instruct/avqa_train_clean.jsonl"
fi

# Create 2K subset for GRPO if not exists
GRPO_SUBSET="data/instruct/avqa_grpo_2k.jsonl"
if [ ! -f "${GRPO_SUBSET}" ]; then
    echo "Creating 2K GRPO subset..."
    head -2000 "${GRPO_DATA}" > "${GRPO_SUBSET}"
fi

echo "============================================"
echo "  Full-Scale v7: GRPO RL (4-bit QLoRA)"
echo "  SFT checkpoint: ${SFT_CKPT}"
echo "  GRPO data: ${GRPO_SUBSET} (2000 samples)"
echo "  $(date)"
echo "============================================"

$PYTHON scripts/train_grpo.py \
    --model_path "${SFT_CKPT}" \
    --data_path "${GRPO_SUBSET}" \
    --video_dir data/videos \
    --output_dir checkpoints/full_v7_grpo \
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

# Step 2: Create eval-compatible merged checkpoint
echo ""
echo ">>> Merging checkpoint for eval..."
$PYTHON -c "
from safetensors.torch import load_file, save_file
# Load the SFT base (bf16)
base = load_file('${SFT_CKPT}/model.safetensors')
# Load GRPO checkpoint, filter out quantized base LLM weights
grpo = load_file('checkpoints/full_v7_grpo/final/model.safetensors')
filtered = {k: v for k, v in grpo.items()
            if not ('llm' in k and 'base_layer' in k)}
merged = dict(base)
for k, v in filtered.items():
    merged[k] = v
save_file(merged, 'checkpoints/full_v7_grpo/final/model.safetensors')
print(f'Merged: base={len(base)}, grpo_filtered={len(filtered)}, final={len(merged)}')
"

# Step 3: Auto-run 5-mode eval
echo ""
echo ">>> Running 5-mode AVQA evaluation..."
CKPT="checkpoints/full_v7_grpo/final"
for mode in real shuffled shifted noise silent; do
    echo "  Evaluating mode=${mode}..."
    $PYTHON scripts/eval_avqa_fast.py \
        --checkpoint "${CKPT}" \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --audio_mode "${mode}" \
        --output "eval_results/full_v7_grpo_${mode}.json"
done

# Step 4: Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/full_v7_grpo_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']
summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'audio_contribution_positive': scores['real'] > scores['noise'] + 5,
    'training_phase': 'grpo_rl_4bit_fullscale',
    'sft_checkpoint': '${SFT_CKPT}',
    'grpo_data': '${GRPO_SUBSET}',
    'beta': 0.04,
    'lr': 5e-7,
    'num_generations': 4,
    'quantization': '4bit_nf4_double_quant',
}
with open('eval_results/full_v7_grpo_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
lines = ['# Full v7 GRPO RL Results (4-bit QLoRA)', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open('eval_results/full_v7_grpo_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/full_v7_grpo_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  Full v7 GRPO Complete! $(date)"
echo "============================================"
