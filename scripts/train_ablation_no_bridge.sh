#!/bin/bash
################################################################################
# Ablation: Audio WITHOUT AGTA Bridge (concat baseline)
#
# Uses audio (CLAP encoder + audio projector) but NO temporal bridge.
# Audio and visual tokens are simply concatenated before the LLM.
# This isolates the contribution of the AGTA cross-attention bridge vs
# naive audio-visual concatenation.
#
# Config: use_audio=True, use_temporal_bridge=False
# Base: QB v2 Stage 2 (loads STC + audio projector, no bridge weights)
# Data: avqa_train_v3.jsonl (25,371 samples)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_ablation_no_bridge.sh > logs/train_no_bridge.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/ablation/no_bridge"
QB_V2_S2="checkpoints/audio_quick_bridge_v2/stage2_bridge"

echo "============================================"
echo "  Ablation: No Bridge (audio concat only)"
echo "  $(date)"
echo "============================================"

echo ""
echo ">>> Stage 3: Instruction (25K samples, 3 epochs, audio=True, bridge=False)"
echo "    Loading STC + audio projector from ${QB_V2_S2}"
echo "    Start: $(date)"

$PYTHON src/videollm/train.py \
    --data_path data/instruct/avqa_train_v3.jsonl \
    --output_dir "${BASE_DIR}/stage3_instruction" \
    --load_from "${QB_V2_S2}" \
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
    --no-use_temporal_bridge \
    --mm_projector_type stc \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_total_limit 2 \
    --dataloader_num_workers 0 \
    --report_to none \
    --audio_dropout_prob 0.3

echo "    Stage 3 complete: $(date)"

# Auto-run 5-mode eval
echo ""
echo ">>> Running 5-mode AVQA evaluation..."
CKPT="${BASE_DIR}/stage3_instruction"
for mode in real shuffled shifted noise silent; do
    echo "  Evaluating mode=${mode}..."
    $PYTHON scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --audio_mode "${mode}" \
        --use_audio \
        --no-use_temporal_bridge \
        --output "eval_results/no_bridge_${mode}.json"
done

# Run MVBench evaluation
echo ""
echo ">>> Running MVBench evaluation..."
$PYTHON scripts/eval_mvbench.py \
    --checkpoint "${CKPT}" \
    --use_audio \
    --no-use_temporal_bridge \
    --output eval_results/mvbench_no_bridge.json

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/no_bridge_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']

mvbench_acc = None
try:
    with open('eval_results/mvbench_no_bridge.json') as f:
        mvbench_acc = json.load(f).get('overall_accuracy', None)
except Exception:
    pass

summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'mvbench_accuracy': mvbench_acc,
    'training_phase': 'ablation_no_bridge',
    'dataset': 'avqa_train_v3.jsonl (25K)',
    'audio_enabled': True,
    'bridge_enabled': False,
    'audio_dropout_prob': 0.3,
    'base_checkpoint': 'qb_v2_stage2',
}
with open('eval_results/no_bridge_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

lines = ['# No-Bridge Ablation Results (audio concat, 25K dataset)', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
if mvbench_acc is not None:
    lines.append(f'- MVBench: {mvbench_acc:.1f}%')
with open('eval_results/no_bridge_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/no_bridge_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  No-Bridge Ablation Complete! $(date)"
echo "============================================"
