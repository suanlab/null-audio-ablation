#!/bin/bash
################################################################################
# Vision-Only Baseline: SigLIP + STC + Qwen2-7B (NO audio, NO AGTA bridge)
#
# Critical ablation for isolating the contribution of audio/AGTA bridge.
# Uses the SAME STC projector init (from QB v2 Stage 2), SAME data (25K),
# SAME hyperparameters as Full v6 — only difference is --no-use_audio
# and --no-use_temporal_bridge.
#
# Base: QB v2 Stage 2 STC projector weights (strict=False ignores audio/bridge)
# Data: avqa_train_v3.jsonl (25,371 samples)
# Expected time: ~14-18 hours on 1x A100 80GB (3 epochs, no audio overhead)
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_vision_only_baseline.sh > logs/train_vision_only.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
BASE_DIR="checkpoints/ablation/vision_only"
QB_V2_S2="checkpoints/audio_quick_bridge_v2/stage2_bridge"

echo "============================================"
echo "  Vision-Only Baseline (25K, no audio)"
echo "  $(date)"
echo "============================================"

echo ""
echo ">>> Stage 3: Instruction (25K samples, 3 epochs, vision-only)"
echo "    Loading STC projector from ${QB_V2_S2}"
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
    --no-use_audio \
    --no-use_temporal_bridge \
    --mm_projector_type stc \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 50 \
    --save_total_limit 2 \
    --dataloader_num_workers 0 \
    --report_to none

echo "    Stage 3 complete: $(date)"

# Auto-run 5-mode eval (vision-only: real/silent should be identical)
echo ""
echo ">>> Running 5-mode AVQA evaluation..."
CKPT="${BASE_DIR}/stage3_instruction"
for mode in real shuffled shifted noise silent; do
    echo "  Evaluating mode=${mode}..."
    $PYTHON scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --audio_mode "${mode}" \
        --no-use_audio \
        --no-use_temporal_bridge \
        --output "eval_results/vision_only_${mode}.json"
done

# Run MVBench evaluation
echo ""
echo ">>> Running MVBench evaluation..."
$PYTHON scripts/eval_mvbench.py \
    --checkpoint "${CKPT}" \
    --no-use_audio \
    --no-use_temporal_bridge \
    --output eval_results/mvbench_vision_only.json

# Generate summary
$PYTHON -c "
import json
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    with open(f'eval_results/vision_only_{m}.json') as f:
        scores[m] = json.load(f)['accuracy']

# MVBench
mvbench_acc = None
try:
    with open('eval_results/mvbench_vision_only.json') as f:
        mvbench_acc = json.load(f).get('overall_accuracy', None)
except Exception:
    pass

summary = {
    'scores': scores,
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'all_modes_equal_expected': abs(scores['real'] - scores['silent']) < 3.0,
    'mvbench_accuracy': mvbench_acc,
    'training_phase': 'vision_only_baseline',
    'dataset': 'avqa_train_v3.jsonl (25K)',
    'audio_enabled': False,
    'bridge_enabled': False,
    'base_checkpoint': 'qb_v2_stage2',
}
with open('eval_results/vision_only_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

lines = ['# Vision-Only Baseline Results (25K dataset)', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-silent gap: {scores[\"real\"] - scores[\"silent\"]:.2f} (expect ~0)')
if mvbench_acc is not None:
    lines.append(f'- MVBench: {mvbench_acc:.1f}%')
with open('eval_results/vision_only_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Summary saved to eval_results/vision_only_summary.json')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  Vision-Only Baseline Complete! $(date)"
echo "============================================"
