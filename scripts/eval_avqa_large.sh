#!/bin/bash
################################################################################
# Evaluate on expanded AVQA eval set (548 samples)
#
# Runs 5-mode audio ablation on the large eval set for any checkpoint.
# Usage:
#   bash scripts/eval_avqa_large.sh <checkpoint_dir> <prefix> [--no-use_audio] [--no-use_temporal_bridge]
#
# Examples:
#   # Full AGTA model
#   bash scripts/eval_avqa_large.sh checkpoints/full_v6/stage3_instruction full_v6
#
#   # Vision-only baseline
#   bash scripts/eval_avqa_large.sh checkpoints/ablation/vision_only/stage3_instruction vision_only --no-use_audio --no-use_temporal_bridge
#
#   # No-bridge ablation
#   bash scripts/eval_avqa_large.sh checkpoints/ablation/no_bridge/stage3_instruction no_bridge --no-use_temporal_bridge
################################################################################
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <prefix> [extra_args...]"
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
CKPT="$1"
PREFIX="$2"
shift 2
EXTRA_ARGS="$*"

EVAL_DATA="data/instruct/avqa_eval_large.jsonl"

echo "============================================"
echo "  AVQA Large Eval: ${PREFIX}"
echo "  Checkpoint: ${CKPT}"
echo "  Eval data: ${EVAL_DATA} (548 samples)"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "  $(date)"
echo "============================================"

for mode in real shuffled shifted noise silent; do
    echo ""
    echo ">>> mode=${mode}..."
    $PYTHON scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data "${EVAL_DATA}" \
        --audio_mode "${mode}" \
        --output "eval_results/${PREFIX}_large_${mode}.json" \
        --sample_timeout 120 \
        ${EXTRA_ARGS}
done

# Generate summary
$PYTHON -c "
import json, sys
prefix = '${PREFIX}'
modes = ['real', 'shuffled', 'shifted', 'noise', 'silent']
scores = {}
for m in modes:
    path = f'eval_results/{prefix}_large_{m}.json'
    try:
        with open(path) as f:
            scores[m] = json.load(f)['accuracy']
    except Exception as e:
        print(f'Warning: Could not load {path}: {e}', file=sys.stderr)
        scores[m] = 0.0

summary = {
    'scores': scores,
    'delta_real_minus_noise': scores['real'] - scores['noise'],
    'delta_real_minus_silent': scores['real'] - scores['silent'],
    'delta_real_minus_shuffled': scores['real'] - scores['shuffled'],
    'eval_set': 'avqa_eval_large.jsonl (548 samples)',
    'checkpoint': '${CKPT}',
}
with open(f'eval_results/{prefix}_large_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

lines = [f'# {prefix} — Large Eval Results (548 samples)', '']
for m in modes:
    lines.append(f'- {m}: {scores[m]:.2f}')
lines.append('')
lines.append(f'- real-noise: {scores[\"real\"] - scores[\"noise\"]:.2f}')
lines.append(f'- real-silent: {scores[\"real\"] - scores[\"silent\"]:.2f}')
lines.append(f'- real-shuffled: {scores[\"real\"] - scores[\"shuffled\"]:.2f}')
with open(f'eval_results/{prefix}_large_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
"

echo ""
echo "============================================"
echo "  Large Eval Complete: ${PREFIX} $(date)"
echo "============================================"
