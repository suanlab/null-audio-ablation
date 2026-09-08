#!/bin/bash
################################################################################
# Waits for v6 training (train_full_v6.sh) to finish, then launches:
#   1. MVBench eval on v6 checkpoint
#   2. GRPO v7 training + eval
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v7_after_v6.sh > logs/run_v7_after_v6.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"

echo "Waiting for v6 training (train_full_v6/resume_v6) to finish..."
echo "$(date)"
while pgrep -f "(train_full_v6|resume_v6)" > /dev/null 2>&1; do
    sleep 300
done
V6_CKPT="checkpoints/full_v6/stage3_instruction"
while [ ! -f "${V6_CKPT}/model.safetensors" ]; do
    echo "v6 process exited but final checkpoint is not ready yet; waiting 60s..."
    sleep 60
done
echo "v6 training done at $(date)."

# Verify checkpoint exists
if [ ! -f "${V6_CKPT}/model.safetensors" ]; then
    echo "ERROR: v6 checkpoint not found at ${V6_CKPT}"
    exit 1
fi

# Step 1: MVBench eval on v6
echo ""
echo "============================================"
echo "  Step 1: MVBench eval on v6"
echo "  $(date)"
echo "============================================"
$PYTHON scripts/eval_mvbench.py \
    --checkpoint "${V6_CKPT}" \
    --mvbench_dir data/MVBench \
    --output eval_results/mvbench_full_v6.json

echo "  MVBench eval complete."
$PYTHON -c "
import json
with open('eval_results/mvbench_full_v6.json') as f:
    d = json.load(f)
print(f'  MVBench v6: {d[\"overall_accuracy\"]:.1f}% ({d[\"total_correct\"]}/{d[\"total_samples\"]})')
"

# Step 2: GRPO v7
echo ""
echo "============================================"
echo "  Step 2: GRPO v7 training"
echo "  $(date)"
echo "============================================"
bash scripts/train_full_v7_grpo.sh

echo ""
echo "============================================"
echo "  All v6→v7 pipeline complete! $(date)"
echo "============================================"
