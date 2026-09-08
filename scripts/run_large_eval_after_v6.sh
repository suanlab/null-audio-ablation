#!/bin/bash
################################################################################
# Waits for v6 training to finish, then runs expanded AVQA eval (548 samples)
# on the v6 checkpoint. This complements the v7 auto-chain (which runs
# MVBench + GRPO) with a larger audio grounding evaluation.
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_large_eval_after_v6.sh > logs/run_large_eval_v6.log 2>&1 &
################################################################################
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"

echo "Waiting for v6 training to finish..."
while pgrep -f "(train_full_v6|resume_v6)" > /dev/null 2>&1; do
    sleep 300
done

V6_CKPT="checkpoints/full_v6/stage3_instruction"
while [ ! -f "${V6_CKPT}/model.safetensors" ]; do
    echo "v6 process exited but final checkpoint is not ready yet; waiting 60s..."
    sleep 60
done
echo "v6 training done at $(date). Running large eval..."

if [ ! -f "${V6_CKPT}/model.safetensors" ]; then
    echo "ERROR: v6 checkpoint not found at ${V6_CKPT}"
    exit 1
fi

bash scripts/eval_avqa_large.sh "${V6_CKPT}" full_v6

echo "Large eval complete at $(date)."
