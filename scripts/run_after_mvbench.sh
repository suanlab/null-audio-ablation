#!/bin/bash
################################################################################
# Waits for MVBench eval to finish, then runs GRPO v5b on the same GPU.
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_after_mvbench.sh > logs/run_after_mvbench.log 2>&1 &
################################################################################
set -euo pipefail

echo "Waiting for MVBench eval (eval_mvbench) to finish..."
while pgrep -f "eval_mvbench" > /dev/null 2>&1; do
    sleep 60
done
echo "MVBench eval done at $(date). Starting GRPO v5b..."

# Run GRPO v5b
bash scripts/train_qb_v5b_grpo_fixed.sh

echo "All done at $(date)"
