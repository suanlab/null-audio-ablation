#!/bin/bash
################################################################################
# Auto-launch QB v4 when audio utility scoring completes.
# Polls for the weights file, then starts QB v4 training.
#
# Usage:
#   CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=2 nohup bash scripts/auto_launch_v4.sh > logs/auto_launch_v4.log 2>&1 &
################################################################################
set -euo pipefail

WEIGHTS_FILE="data/instruct/avqa_train_quick5k_weights.json"
POLL_INTERVAL=120  # Check every 2 minutes
MAX_WAIT=43200     # 12 hours max

echo "============================================"
echo "  Waiting for audio utility scoring to complete..."
echo "  Polling for: ${WEIGHTS_FILE}"
echo "  $(date)"
echo "============================================"

elapsed=0
while [ ! -f "${WEIGHTS_FILE}" ]; do
    sleep ${POLL_INTERVAL}
    elapsed=$((elapsed + POLL_INTERVAL))
    if [ ${elapsed} -ge ${MAX_WAIT} ]; then
        echo "ERROR: Timed out waiting for ${WEIGHTS_FILE} after ${MAX_WAIT}s"
        exit 1
    fi
    echo "  Still waiting... (${elapsed}s elapsed)"
done

echo ""
echo ">>> Weights file found! $(date)"
echo ">>> Launching QB v4 training..."
echo ""

# Launch QB v4
exec bash scripts/train_quick_bridge_v4_weighted.sh
