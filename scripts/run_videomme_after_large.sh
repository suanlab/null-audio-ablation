#!/bin/bash
# Watcher: Run Video-MME eval after v6 large eval completes
set -euo pipefail

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Video-MME watcher started"
echo "  Waiting for large eval to finish..."

# Wait for large eval processes to finish
while pgrep -f "eval_avqa_large.sh" > /dev/null 2>&1; do
    sleep 60
done

# Also wait for the eval_avqa.py process (runs 5 modes sequentially)
while pgrep -f "eval_avqa.py.*eval_large" > /dev/null 2>&1; do
    sleep 60
done

# Wait for summary file
while [ ! -f "eval_results/full_v6_large_summary.json" ]; do
    echo "[$(date '+%H:%M:%S')] Waiting for large eval summary..."
    sleep 30
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Large eval done. Starting Video-MME eval..."

# Run Video-MME eval on v6
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/tmp/fake_cuda nohup ~/.venv/videollm/bin/python scripts/eval_videomme.py \
    --checkpoint checkpoints/full_v6/stage3_instruction \
    --parquet data/Video-MME/videomme/test-00000-of-00001.parquet \
    --video_dir data/Video-MME/videos \
    --output eval_results/videomme_full_v6.json \
    --num_frames 16 \
    --max_new_tokens 64 \
    > logs/eval_videomme_v6.log 2>&1 &

VID_PID=$!
echo "[$(date '+%H:%M:%S')] Video-MME eval started (PID: $VID_PID)"

# Wait for completion
wait $VID_PID

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Video-MME eval complete!"
