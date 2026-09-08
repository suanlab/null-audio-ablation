#!/bin/bash
# Run Video-MME evaluation for a given checkpoint.
# Usage: bash scripts/eval_videomme.sh <checkpoint_path> <output_name>
set -euo pipefail

CKPT="${1:?Usage: eval_videomme.sh <checkpoint> <output_name>}"
NAME="${2:?Usage: eval_videomme.sh <checkpoint> <output_name>}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Video-MME eval: $NAME"
echo "  Checkpoint: $CKPT"
echo "  Output: eval_results/videomme_${NAME}.json"

CUDA_HOME=/tmp/fake_cuda ~/.venv/videollm/bin/python scripts/eval_videomme.py \
    --checkpoint "$CKPT" \
    --parquet data/Video-MME/videomme/test-00000-of-00001.parquet \
    --video_dir data/Video-MME/videos \
    --output "eval_results/videomme_${NAME}.json" \
    --num_frames 16 \
    --max_new_tokens 64 \
    --use_audio --use_temporal_bridge --mm_projector_type stc

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Video-MME eval done: $NAME"
