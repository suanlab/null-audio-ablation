#!/bin/bash
# Blanked-video positive control on the audio-explicit subset (n=47).
# For AGTA / VideoLLaMA2.1-AV / Qwen2.5-Omni, run modes {real, silent} with the
# video stream zeroed (POS_BLANK=1) restricted to POS_IDX. Outputs:
#   eval_results/poscontrol_{model}_{mode}.json
# The "video present" column is computed offline from existing per-mode JSONs.
set -euo pipefail

ROOT=$REPO_ROOT
IDX=$ROOT/eval_results/matched_98/audio_explicit_idx.json
DATA=$ROOT/data/instruct/avqa_test_clean.jsonl
VID=$ROOT/data/videos
GPU="${GPU:?set GPU index}"
export POS_IDX="$IDX"
export POS_BLANK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"

PY_IH=python
PY_VL=$CONDA_ENVS/videollama2/bin/python

cd "$ROOT"
for M in real silent; do
  echo "=== [$(date +%T)] AGTA blank-video $M ==="
  $PY_IH scripts/eval_avqa.py \
    --checkpoint checkpoints/full_v6/stage3_instruction \
    --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode "$M" --output "eval_results/poscontrol_agta_${M}.json"

  echo "=== [$(date +%T)] Qwen2.5-Omni blank-video $M ==="
  $PY_IH scripts/eval_avqa_qwenomni.py \
    --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode "$M" --output "eval_results/poscontrol_qwenomni_${M}.json"
done

cd $VIDEOLLAMA2_DIR
for M in real silent; do
  echo "=== [$(date +%T)] VideoLLaMA2.1-AV blank-video $M ==="
  $PY_VL eval_avqa_5mode.py \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode "$M" --output "$ROOT/eval_results/poscontrol_videollama2_${M}.json"
done
echo "=== [$(date +%T)] POSCONTROL ALL DONE ==="
