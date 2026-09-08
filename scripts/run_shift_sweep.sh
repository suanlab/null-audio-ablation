#!/bin/bash
# P1-C: finer shift-magnitude sweep on the matched 98-sample subset.
# HARD CONSTRAINT: GPU 2 ONLY. We gate specifically on GPU index 2's free
# memory (nvidia-smi -i 2) and pin CUDA_VISIBLE_DEVICES=2. We deliberately do
# NOT reuse gated_poscontrol.sh's "any GPU >=20GB" awk, which would select
# GPU 0 (off-limits on this shared machine).
#
# Reuses existing audited canonical JSONs for the points that already exist
# (AGTA real/1s/3s, VideoLLaMA2.1-AV real/3s); only the NEW magnitudes are run
# here, into eval_results/shift_sweep/.
set -uo pipefail

ROOT=$REPO_ROOT
DATA=$ROOT/data/instruct/avqa_test_clean.jsonl   # the 98-sample matched subset
VID=$ROOT/data/videos
OUT=$ROOT/eval_results/shift_sweep
LOG=/tmp/shift_sweep.log
PY_IH=python
PY_VL=$CONDA_ENVS/videollama2/bin/python
AGTA_CKPT=$ROOT/checkpoints/full_v6/stage3_instruction
mkdir -p "$OUT"
: > "$LOG"

gate_gpu2() {
  # Block until physical GPU 2 has >=20GB free. Never touches any other GPU.
  while true; do
    F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 2>/dev/null)
    if [ -n "${F:-}" ] && [ "$F" -ge 20000 ]; then
      echo "[$(date +%T)] GPU2 free=${F}MiB >=20000; proceeding" | tee -a "$LOG"
      return 0
    fi
    echo "[$(date +%T)] GPU2 free=${F:-NA}MiB <20000; waiting" | tee -a "$LOG"
    sleep 120
  done
}

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2

# AGTA: new magnitudes only (real/1.0/3.0 reused from canonical JSONs).
for S in 0.10 0.25 0.50 2.0 5.0; do
  TAG=$(echo "$S" | sed 's/\./p/')
  O="$OUT/full_v6_s${TAG}.json"
  if [ -s "$O" ]; then echo "[$(date +%T)] skip AGTA s=$S (exists)" | tee -a "$LOG"; continue; fi
  gate_gpu2
  echo "[$(date +%T)] START AGTA shift=$S -> $O" | tee -a "$LOG"
  $PY_IH scripts/eval_avqa.py \
    --checkpoint "$AGTA_CKPT" --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode shifted --shift_seconds "$S" --output "$O" >>"$LOG" 2>&1
  echo "[$(date +%T)] DONE AGTA shift=$S exit=$? -> $O" | tee -a "$LOG"
done

# VideoLLaMA2.1-AV: new magnitudes only (real/3.0 reused from canonical JSONs).
cd "$ROOT/../VideoLLaMA2"
for S in 0.10 0.25 0.50 1.0 2.0 5.0; do
  TAG=$(echo "$S" | sed 's/\./p/')
  O="$OUT/videollama2_s${TAG}.json"
  if [ -s "$O" ]; then echo "[$(date +%T)] skip VL2 s=$S (exists)" | tee -a "$LOG"; continue; fi
  gate_gpu2
  echo "[$(date +%T)] START VL2 shift=$S -> $O" | tee -a "$LOG"
  $PY_VL eval_avqa_5mode.py \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode shifted --shift_seconds "$S" --output "$O" >>"$LOG" 2>&1
  echo "[$(date +%T)] DONE VL2 shift=$S exit=$? -> $O" | tee -a "$LOG"
done

echo "[$(date +%T)] SWEEP ALL DONE" | tee -a "$LOG"
