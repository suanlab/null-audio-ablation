#!/bin/bash
# P1-B: larger-n (n=548) robustness of the EXTERNAL positive-controlled
# dissociation. Pool = avqa_eval_large.jsonl (548). NOTE: this pool overlaps
# our in-house AGTA training (avqa_train_v3), so it is used ONLY for the
# OFF-THE-SHELF external models (VideoLLaMA2.1-AV, video-SALMONN 2+), for
# which our training is irrelevant; AGTA's clean comparison stays on the
# matched-98. Modes: real, silent (the R-S dissociation) + shifted (large-n
# temporal-invariance). GPU 2 ONLY (gate on nvidia-smi -i 2).
set -uo pipefail
ROOT=$REPO_ROOT
DATA=$ROOT/data/instruct/avqa_eval_large.jsonl
VID=$ROOT/data/videos
LOG=/tmp/p1b_large.log
VS_ENV=$CONDA_ENVS/videosalmonn2
PY_VL=$CONDA_ENVS/videollama2/bin/python
PY_VS=$VS_ENV/bin/python
VS_MODEL=$MODELS_DIR/video-SALMONN2_plus_7B_full
: > "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
cd "$ROOT"

gate_gpu2() {
  while true; do
    F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 2>/dev/null)
    [ -n "${F:-}" ] && [ "$F" -ge 20000 ] && { echo "[$(date +%T)] GPU2 free=${F}MiB; go" | tee -a "$LOG"; return 0; }
    echo "[$(date +%T)] GPU2 free=${F:-NA}MiB <20000; waiting" | tee -a "$LOG"; sleep 120
  done
}

vl_run() {  # $1=mode $2=out  $3...=extra
  local mode="$1" out="$2"; shift 2
  [ -s "$out" ] && { echo "[$(date +%T)] skip $out" | tee -a "$LOG"; return 0; }
  gate_gpu2; echo "[$(date +%T)] START VL2 $mode -> $out" | tee -a "$LOG"
  $PY_VL "$ROOT/../VideoLLaMA2/eval_avqa_5mode.py" \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --eval_data "$DATA" --video_dir "$VID" --audio_mode "$mode" --output "$out" "$@" >>"$LOG" 2>&1
  echo "[$(date +%T)] DONE VL2 $mode exit=$? -> $out" | tee -a "$LOG"
}
vs_run() {  # $1=mode $2=out  $3...=extra
  local mode="$1" out="$2"; shift 2
  [ -s "$out" ] && { echo "[$(date +%T)] skip $out" | tee -a "$LOG"; return 0; }
  gate_gpu2; echo "[$(date +%T)] START VS2 $mode -> $out" | tee -a "$LOG"
  LD_LIBRARY_PATH="$VS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  $PY_VS scripts/eval_avqa_videosalmonn2.py \
    --model_path "$VS_MODEL" --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode "$mode" --output "$out" "$@" >>"$LOG" 2>&1
  echo "[$(date +%T)] DONE VS2 $mode exit=$? -> $out" | tee -a "$LOG"
}

for M in real silent; do
  vl_run "$M" "$ROOT/eval_results/large548_videollama2_${M}.json"
  vs_run "$M" "$ROOT/eval_results/large548_vsalm2_${M}.json"
done
vl_run shifted "$ROOT/eval_results/large548_videollama2_shifted.json" --shift_seconds 3.0
vs_run shifted "$ROOT/eval_results/large548_vsalm2_shifted.json"  --shift_seconds 3.0

echo "[$(date +%T)] P1B LARGE ALL DONE" | tee -a "$LOG"
