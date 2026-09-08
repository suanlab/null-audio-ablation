#!/bin/bash
# P1-A: five-mode protocol + blanked-video positive control for the 2nd
# external model, video-SALMONN 2+ 7B. GPU 2 ONLY (gate on nvidia-smi -i 2,
# pin CUDA_VISIBLE_DEVICES=2 — never the any-GPU awk). LD_LIBRARY_PATH must
# point at the env lib so torchcodec finds the conda FFmpeg (libavutil.so.59).
set -uo pipefail

ROOT=$REPO_ROOT
DATA=$ROOT/data/instruct/avqa_test_clean.jsonl     # matched 98-subset
VID=$ROOT/data/videos
IDX=$ROOT/eval_results/matched_98/audio_explicit_idx.json   # n=47 audio-explicit
MODEL=$MODELS_DIR/video-SALMONN2_plus_7B_full
ENVP=$CONDA_ENVS/videosalmonn2
PY=$ENVP/bin/python
LOG=/tmp/vsalm2.log
: > "$LOG"

export LD_LIBRARY_PATH="$ENVP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
cd "$ROOT"

gate_gpu2() {
  while true; do
    F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 2>/dev/null)
    if [ -n "${F:-}" ] && [ "$F" -ge 20000 ]; then
      echo "[$(date +%T)] GPU2 free=${F}MiB; proceeding" | tee -a "$LOG"; return 0
    fi
    echo "[$(date +%T)] GPU2 free=${F:-NA}MiB <20000; waiting" | tee -a "$LOG"; sleep 120
  done
}

run() {  # $1=output  $2=audio_mode  $3...=extra args
  local out="$1" mode="$2"; shift 2
  if [ -s "$out" ]; then echo "[$(date +%T)] skip $out (exists)" | tee -a "$LOG"; return 0; fi
  gate_gpu2
  echo "[$(date +%T)] START $mode -> $out $*" | tee -a "$LOG"
  "$PY" scripts/eval_avqa_videosalmonn2.py \
    --model_path "$MODEL" --eval_data "$DATA" --video_dir "$VID" \
    --audio_mode "$mode" --output "$out" "$@" >>"$LOG" 2>&1
  echo "[$(date +%T)] DONE $mode exit=$? -> $out" | tee -a "$LOG"
}

# --- five-mode protocol on the matched 98-subset ---
run "$ROOT/eval_results/vsalm2_real.json"     real
run "$ROOT/eval_results/vsalm2_silent.json"   silent
run "$ROOT/eval_results/vsalm2_noise.json"    noise
run "$ROOT/eval_results/vsalm2_shuffled.json" shuffled
run "$ROOT/eval_results/vsalm2_shifted.json"  shifted --shift_seconds 3.0

# --- blanked-video positive control on the n=47 audio-explicit subset ---
export POS_BLANK=1
export POS_IDX="$IDX"
run "$ROOT/eval_results/poscontrol_vsalm2_real.json"   real
run "$ROOT/eval_results/poscontrol_vsalm2_silent.json" silent
unset POS_BLANK POS_IDX

echo "[$(date +%T)] SALMONN ALL DONE" | tee -a "$LOG"
