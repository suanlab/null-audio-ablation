#!/usr/bin/env bash
# Qwen2.5-Omni with the whole audio track (added 2026-11-09).
#
# Every Qwen2.5-Omni cell cited before this date fed the model the first 30 s of each track
# (eval_avqa_qwenomni.py, CLIP_SECONDS); most clips run about 60 s. This reruns the four
# conditions the estimands need -- {blank, clean video} x {real, silent} -- on both pilot
# domains with QWEN_AUDIO_SECONDS=full. Silence there is as long as that item's real track, so
# real and null differ in content only.
#
#   ARM=music|avut GPU=0 bash scripts/run_qwen_fulltrack.sh
set -uo pipefail
ROOT=$REPO_ROOT
R="$ROOT/eval_results"
LOG=/tmp/qwen_fulltrack.log
MUSIC_D="$ROOT/data/instruct/music_avqa_pilot_300.jsonl"
MUSIC_V="$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real"
AVUT_D="$ROOT/data/instruct/avut_pilot_300.jsonl"
AVUT_V="$ROOT/data/avut/videos"

cell() {  # cell <domain> <eval_data> <video_dir> <answer_format> <blank:0|1> <audio_mode>
    local prefix; [ "$5" = 1 ] && prefix=poscontrol_cmss || prefix=avqa_cmss
    local out="$R/${prefix}_$1_qwenomni_fulltrack_$6.json"
    [ -s "$out" ] && { echo "[skip] $(basename "$out")" | tee -a "$LOG"; return 0; }
    echo "[$(date +%T)] [run ] $(basename "$out")" | tee -a "$LOG"
    local blank=""; [ "$5" = 1 ] && blank=1
    POS_BLANK=$blank QWEN_AUDIO_SECONDS=full CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $VENV/bin/python "$ROOT/scripts/eval_avqa_qwenomni.py" \
        --eval_data "$2" --video_dir "$3" --audio_mode "$6" \
        --visual_mode clean --visual_severity 1.0 --answer_format "$4" \
        --output "$out" >>"$LOG.$1" 2>&1
    local rc=$?  # read before any $(...) below resets it
    echo "[$(date +%T)] DONE exit=$rc -> $(basename "$out")" | tee -a "$LOG"
    [ "$rc" = 0 ] && [ -s "$out" ]
}
run_domain() {  # blank cells first: they carry the delivery gate and the chance analogue
    for blank in 1 0; do for mode in real silent; do cell "$@" "$blank" "$mode"; done; done
}
case "${ARM:?set ARM=music|avut}" in
  music) run_domain music "$MUSIC_D" "$MUSIC_V" vocab ;;
  avut)  run_domain avut  "$AVUT_D"  "$AVUT_V"  letter ;;
esac
