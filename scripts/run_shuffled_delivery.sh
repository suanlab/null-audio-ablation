#!/usr/bin/env bash
# Blank-video delivery control with SHUFFLED audio (a donor clip's real waveform).
#
# Fusion loss is G_A - Delta_A(0), where G_A compares real audio against silence with the
# video blanked. A hostile reading of a large fusion loss is that silence degenerates the
# model, inflating G_A, so the "discarded information" is a silence artefact. Shuffled audio
# has the acoustic character of real audio and none of its answer content: if the gain
# measured against shuffled survives, the gain is content-driven and the fusion loss stands.
#
# Four cells: {BEATs, Whisper} x {MUSIC-AVQA, AVUT}, blank video throughout.
set -uo pipefail
ROOT=$REPO_ROOT
R="$ROOT/eval_results"
LOG=/tmp/shuffled_delivery.log

vl2() {  # vl2 <domain-tag> <eval_data> <video_dir> <answer_format> <gpu>
    local out="$R/shuf_poscontrol_cmss_$1_vl2.json"
    [ -s "$out" ] && { echo "[skip] $(basename $out)" | tee -a "$LOG"; return 0; }
    echo "[$(date +%T)] [run ] BEATs $1 shuffled" | tee -a "$LOG"
    ( cd $VIDEOLLAMA2_DIR || exit 1
      POS_BLANK=1 CUDA_VISIBLE_DEVICES="$5" $CONDA_ENVS/videollama2/bin/python eval_avqa_5mode.py \
        --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV --eval_data "$2" --video_dir "$3" \
        --audio_mode shuffled --visual_mode clean --visual_severity 1.0 \
        --answer_format "$4" --output "$out" >>"$LOG" 2>&1 )
    echo "[$(date +%T)] DONE exit=$? -> $(basename $out)" | tee -a "$LOG"
}
vsalm() {
    local out="$R/shuf_poscontrol_cmss_$1_vsalm2.json"
    [ -s "$out" ] && { echo "[skip] $(basename $out)" | tee -a "$LOG"; return 0; }
    echo "[$(date +%T)] [run ] Whisper $1 shuffled" | tee -a "$LOG"
    POS_BLANK=1 CUDA_VISIBLE_DEVICES="$5" \
    LD_LIBRARY_PATH=$CONDA_ENVS/videosalmonn2/lib \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $CONDA_ENVS/videosalmonn2/bin/python "$ROOT/scripts/eval_avqa_videosalmonn2.py" \
        --model_path $MODELS_DIR/video-SALMONN2_plus_7B_full \
        --eval_data "$2" --video_dir "$3" --audio_mode shuffled \
        --visual_mode clean --visual_severity 1.0 --answer_format "$4" \
        --output "$out" >>"$LOG" 2>&1
    echo "[$(date +%T)] DONE exit=$? -> $(basename $out)" | tee -a "$LOG"
}
MUSIC_D="$ROOT/data/instruct/music_avqa_pilot_300.jsonl"
MUSIC_V="$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real"
AVUT_D="$ROOT/data/instruct/avut_pilot_300.jsonl"
AVUT_V="$ROOT/data/avut/videos"

case "${ARM:?set ARM=beats|whisper}" in
  beats)   vl2   music "$MUSIC_D" "$MUSIC_V" vocab  "${GPU:-2}"
           vl2   avut  "$AVUT_D"  "$AVUT_V"  letter "${GPU:-2}" ;;
  whisper) vsalm music "$MUSIC_D" "$MUSIC_V" vocab  "${GPU:-3}"
           vsalm avut  "$AVUT_D"  "$AVUT_V"  letter "${GPU:-3}" ;;
esac
echo "[$(date +%T)] arm ${ARM} finished" | tee -a "$LOG"
