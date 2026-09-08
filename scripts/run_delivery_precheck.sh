#!/usr/bin/env bash
# Pre-flight delivery check on 60 held-out MUSIC-AVQA items, before committing 56 cells.
#
# Delivery validation is what disqualified the AVUT domain for both models, and it was run
# last there, so the cost was only visible after the whole grid. Running it first here is
# the cheapest possible way to learn whether this anchor is usable at all.
set -uo pipefail
ROOT=$REPO_ROOT
DATA=$ROOT/data/instruct/music_avqa_format_check_60.jsonl
VIDS=$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real
LOG=/tmp/music_avqa_delivery_precheck.log
GPU="${GPU:-3}"

vs() {  # vs <audio_mode> <blank> <out>
    CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH=$CONDA_ENVS/videosalmonn2/lib \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True POS_BLANK="$2" \
    $CONDA_ENVS/videosalmonn2/bin/python "$ROOT/scripts/eval_avqa_videosalmonn2.py" \
        --model_path $MODELS_DIR/video-SALMONN2_plus_7B_full \
        --eval_data "$DATA" --video_dir "$VIDS" --audio_mode "$1" \
        --visual_mode clean --visual_severity 1.0 --answer_format vocab \
        --output "$3" >>"$LOG" 2>&1
    echo "[$(date +%T)] vsalm2 audio=$1 blank=$2 exit=$? -> $3" | tee -a "$LOG"
}
vl() {
    ( cd $VIDEOLLAMA2_DIR || exit 1
      CUDA_VISIBLE_DEVICES="$GPU" POS_BLANK="$2" \
      $CONDA_ENVS/videollama2/bin/python eval_avqa_5mode.py \
        --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
        --eval_data "$DATA" --video_dir "$VIDS" --audio_mode "$1" \
        --visual_mode clean --visual_severity 1.0 --answer_format vocab \
        --output "$3" >>"$LOG" 2>&1 )
    echo "[$(date +%T)] vl2 audio=$1 blank=$2 exit=$? -> $3" | tee -a "$LOG"
}
R=$ROOT/eval_results
vs silent 0 "$R/pre_music_vsalm2_silent.json"           # vision only  (real video, no audio)
vs real   1 "$R/pre_music_vsalm2_blank_real.json"       # audio only   (blank video)
vs silent 1 "$R/pre_music_vsalm2_blank_silent.json"     # no-information floor
vl silent 0 "$R/pre_music_vl2_silent.json"
vl real   1 "$R/pre_music_vl2_blank_real.json"
vl silent 1 "$R/pre_music_vl2_blank_silent.json"
echo "[$(date +%T)] delivery precheck done" | tee -a "$LOG"
