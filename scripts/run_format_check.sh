#!/usr/bin/env bash
# Format validation before committing to the 56-cell MUSIC-AVQA grid.
#
# The previous anchor was scaled to 112 cells without anyone checking what the data was.
# This runs 60 held-out items (never the pilot) through both models on the real audio
# condition and reports how often the reply can be read as one of the 41 vocabulary words.
# A poor parse rate here means the prompt format is wrong, and it is far cheaper to learn
# that now than after 56 cells.
set -uo pipefail
ROOT=$REPO_ROOT
DATA=$ROOT/data/instruct/music_avqa_format_check_60.jsonl
VIDS=$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real
LOG=/tmp/music_avqa_format_check.log

echo "[$(date +%T)] video-SALMONN 2+" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${GPU_A:-3}" \
LD_LIBRARY_PATH=$CONDA_ENVS/videosalmonn2/lib \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$CONDA_ENVS/videosalmonn2/bin/python "$ROOT/scripts/eval_avqa_videosalmonn2.py" \
    --model_path $MODELS_DIR/video-SALMONN2_plus_7B_full \
    --eval_data "$DATA" --video_dir "$VIDS" \
    --audio_mode real --visual_mode clean --visual_severity 1.0 \
    --answer_format vocab \
    --output "$ROOT/eval_results/fmtcheck_music_avqa_vsalm2.json" >>"$LOG" 2>&1
echo "[$(date +%T)] vsalm2 exit=$?" | tee -a "$LOG"

echo "[$(date +%T)] VideoLLaMA2.1-AV" | tee -a "$LOG"
cd $VIDEOLLAMA2_DIR || exit 1
CUDA_VISIBLE_DEVICES="${GPU_B:-3}" \
$CONDA_ENVS/videollama2/bin/python eval_avqa_5mode.py \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --eval_data "$DATA" --video_dir "$VIDS" \
    --audio_mode real --visual_mode clean --visual_severity 1.0 \
    --answer_format vocab \
    --output "$ROOT/eval_results/fmtcheck_music_avqa_vl2.json" >>"$LOG" 2>&1
echo "[$(date +%T)] vl2 exit=$?" | tee -a "$LOG"
