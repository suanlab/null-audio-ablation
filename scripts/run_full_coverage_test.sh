#!/usr/bin/env bash
# Match the BEATs-fronted model's audio coverage to the Whisper-fronted model's, then
# re-measure delivery on both domains.
#
# Why this is mandatory rather than nice-to-have: the manuscript states that audio access
# is balanced across the comparison. It is balanced across *domains* but not across
# *models*. video-SALMONN 2+ chunks the whole track into consecutive 30 s windows (100%
# coverage); VideoLLaMA2.1-AV receives eight spliced 2 s windows (~27%). The asymmetry runs
# in the same direction as the crossover it is used to support, so it has to be removed
# rather than argued around. BEATs has no length cap, so this is measurable.
#
# If BEATs/AVUT stays ~0 at full coverage, the front-end reading is as strong as this
# design allows. If it rises, the paper has found that a widely reported null is partly a
# harness artifact -- a better result, but a different one.
set -uo pipefail
ROOT=$REPO_ROOT
ADAPTER=$VIDEOLLAMA2_DIR
PY=$CONDA_ENVS/videollama2/bin/python
LOG=/tmp/full_coverage_test.log
GPU="${GPU:-2}"
cd "$ADAPTER" || exit 1

cell() {  # cell <domain> <eval_data> <video_dir> <audio_mode> <out>
    local out="$5"
    if [ -s "$out" ]; then echo "  [skip] $(basename "$out")" | tee -a "$LOG"; return 0; fi
    echo "[$(date +%T)] [run ] $1 audio=$4 FULL coverage, blank video" | tee -a "$LOG"
    POS_BLANK=1 CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval_avqa_5mode.py \
        --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
        --eval_data "$2" --video_dir "$3" \
        --audio_mode "$4" --visual_mode clean --visual_severity 1.0 \
        --audio_access full --answer_format "$6" \
        --output "$out" >>"$LOG" 2>&1
    echo "[$(date +%T)] DONE exit=$? -> $(basename "$out")" | tee -a "$LOG"
}

R="$ROOT/eval_results"
# The silent condition is unaffected by the access flag (it is a zero waveform), but it is
# re-run so the pair is produced by one identical code path.
cell AVUT  "$ROOT/data/instruct/avut_pilot_300.jsonl" "$ROOT/data/avut/videos" \
     real   "$R/full_poscontrol_cmss_avut_vl2_real.json"   letter
cell AVUT  "$ROOT/data/instruct/avut_pilot_300.jsonl" "$ROOT/data/avut/videos" \
     silent "$R/full_poscontrol_cmss_avut_vl2_silent.json" letter
cell MUSIC "$ROOT/data/instruct/music_avqa_pilot_300.jsonl" \
     "$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real" \
     real   "$R/full_poscontrol_cmss_music_vl2_real.json"   vocab
cell MUSIC "$ROOT/data/instruct/music_avqa_pilot_300.jsonl" \
     "$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real" \
     silent "$R/full_poscontrol_cmss_music_vl2_silent.json" vocab
echo "[$(date +%T)] full-coverage delivery controls finished" | tee -a "$LOG"
