#!/usr/bin/env bash
# CMSS grid on the real MUSIC-AVQA anchor (Li et al., CVPR 2022).
#
# Replaces the previous anchor, which was labelled MUSIC-AVQA but was a VGGSound-derived
# audio-event set. Two changes follow from that episode and are deliberate:
#
#   * Delivery controls run FIRST, not last. On AVUT they ran last, so 28 cells were spent
#     before learning the domain was unusable. Registered as an ordering change 2026-08-16.
#   * --answer_format vocab. MUSIC-AVQA answers are 41 words, not A-D letters; the reader
#     is shared between adapters so a parser difference cannot masquerade as a model one.
#
# Shardable by family so the slow model can occupy several GPUs at once:
#   MODEL=vsalm2 GPU=1 FAMILIES="motion_blur occlusion" bash scripts/run_cmss_music_avqa.sh
#
# Existing outputs are skipped, so a re-run resumes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:?set MODEL=vsalm2|vl2}"
GPU="${GPU:-3}"
FAMILIES="${FAMILIES:-motion_blur occlusion frame_drop downscale}"
SEVERITIES="${SEVERITIES:-0.33 0.66 1.0}"
AUDIO_MODES="${AUDIO_MODES:-real silent}"
DO_CONTROLS="${DO_CONTROLS:-1}"
EVAL_DATA="$ROOT/data/instruct/music_avqa_pilot_300.jsonl"
VIDEO_DIR="$ROOT/data/raw/music_avqa/videos/MUSIC-AVQA-videos-Real"
OUT_DIR="$ROOT/eval_results"
PREFIX="cmss_music_${MODEL}"
LOG="${LOG:-/tmp/cmss_music_${MODEL}_gpu${GPU}.log}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ] && { echo "ABORT: ${FREE_GB}GB free < ${MIN_FREE_GB}GB" | tee -a "$LOG"; exit 1; }
[ -s "$EVAL_DATA" ] || { echo "ABORT: no eval data at $EVAL_DATA" | tee -a "$LOG"; exit 1; }

LOCK="/tmp/cmss_music_${MODEL}_${GPU}.lock"
exec 9>"$LOCK"; flock -n 9 || { echo "ABORT: another run holds ${LOCK}" >&2; exit 1; }

run_cell() {  # run_cell <audio> <visual> <severity> <out> <blank>
    local audio="$1" visual="$2" sev="$3" out="$4" blank="${5:-0}"
    if [ -s "$out" ]; then echo "  [skip] $(basename "$out")" | tee -a "$LOG"; return 0; fi
    echo "[$(date +%T)] [run ] ${MODEL} gpu${GPU} audio=${audio} visual=${visual} sev=${sev} blank=${blank}" | tee -a "$LOG"
    if [ "$MODEL" = "vsalm2" ]; then
        CUDA_VISIBLE_DEVICES="$GPU" \
        LD_LIBRARY_PATH=$CONDA_ENVS/videosalmonn2/lib \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True POS_BLANK="$blank" \
        $CONDA_ENVS/videosalmonn2/bin/python "$ROOT/scripts/eval_avqa_videosalmonn2.py" \
            --model_path $MODELS_DIR/video-SALMONN2_plus_7B_full \
            --eval_data "$EVAL_DATA" --video_dir "$VIDEO_DIR" \
            --audio_mode "$audio" --visual_mode "$visual" --visual_severity "$sev" \
            --answer_format vocab --output "$out" >>"$LOG" 2>&1
    else
        ( cd $VIDEOLLAMA2_DIR || exit 1
          CUDA_VISIBLE_DEVICES="$GPU" POS_BLANK="$blank" \
          $CONDA_ENVS/videollama2/bin/python eval_avqa_5mode.py \
            --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
            --eval_data "$EVAL_DATA" --video_dir "$VIDEO_DIR" \
            --audio_mode "$audio" --visual_mode "$visual" --visual_severity "$sev" \
            --answer_format vocab --output "$out" >>"$LOG" 2>&1 )
    fi
    echo "[$(date +%T)] DONE exit=$? -> $(basename "$out")" | tee -a "$LOG"
}

if [ "$DO_CONTROLS" = "1" ]; then
    echo "--- delivery controls (run first) ---" | tee -a "$LOG"
    for audio in ${AUDIO_MODES}; do
        run_cell "$audio" clean 1.0 "${OUT_DIR}/poscontrol_${PREFIX}_${audio}.json" 1
    done
    echo "--- clean baseline ---" | tee -a "$LOG"
    for audio in ${AUDIO_MODES}; do
        run_cell "$audio" clean 1.0 "${OUT_DIR}/avqa_${PREFIX}_${audio}.json" 0
    done
fi

for family in ${FAMILIES}; do
    echo "--- family: ${family} ---" | tee -a "$LOG"
    for sev in ${SEVERITIES}; do
        tag=$(python -c "print('%g' % float('${sev}'))")
        for audio in ${AUDIO_MODES}; do
            run_cell "$audio" "$family" "$sev" \
                "${OUT_DIR}/avqa_${PREFIX}_${audio}_vis-${family}-s${tag}.json" 0
        done
    done
done
echo "[$(date +%T)] shard finished (${MODEL} gpu${GPU}: ${FAMILIES})" | tee -a "$LOG"
