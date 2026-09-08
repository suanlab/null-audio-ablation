#!/usr/bin/env bash
# Run the AVUT blank-video delivery positive controls out of band.
#
# Why this exists as a separate script: both sweep scripts place the positive
# controls LAST, so their evidence arrives only after the whole 28-cell grid is
# done. On AVUT that ordering is backwards -- VideoLLaMA2.1-AV measures
# Delta_A ~ 0 at every visual severity, and without the delivery control there is
# no way to tell "the model does not use audio here" from "audio never reached
# the model on these longer clips". Per preregistration 3.3 an unvalidated model
# cannot be interpreted at all, so this evidence gates the whole domain.
#
# Each control is written to exactly the filename its sweep script expects, so
# when the sweep reaches its own positive-control stage it sees a non-empty file
# and skips. The sweeps have hours of graded cells left, so these finish first.
#
# Runs each model on the GPU its own sweep is NOT using, to avoid contending for
# device memory with the run already in flight.
#
# Usage: bash scripts/run_avut_poscontrols.sh [vsalm2|vl2|both]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH="${1:-both}"
EVAL_DATA="$ROOT/data/instruct/avut_pilot_300.jsonl"
VIDEO_DIR="$ROOT/data/avut/videos"
OUT_DIR="$ROOT/eval_results"

run_vsalm2() {
    local envp=$CONDA_ENVS/videosalmonn2
    local log=/tmp/cmss_avut_poscontrol_vsalm2.log
    export LD_LIBRARY_PATH="$envp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES="${VSALM_GPU:-1}"  # its sweep holds GPU3
    cd "$ROOT" || return 1
    for audio in real silent; do
        local out="$OUT_DIR/poscontrol_cmss_avut_vsalm2_${audio}.json"
        if [ -s "$out" ]; then echo "[skip] $out" | tee -a "$log"; continue; fi
        echo "[$(date +%T)] [run ] vsalm2 poscontrol audio=${audio} blank-video" | tee -a "$log"
        POS_BLANK=1 "$envp/bin/python" scripts/eval_avqa_videosalmonn2.py \
            --model_path $MODELS_DIR/video-SALMONN2_plus_7B_full \
            --eval_data "$EVAL_DATA" --video_dir "$VIDEO_DIR" \
            --audio_mode "$audio" --visual_mode clean --visual_severity 1.0 \
            --output "$out" >>"$log" 2>&1
        echo "[$(date +%T)] DONE exit=$? -> $out" | tee -a "$log"
    done
}

run_vl2() {
    local envp=$CONDA_ENVS/videollama2
    local adapter=$VIDEOLLAMA2_DIR
    local log=/tmp/cmss_avut_poscontrol_vl2.log
    export CUDA_VISIBLE_DEVICES="${VL2_GPU:-3}"  # its sweep holds GPU1
    cd "$adapter" || return 1
    for audio in real silent; do
        local out="$OUT_DIR/poscontrol_cmss_avut_vl2_${audio}.json"
        if [ -s "$out" ]; then echo "[skip] $out" | tee -a "$log"; continue; fi
        echo "[$(date +%T)] [run ] vl2 poscontrol audio=${audio} blank-video" | tee -a "$log"
        POS_BLANK=1 "$envp/bin/python" eval_avqa_5mode.py \
            --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
            --eval_data "$EVAL_DATA" --video_dir "$VIDEO_DIR" \
            --audio_mode "$audio" --visual_mode clean --visual_severity 1.0 \
            --output "$out" >>"$log" 2>&1
        echo "[$(date +%T)] DONE exit=$? -> $out" | tee -a "$log"
    done
}

case "$WHICH" in
    vsalm2) run_vsalm2 ;;
    vl2)    run_vl2 ;;
    both)   run_vl2 & run_vsalm2 & wait ;;
    *)      echo "usage: $0 [vsalm2|vl2|both]" >&2; exit 1 ;;
esac
