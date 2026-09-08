#!/usr/bin/env bash
# Controlled test: does VideoLLaMA2.1-AV's null audio delivery on AVUT come from the
# harness's audio ACCESS, or from its BEATs front end?
#
# Upstream feeds the real condition eight 2-second windows spliced end to end and padded
# to 30 s -- 16 s of audio with the time axis broken at every splice, whatever the clip
# length. On the anchor domain (median 10 s) the windows overlap and cover everything; on
# AVUT (median 56 s) they cover 28.5% median / 12.2% worst. So audio access is fully
# confounded with domain, and the AVUT delivery number cannot be read as a model property.
#
# This runs the same blank-video delivery control with contiguous audio instead:
#   cell A (AVUT)   -- the test. Gain rises => access was the binding constraint.
#                      Gain flat  => the BEATs front end is, and the front-end claim is clean.
#   cell B (anchor) -- sanity. Both paths are near-equivalent here (all clips <= 18 s), so a
#                      collapse would mean the patch broke audio rather than improved it.
#
# The silent condition is deliberately NOT re-run: it is waveform_to_fbank(zeros) on both
# paths, so it is byte-identical and the existing cell is reused as the baseline.
set -uo pipefail
ROOT=$REPO_ROOT
ADAPTER=$VIDEOLLAMA2_DIR
PY=$CONDA_ENVS/videollama2/bin/python
LOG=/tmp/contig_audio_test.log
export CUDA_VISIBLE_DEVICES="${GPU:-3}"
cd "$ADAPTER" || exit 1

run() {  # run <tag> <eval_data> <video_dir> <out>
    local tag="$1" data="$2" vdir="$3" out="$4"
    if [ -s "$out" ]; then echo "[skip] $out" | tee -a "$LOG"; return 0; fi
    echo "[$(date +%T)] [run ] ${tag}: blank video + real audio, CONTIGUOUS" | tee -a "$LOG"
    POS_BLANK=1 "$PY" eval_avqa_5mode.py \
        --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
        --eval_data "$data" --video_dir "$vdir" \
        --audio_mode real --visual_mode clean --visual_severity 1.0 \
        --audio_access contiguous \
        --output "$out" >>"$LOG" 2>&1
    echo "[$(date +%T)] DONE exit=$? -> $out" | tee -a "$LOG"
}

run "AVUT"   "$ROOT/data/instruct/avut_pilot_300.jsonl" "$ROOT/data/avut/videos" \
             "$ROOT/eval_results/contig_poscontrol_cmss_avut_vl2_real.json"
run "anchor" "$ROOT/data/instruct/avqa_pilot_300.jsonl" "$ROOT/data/videos" \
             "$ROOT/eval_results/contig_poscontrol_cmss_vl2_real.json"
echo "[$(date +%T)] all cells finished" | tee -a "$LOG"
