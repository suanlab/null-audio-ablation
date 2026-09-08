#!/bin/bash
################################################################################
# CMSS pilot — video-SALMONN 2+ 7B (external, positive-controlled)
#
# Same 28-cell grid as scripts/run_cmss_pilot.sh, but driven through the
# video-SALMONN adapter inside its own conda env. The env incantations below are
# carried over verbatim from scripts/run_videosalmonn2.sh: LD_LIBRARY_PATH must
# point at the env lib so torchcodec finds the conda FFmpeg (libavutil.so.59).
#
# Grid (28 runs):
#   clean x {real,silent}                                    =  2
#   {motion_blur,occlusion,frame_drop,downscale} x {.33,.66,1.0} x {real,silent} = 24
#   blank-video positive control x {real,silent}             =  2
#
# Usage:
#   bash scripts/run_cmss_pilot_vsalm2.sh [prefix]
#
# Env overrides:
#   EVAL_DATA / VIDEO_DIR / FAMILIES / SEVERITIES / AUDIO_MODES / OUT_DIR
#   GPU         GPU index                (default 2)
#   MIN_FREE_MB gate before each cell    (default 20000)
#   MIN_FREE_GB abort if disk below      (default 100)
#   DRY_RUN=1   print the plan only
#
# Existing outputs are skipped, so re-running resumes an interrupted sweep.
################################################################################
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:-cmss_vsalm2}"

ENVP=$CONDA_ENVS/videosalmonn2
PY="$ENVP/bin/python"
MODEL="${MODEL:-$MODELS_DIR/video-SALMONN2_plus_7B_full}"

EVAL_DATA="${EVAL_DATA:-$ROOT/data/instruct/avqa_pilot_300.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/data/videos}"
FAMILIES="${FAMILIES:-motion_blur occlusion frame_drop downscale}"
SEVERITIES="${SEVERITIES:-0.33 0.66 1.0}"
AUDIO_MODES="${AUDIO_MODES:-real silent}"
OUT_DIR="${OUT_DIR:-$ROOT/eval_results}"
GPU="${GPU:-2}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
DRY_RUN="${DRY_RUN:-0}"
LOG="${LOG:-/tmp/cmss_pilot_vsalm2.log}"

export LD_LIBRARY_PATH="$ENVP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

# --- single-instance lock ----------------------------------------------------
# Two concurrent runs of the same prefix race on the same output JSONs and split
# the GPU, which looks like a hang rather than a collision. Fail fast instead.
LOCK="/tmp/cmss_pilot_${PREFIX}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "ABORT: another run for prefix '${PREFIX}' is already active (lock: ${LOCK})." >&2
    exit 1
fi

# --- guards ------------------------------------------------------------------
[ -x "$PY" ] || { echo "ABORT: env python not found: $PY" >&2; exit 1; }
[ -f "$EVAL_DATA" ] || { echo "ABORT: eval data not found: $EVAL_DATA" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "ABORT: model dir not found: $MODEL" >&2; exit 1; }
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    echo "ABORT: only ${FREE_GB}GB free, need >= ${MIN_FREE_GB}GB." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "============================================" | tee -a "$LOG"
echo "  CMSS pilot: ${PREFIX} (video-SALMONN 2+)"   | tee -a "$LOG"
echo "  Model      : ${MODEL}"                      | tee -a "$LOG"
echo "  Eval data  : ${EVAL_DATA}"                  | tee -a "$LOG"
echo "  GPU        : ${GPU} (gate ${MIN_FREE_MB}MiB)" | tee -a "$LOG"
echo "  Families   : ${FAMILIES}"                   | tee -a "$LOG"
echo "  Severities : ${SEVERITIES}"                 | tee -a "$LOG"
echo "  Free disk  : ${FREE_GB}GB"                  | tee -a "$LOG"
echo "  $(date)"                                    | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

RUN_COUNT=0
SKIP_COUNT=0

gate_gpu() {
    while true; do
        F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
        if [ -n "${F:-}" ] && [ "$F" -ge "$MIN_FREE_MB" ]; then return 0; fi
        echo "[$(date +%T)] GPU${GPU} free=${F:-NA}MiB < ${MIN_FREE_MB}; waiting" | tee -a "$LOG"
        sleep 120
    done
}

# run_cell <audio_mode> <visual_mode> <severity> <output> <blank>
run_cell() {
    local audio_mode="$1" visual_mode="$2" severity="$3" out="$4" blank="${5:-0}"

    if [ -s "${out}" ]; then
        echo "  [skip] ${out}" | tee -a "$LOG"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi
    echo "  [run ] audio=${audio_mode} visual=${visual_mode} sev=${severity} blank=${blank} -> ${out}" | tee -a "$LOG"
    RUN_COUNT=$((RUN_COUNT + 1))
    [ "${DRY_RUN}" = "1" ] && return 0

    gate_gpu
    POS_BLANK="${blank}" "$PY" scripts/eval_avqa_videosalmonn2.py \
        --model_path "$MODEL" \
        --eval_data "$EVAL_DATA" \
        --video_dir "$VIDEO_DIR" \
        --audio_mode "$audio_mode" \
        --visual_mode "$visual_mode" \
        --visual_severity "$severity" \
        --output "$out" >>"$LOG" 2>&1
    echo "[$(date +%T)] DONE exit=$? -> ${out}" | tee -a "$LOG"
}

# --- 1. clean baseline --------------------------------------------------------
echo "--- clean (visual severity 0) ---" | tee -a "$LOG"
for audio in ${AUDIO_MODES}; do
    run_cell "${audio}" "clean" "1.0" "${OUT_DIR}/avqa_${PREFIX}_${audio}.json" 0
done

# --- 2. graded visual degradation --------------------------------------------
for family in ${FAMILIES}; do
    echo "--- family: ${family} ---" | tee -a "$LOG"
    for sev in ${SEVERITIES}; do
        sev_tag=$("$PY" -c "print('%g' % float('${sev}'))")
        for audio in ${AUDIO_MODES}; do
            run_cell "${audio}" "${family}" "${sev}" \
                "${OUT_DIR}/avqa_${PREFIX}_${audio}_vis-${family}-s${sev_tag}.json" 0
        done
    done
done

# --- 3. blank-video delivery positive control --------------------------------
echo "--- positive control (blank video) ---" | tee -a "$LOG"
for audio in ${AUDIO_MODES}; do
    run_cell "${audio}" "clean" "1.0" "${OUT_DIR}/poscontrol_${PREFIX}_${audio}.json" 1
done

echo "============================================" | tee -a "$LOG"
echo "  Done: ${RUN_COUNT} run, ${SKIP_COUNT} skipped   $(date)" | tee -a "$LOG"
echo "  Aggregate: python scripts/aggregate_cmss.py --model ${PREFIX} --family <family>" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
