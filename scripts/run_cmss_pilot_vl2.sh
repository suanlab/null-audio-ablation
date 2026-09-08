#!/bin/bash
################################################################################
# CMSS pilot — VideoLLaMA2.1-7B-AV (external, second delivery-validated model)
#
# Same 28-cell grid as scripts/run_cmss_pilot.sh, driven through the VideoLLaMA2
# adapter in its own conda env. This is the cheapest route to M1's ">= 2 models"
# requirement: the model already passed its blank-video positive control in the
# earlier five-mode work but was never run through CMSS.
#
# Grid (28 runs):
#   clean x {real,silent}                                                       =  2
#   {motion_blur,occlusion,frame_drop,downscale} x {.33,.66,1.0} x {real,silent} = 24
#   blank-video positive control x {real,silent}                                =  2
#
# Usage:
#   bash scripts/run_cmss_pilot_vl2.sh [prefix]
#
# Env overrides:
#   EVAL_DATA / VIDEO_DIR / FAMILIES / SEVERITIES / AUDIO_MODES / OUT_DIR
#   GPU         GPU index             (default 1)
#   MIN_FREE_MB gate before each cell (default 20000)
#   MIN_FREE_GB abort if disk below   (default 100)
#   DRY_RUN=1   print the plan only
#
# Existing outputs are skipped, so re-running resumes an interrupted sweep.
################################################################################
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:-cmss_vl2}"

ENVP=$CONDA_ENVS/videollama2
PY="$ENVP/bin/python"
ADAPTER_DIR="${ADAPTER_DIR:-$VIDEOLLAMA2_DIR}"
MODEL="${MODEL:-DAMO-NLP-SG/VideoLLaMA2.1-7B-AV}"

EVAL_DATA="${EVAL_DATA:-$ROOT/data/instruct/avqa_pilot_300.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/data/videos}"
FAMILIES="${FAMILIES:-motion_blur occlusion frame_drop downscale}"
SEVERITIES="${SEVERITIES:-0.33 0.66 1.0}"
AUDIO_MODES="${AUDIO_MODES:-real silent}"
OUT_DIR="${OUT_DIR:-$ROOT/eval_results}"
GPU="${GPU:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
DRY_RUN="${DRY_RUN:-0}"
LOG="${LOG:-/tmp/cmss_pilot_vl2.log}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"

# --- single-instance lock ----------------------------------------------------
# Two concurrent runs of one prefix race on the same JSONs and split the GPU,
# which looks like a hang rather than a collision. Fail fast instead.
LOCK="/tmp/cmss_pilot_${PREFIX}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "ABORT: another run for prefix '${PREFIX}' is already active (lock: ${LOCK})." >&2
    exit 1
fi

# --- guards ------------------------------------------------------------------
[ -x "$PY" ] || { echo "ABORT: env python not found: $PY" >&2; exit 1; }
[ -f "$EVAL_DATA" ] || { echo "ABORT: eval data not found: $EVAL_DATA" >&2; exit 1; }
[ -f "$ADAPTER_DIR/eval_avqa_5mode.py" ] || { echo "ABORT: adapter not found in $ADAPTER_DIR" >&2; exit 1; }
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    echo "ABORT: only ${FREE_GB}GB free, need >= ${MIN_FREE_GB}GB." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
cd "$ADAPTER_DIR"

echo "============================================" | tee -a "$LOG"
echo "  CMSS pilot: ${PREFIX} (VideoLLaMA2.1-AV)"    | tee -a "$LOG"
echo "  Adapter    : ${ADAPTER_DIR}"                 | tee -a "$LOG"
echo "  Eval data  : ${EVAL_DATA}"                   | tee -a "$LOG"
echo "  GPU        : ${GPU} (gate ${MIN_FREE_MB}MiB)" | tee -a "$LOG"
echo "  Families   : ${FAMILIES}"                    | tee -a "$LOG"
echo "  Free disk  : ${FREE_GB}GB"                   | tee -a "$LOG"
echo "  $(date)"                                     | tee -a "$LOG"
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
        echo "  [skip] $(basename "${out}")" | tee -a "$LOG"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi
    echo "  [run ] audio=${audio_mode} visual=${visual_mode} sev=${severity} blank=${blank}" | tee -a "$LOG"
    RUN_COUNT=$((RUN_COUNT + 1))
    [ "${DRY_RUN}" = "1" ] && return 0

    gate_gpu
    POS_BLANK="${blank}" "$PY" eval_avqa_5mode.py \
        --model_path "$MODEL" \
        --eval_data "$EVAL_DATA" \
        --video_dir "$VIDEO_DIR" \
        --audio_mode "$audio_mode" \
        --visual_mode "$visual_mode" \
        --visual_severity "$severity" \
        --output "$out" >>"$LOG" 2>&1
    echo "[$(date +%T)] DONE exit=$? -> $(basename "${out}")" | tee -a "$LOG"
}

echo "--- clean (visual severity 0) ---" | tee -a "$LOG"
for audio in ${AUDIO_MODES}; do
    run_cell "${audio}" "clean" "1.0" "${OUT_DIR}/avqa_${PREFIX}_${audio}.json" 0
done

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

# Delivery positive control: not a severity level. Validates that audio reaches the
# model, so a null audio effect can be told apart from a harness that never delivered it.
echo "--- positive control (blank video) ---" | tee -a "$LOG"
for audio in ${AUDIO_MODES}; do
    run_cell "${audio}" "clean" "1.0" "${OUT_DIR}/poscontrol_${PREFIX}_${audio}.json" 1
done

echo "============================================" | tee -a "$LOG"
echo "  Done: ${RUN_COUNT} run, ${SKIP_COUNT} skipped   $(date)" | tee -a "$LOG"
echo "  Aggregate: python scripts/aggregate_cmss.py --model ${PREFIX} --family <family>" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
