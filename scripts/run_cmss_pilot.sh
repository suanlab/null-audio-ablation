#!/bin/bash
################################################################################
# CMSS pilot — 2D audio x visual intervention grid (preregistration W3 gate)
#
# Runs the graded visual-degradation axis crossed with the audio axis on a pilot
# subset, plus the blank-video delivery positive control. Everything here is
# inference-only; no training.
#
# Grid per model:
#   visual: clean (severity 0) + {mild .33, medium .66, severe 1.0} x 4 families
#   audio:  real, silent            (the preregistered primary null is "silent")
#   control: blank video x {real, silent}   -- delivery validation, NOT a severity
#
#   cells = 2 (clean) + 2*3*4 (graded) + 2 (control) = 28 runs per model
#
# The pilot/threshold set MUST stay video-disjoint from the final test split
# (preregistration §6). Point EVAL_DATA at the pilot split, not the test split.
#
# Usage:
#   bash scripts/run_cmss_pilot.sh <checkpoint_dir> <prefix> [extra_args...]
#
# Examples:
#   bash scripts/run_cmss_pilot.sh checkpoints/full_v6/stage3_instruction full_v6
#   FAMILIES="motion_blur" SEVERITIES="1.0" bash scripts/run_cmss_pilot.sh <ckpt> quick
#
# Env overrides:
#   EVAL_DATA   pilot JSONL             (default data/instruct/avqa_pilot_300.jsonl)
#   VIDEO_DIR   video root              (default data/videos)
#   FAMILIES    corruption families     (default "motion_blur occlusion frame_drop downscale")
#   SEVERITIES  degraded severities     (default "0.33 0.66 1.0")
#   AUDIO_MODES audio conditions        (default "real silent")
#   OUT_DIR     eval-JSON destination   (default eval_results)
#   MIN_FREE_GB abort if disk below     (default 100) -- see the 2026-07-12 ENOSPC incident
#   SEED        base seed               (default 42)
#   DRY_RUN=1   print the plan, run nothing
#
# Existing outputs are skipped, so re-running resumes an interrupted sweep.
################################################################################
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <prefix> [extra_args...]"
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/tmp/fake_cuda}"
PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
CKPT="$1"
PREFIX="$2"
shift 2
EXTRA_ARGS="$*"

EVAL_DATA="${EVAL_DATA:-data/instruct/avqa_pilot_300.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-data/videos}"
FAMILIES="${FAMILIES:-motion_blur occlusion frame_drop downscale}"
SEVERITIES="${SEVERITIES:-0.33 0.66 1.0}"
AUDIO_MODES="${AUDIO_MODES:-real silent}"
OUT_DIR="${OUT_DIR:-eval_results}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

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
# A full volume silently kills a long sweep mid-run (2026-07-12 incident), so check first.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    echo "ABORT: only ${FREE_GB}GB free on this volume, need >= ${MIN_FREE_GB}GB." >&2
    echo "Free space or lower MIN_FREE_GB before starting a sweep." >&2
    exit 1
fi
if [ ! -f "${EVAL_DATA}" ]; then
    echo "ABORT: eval data not found: ${EVAL_DATA}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "============================================"
echo "  CMSS pilot: ${PREFIX}"
echo "  Checkpoint : ${CKPT}"
echo "  Eval data  : ${EVAL_DATA}"
echo "  Families   : ${FAMILIES}"
echo "  Severities : ${SEVERITIES}"
echo "  Audio      : ${AUDIO_MODES}"
echo "  Free disk  : ${FREE_GB}GB (min ${MIN_FREE_GB}GB)"
echo "  Dry run    : ${DRY_RUN}"
echo "  $(date)"
echo "============================================"

RUN_COUNT=0
SKIP_COUNT=0

# run_cell <audio_mode> <visual_mode> <severity> <output_json> [env_prefix]
run_cell() {
    local audio_mode="$1" visual_mode="$2" severity="$3" out="$4" blank="${5:-0}"

    if [ -f "${out}" ]; then
        echo "  [skip] ${out} (exists)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi
    echo "  [run ] audio=${audio_mode} visual=${visual_mode} sev=${severity} blank=${blank} -> ${out}"
    RUN_COUNT=$((RUN_COUNT + 1))
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi

    POS_BLANK="${blank}" ${PYTHON} scripts/eval_avqa.py \
        --checkpoint "${CKPT}" \
        --eval_data "${EVAL_DATA}" \
        --video_dir "${VIDEO_DIR}" \
        --audio_mode "${audio_mode}" \
        --visual_mode "${visual_mode}" \
        --visual_severity "${severity}" \
        --seed "${SEED}" \
        --output "${out}" \
        ${EXTRA_ARGS}
}

# --- 1. clean baseline (severity 0) ------------------------------------------
echo "--- clean (visual severity 0) ---"
for audio in ${AUDIO_MODES}; do
    run_cell "${audio}" "clean" "1.0" "${OUT_DIR}/avqa_${PREFIX}_${audio}.json" 0
done

# --- 2. graded visual degradation --------------------------------------------
for family in ${FAMILIES}; do
    echo "--- family: ${family} ---"
    for sev in ${SEVERITIES}; do
        sev_tag=$(${PYTHON} -c "print('%g' % float('${sev}'))")
        for audio in ${AUDIO_MODES}; do
            run_cell "${audio}" "${family}" "${sev}" \
                "${OUT_DIR}/avqa_${PREFIX}_${audio}_vis-${family}-s${sev_tag}.json" 0
        done
    done
done

# --- 3. blank-video delivery positive control --------------------------------
# Not a severity level: it validates that audio actually reaches the model, so a
# null audio effect can be distinguished from a harness that never delivered audio.
echo "--- positive control (blank video) ---"
for audio in ${AUDIO_MODES}; do
    run_cell "${audio}" "clean" "1.0" "${OUT_DIR}/poscontrol_${PREFIX}_${audio}.json" 1
done

echo "============================================"
echo "  Done: ${RUN_COUNT} run, ${SKIP_COUNT} skipped   $(date)"
echo ""
echo "  Aggregate a surface with:"
for family in ${FAMILIES}; do
    echo "    ${PYTHON} scripts/aggregate_cmss.py --model ${PREFIX} --family ${family} \\"
    echo "        --severities ${SEVERITIES} --eval_dir ${OUT_DIR}"
done
echo "============================================"
