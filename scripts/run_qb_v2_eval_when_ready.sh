#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
CHECKPOINT="checkpoints/audio_quick_bridge_v2/stage3_instruction"
EVAL_DATA="data/instruct/avqa_test_clean.jsonl"
LOG_PREFIX="[qb-v2-eval]"

echo "$LOG_PREFIX started at $(date)"

while true; do
    if [ -f "$CHECKPOINT/model.safetensors" ]; then
        echo "$LOG_PREFIX checkpoint ready: $CHECKPOINT"
        break
    fi

    if ! pgrep -f "scripts/train_quick_bridge_v2.sh|data/bridge/vggsound_train_quick5k.jsonl|data/instruct/avqa_train_quick5k.jsonl" >/dev/null; then
        echo "$LOG_PREFIX training process not found and checkpoint missing; exiting"
        exit 1
    fi

    sleep 300
done

mkdir -p eval_results

for mode in real shuffled shifted noise silent; do
    echo "$LOG_PREFIX running AVQA eval mode=$mode at $(date)"
    "$PYTHON" scripts/eval_avqa.py \
        --checkpoint "$CHECKPOINT" \
        --eval_data "$EVAL_DATA" \
        --audio_mode "$mode" \
        --output "eval_results/qb_v2_${mode}.json"
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path

base = Path("eval_results")
modes = ["real", "shuffled", "shifted", "noise", "silent"]
scores = {}
for mode in modes:
    p = base / f"qb_v2_{mode}.json"
    if not p.exists():
        raise SystemExit(f"missing result: {p}")
    data = json.loads(p.read_text())
    scores[mode] = float(data["accuracy"])

delta_vs_noise = scores["real"] - scores["noise"]
delta_vs_silent = scores["real"] - scores["silent"]
delta_vs_shuffled = scores["real"] - scores["shuffled"]

summary_path = base / "qb_v2_summary.json"
summary = {
    "scores": scores,
    "delta_real_minus_noise": delta_vs_noise,
    "delta_real_minus_silent": delta_vs_silent,
    "delta_real_minus_shuffled": delta_vs_shuffled,
    "audio_contribution_positive": bool(delta_vs_noise > 0 and delta_vs_silent > 0 and delta_vs_shuffled > 0),
}
summary_path.write_text(json.dumps(summary, indent=2))

report_path = base / "qb_v2_summary.md"
report_path.write_text(
    "# QB v2 Audio Ablation\n\n"
    f"- real: {scores['real']:.2f}\n"
    f"- shuffled: {scores['shuffled']:.2f}\n"
    f"- shifted: {scores['shifted']:.2f}\n"
    f"- noise: {scores['noise']:.2f}\n"
    f"- silent: {scores['silent']:.2f}\n\n"
    f"- real-noise: {delta_vs_noise:.2f}\n"
    f"- real-silent: {delta_vs_silent:.2f}\n"
    f"- real-shuffled: {delta_vs_shuffled:.2f}\n"
)

print(f"wrote {summary_path}")
print(f"wrote {report_path}")
PY

echo "$LOG_PREFIX completed at $(date)"
