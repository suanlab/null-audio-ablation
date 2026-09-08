#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-$HOME/.venv/videollm/bin/python}"
CHECKPOINT="checkpoints/audio_v3/stage3_instruction"
EVAL_DATA="data/instruct/avqa_test_clean.jsonl"
LOG_PREFIX="[v3-eval]"

echo "$LOG_PREFIX started at $(date)"

while true; do
    if [ -f "$CHECKPOINT/model.safetensors" ]; then
        echo "$LOG_PREFIX checkpoint ready: $CHECKPOINT"
        break
    fi

    if ! pgrep -f "scripts/train_audio_v3_scaled.sh|data/bridge/vggsound_train_v3.jsonl|data/instruct/avqa_train_v3.jsonl" >/dev/null; then
        echo "$LOG_PREFIX training process not found and checkpoint missing; exiting"
        exit 1
    fi

    sleep 600
done

mkdir -p eval_results

for mode in real shuffled shifted noise silent; do
    echo "$LOG_PREFIX running AVQA eval mode=$mode at $(date)"
    "$PYTHON" scripts/eval_avqa.py \
        --checkpoint "$CHECKPOINT" \
        --eval_data "$EVAL_DATA" \
        --audio_mode "$mode" \
        --output "eval_results/v3_${mode}.json"
done

echo "$LOG_PREFIX running MVBench eval at $(date)"
"$PYTHON" scripts/eval_mvbench.py \
    --checkpoint "$CHECKPOINT" \
    --output "eval_results/v3_mvbench.json"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

base = Path("eval_results")
modes = ["real", "shuffled", "shifted", "noise", "silent"]
scores = {}
for mode in modes:
    p = base / f"v3_{mode}.json"
    if not p.exists():
        raise SystemExit(f"missing result: {p}")
    data = json.loads(p.read_text())
    scores[mode] = float(data["accuracy"])

mvbench_path = base / "v3_mvbench.json"
if not mvbench_path.exists():
    raise SystemExit(f"missing result: {mvbench_path}")
mvbench = json.loads(mvbench_path.read_text())
mvbench_acc = float(mvbench.get("overall_accuracy", 0.0))

delta_vs_noise = scores["real"] - scores["noise"]
delta_vs_silent = scores["real"] - scores["silent"]
delta_vs_shuffled = scores["real"] - scores["shuffled"]

summary_path = base / "v3_summary.json"
summary = {
    "scores": scores,
    "mvbench_overall_accuracy": mvbench_acc,
    "delta_real_minus_noise": delta_vs_noise,
    "delta_real_minus_silent": delta_vs_silent,
    "delta_real_minus_shuffled": delta_vs_shuffled,
    "audio_contribution_positive": bool(delta_vs_noise > 0 and delta_vs_silent > 0 and delta_vs_shuffled > 0),
    "mvbench_above_random_baseline_30_9": bool(mvbench_acc > 30.9),
}
summary_path.write_text(json.dumps(summary, indent=2))

report_path = base / "v3_summary.md"
report_path.write_text(
    "# V3 Evaluation Summary\n\n"
    f"- real: {scores['real']:.2f}\n"
    f"- shuffled: {scores['shuffled']:.2f}\n"
    f"- shifted: {scores['shifted']:.2f}\n"
    f"- noise: {scores['noise']:.2f}\n"
    f"- silent: {scores['silent']:.2f}\n"
    f"- mvbench_overall_accuracy: {mvbench_acc:.2f}\n\n"
    f"- real-noise: {delta_vs_noise:.2f}\n"
    f"- real-silent: {delta_vs_silent:.2f}\n"
    f"- real-shuffled: {delta_vs_shuffled:.2f}\n"
)

print(f"wrote {summary_path}")
print(f"wrote {report_path}")
PY

echo "$LOG_PREFIX completed at $(date)"
