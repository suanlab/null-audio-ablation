#!/usr/bin/env bash
# Wait for the AVUT delivery control, evaluate the registered gate, and launch the grid
# only if it passes.
#
# Per preregistration 3.3 an unvalidated model cannot be interpreted at all, so this
# decides whether 28 cells are worth running. A FAIL is a result, not an error: it would
# be the third instance of the delivery-versus-fusion distinction the paper is about, and
# it is reported rather than retried.
set -uo pipefail
ROOT=$REPO_ROOT
PY=$VENV/bin/python
R="$ROOT/eval_results"
LOG=/tmp/avut_gate.log
cd "$ROOT" || exit 1

echo "[$(date +%T)] waiting for AVUT delivery cells" | tee -a "$LOG"
while [ ! -s "$R/poscontrol_cmss_avut_qwenomni_real.json" ] || \
      [ ! -s "$R/poscontrol_cmss_avut_qwenomni_silent.json" ]; do
    sleep 60
done
echo "[$(date +%T)] both cells present; evaluating gate" | tee -a "$LOG"

VERDICT=$("$PY" - <<'PYEOF'
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from videollm.cmss import BOOTSTRAP_SEED, N_BOOT, cluster_bootstrap_mean
R = Path("eval_results")
load = lambda p: json.loads(p.read_text())["predictions"]
corr = lambda r: np.array([str(x["correct"]).lower() == "true" for x in r], float)
a, b = load(R / "poscontrol_cmss_avut_qwenomni_real.json"), load(R / "poscontrol_cmss_avut_qwenomni_silent.json")
assert [r["video"] for r in a] == [r["video"] for r in b], "cells are not item-paired"
cid = np.array([r["video"] for r in a])
pt, lo, hi = cluster_bootstrap_mean((corr(a) - corr(b)) * 100.0, cid, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
print(f"{'PASS' if lo > 0 else 'FAIL'} G_A={pt:+.2f} CI=[{lo:+.2f},{hi:+.2f}] "
      f"audio_only={100*corr(a).mean():.2f} floor={100*corr(b).mean():.2f} "
      f"unparsed={sum(1 for r in (*a, *b) if not str(r['pred']).strip())}")
PYEOF
)
echo "[$(date +%T)] gate: $VERDICT" | tee -a "$LOG"

case "$VERDICT" in
  PASS*)
    echo "[$(date +%T)] launching AVUT grid on GPU 1 and 2" | tee -a "$LOG"
    MODEL=qwenomni GPU=1 DO_CONTROLS=1 FAMILIES="motion_blur occlusion" \
      bash scripts/run_cmss_avut.sh >>"$LOG" 2>&1 &
    MODEL=qwenomni GPU=2 DO_CONTROLS=0 FAMILIES="frame_drop downscale" \
      bash scripts/run_cmss_avut.sh >>"$LOG" 2>&1 &
    wait
    echo "[$(date +%T)] AVUT grid finished" | tee -a "$LOG"
    ;;
  *)
    echo "[$(date +%T)] gate not passed -- grid NOT launched, by design" | tee -a "$LOG"
    ;;
esac
