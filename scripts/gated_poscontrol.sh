#!/bin/bash
# Wait until some GPU has >=20GB free, then run the positive control on it.
set -uo pipefail
LOG=/tmp/poscontrol.log
: > "$LOG"
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '$2>=20000{print $1; exit}')
  if [ -n "${G:-}" ]; then
     echo "[$(date +%T)] GPU $G has >=20GB free; launching poscontrol" | tee -a "$LOG"
     GPU="$G" bash scripts/run_poscontrol.sh >>"$LOG" 2>&1
     echo "[$(date +%T)] poscontrol exit=$?" | tee -a "$LOG"
     break
  fi
  sleep 180
done
