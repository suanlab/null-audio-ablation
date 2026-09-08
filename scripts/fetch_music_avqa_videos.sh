#!/usr/bin/env bash
# Fetch the official MUSIC-AVQA real-video archive.
#
# The dataset is distributed only as two bulk zips on Google Drive (no selective download,
# no video_id -> YouTube mapping is published), so the whole 36.67 GB real archive has to
# come down even though the pilot needs a few hundred clips. The synthetic archive
# (11.59 GB) is deliberately skipped: restricting the pilot to real videos is a cleaner
# evaluation claim and is recorded as such in the split manifest.
#
# Resumable: gdown continues a partial file, and an already-extracted tree is left alone.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/music_avqa"
PY=python
LOG=/tmp/fetch_music_avqa.log
REAL_ID=1Ovj5Ay8rDXaPy57CNCHes0A99S43lPBy
MIN_FREE_GB=90   # 37 GB zip + ~37 GB extracted + headroom

FREE_GB=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
    echo "ABORT: only ${FREE_GB}GB free, need >= ${MIN_FREE_GB}GB." | tee -a "$LOG"; exit 1
fi
echo "[$(date +%T)] free=${FREE_GB}GB; fetching real-video archive" | tee -a "$LOG"

cd "$DEST" || exit 1
if [ ! -s MUSIC-AVQA-videos-Real.zip ]; then
    "$PY" -c "
import gdown
gdown.download(id='${REAL_ID}', output='MUSIC-AVQA-videos-Real.zip', quiet=False, resume=True)
" >>"$LOG" 2>&1
fi
echo "[$(date +%T)] download exit=$? size=$(du -h MUSIC-AVQA-videos-Real.zip 2>/dev/null | cut -f1)" | tee -a "$LOG"

if [ -s MUSIC-AVQA-videos-Real.zip ]; then
    if unzip -tq MUSIC-AVQA-videos-Real.zip >>"$LOG" 2>&1; then
        echo "[$(date +%T)] archive integrity OK; extracting" | tee -a "$LOG"
        unzip -n -q MUSIC-AVQA-videos-Real.zip -d videos >>"$LOG" 2>&1
        echo "[$(date +%T)] extracted: $(find videos -name '*.mp4' | wc -l) mp4 files" | tee -a "$LOG"
    else
        echo "[$(date +%T)] ARCHIVE CORRUPT OR INCOMPLETE - not extracting" | tee -a "$LOG"; exit 1
    fi
fi
