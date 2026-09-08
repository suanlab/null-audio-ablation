#!/usr/bin/env bash
# Download datasets for VideoLLM training (AudioCaps, VGGSound, AVQA).
#
# Usage:
#   bash scripts/download_data.sh [--dataset DATASET] [--data-root DIR]
#
# Options:
#   --dataset   One of: audiocaps, vggsound, avqa, all (default: all)
#   --data-root Base directory for downloaded data (default: data/)
#
# Prerequisites:
#   pip install yt-dlp aac-datasets
#   sudo apt install ffmpeg
#
# Directory layout after download:
#   data/
#   ├── raw/
#   │   ├── audiocaps/       # CSV annotations + WAV audio
#   │   ├── vggsound/        # CSV annotations + MP4 videos
#   │   └── avqa/            # JSON annotations (videos from vggsound)
#   └── videos/              # Unified video directory (symlinked)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATASET="all"
DATA_ROOT="data"
MAX_WORKERS=4
YT_DLP_OPTS="--quiet --no-warnings --extract-audio --audio-format wav --audio-quality 0"
YT_DLP_VIDEO_OPTS="--quiet --no-warnings -f bestvideo[height<=480]+bestaudio/best[height<=480] --merge-output-format mp4"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --workers)   MAX_WORKERS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RAW_DIR="${DATA_ROOT}/raw"
VIDEO_DIR="${DATA_ROOT}/videos"

echo "=== VideoLLM Dataset Downloader ==="
echo "Dataset:   ${DATASET}"
echo "Data root: ${DATA_ROOT}"
echo "Workers:   ${MAX_WORKERS}"
echo ""

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

check_deps() {
    local missing=()
    for cmd in yt-dlp ffmpeg python3; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Missing required tools: ${missing[*]}"
        echo "Install with:"
        echo "  pip install yt-dlp"
        echo "  sudo apt install ffmpeg"
        exit 1
    fi
}

check_deps

# ---------------------------------------------------------------------------
# AudioCaps
# ---------------------------------------------------------------------------

download_audiocaps() {
    echo ">>> [1/3] Downloading AudioCaps..."
    local ac_dir="${RAW_DIR}/audiocaps"
    mkdir -p "${ac_dir}"

    # Download CSV annotations from official GitHub repo
    local base_url="https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
    for split in train val test; do
        local csv_file="${ac_dir}/${split}.csv"
        if [[ -f "${csv_file}" ]]; then
            echo "  [skip] ${split}.csv already exists"
        else
            echo "  Downloading ${split}.csv..."
            curl -sL "${base_url}/${split}.csv" -o "${csv_file}"
            echo "  [done] ${split}.csv ($(wc -l < "${csv_file}") lines)"
        fi
    done

    # Download audio/video clips using yt-dlp
    echo "  Downloading audio clips (this may take hours)..."
    python3 -c "
import csv
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ac_dir = Path('${ac_dir}')
video_dir = Path('${VIDEO_DIR}')
video_dir.mkdir(parents=True, exist_ok=True)

def download_clip(row):
    \"\"\"Download a single AudioCaps clip as video+audio.\"\"\"
    yt_id = row['youtube_id']
    start = int(row['start_time'])
    out_file = video_dir / f'{yt_id}_{start}.mp4'
    if out_file.exists():
        return f'skip:{yt_id}'
    url = f'https://www.youtube.com/watch?v={yt_id}'
    cmd = [
        'yt-dlp', '-q', '--no-warnings',
        '-f', 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        '--merge-output-format', 'mp4',
        '--download-sections', f'*{start}-{start+10}',
        '-o', str(out_file),
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
        return f'ok:{yt_id}' if out_file.exists() else f'fail:{yt_id}'
    except Exception:
        return f'fail:{yt_id}'

total, ok, skip, fail = 0, 0, 0, 0
for split in ['train', 'val', 'test']:
    csv_path = ac_dir / f'{split}.csv'
    if not csv_path.exists():
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'  Processing {split}: {len(rows)} clips...')
    with ThreadPoolExecutor(max_workers=${MAX_WORKERS}) as pool:
        futures = {pool.submit(download_clip, r): r for r in rows}
        for fut in as_completed(futures):
            total += 1
            result = fut.result()
            if result.startswith('ok'):
                ok += 1
            elif result.startswith('skip'):
                skip += 1
            else:
                fail += 1
            if total % 500 == 0:
                print(f'    Progress: {total} processed (ok={ok}, skip={skip}, fail={fail})')

print(f'  AudioCaps done: {ok} downloaded, {skip} skipped, {fail} failed out of {total}')
"
    echo "  [done] AudioCaps"
}

# ---------------------------------------------------------------------------
# VGGSound
# ---------------------------------------------------------------------------

download_vggsound() {
    echo ">>> [2/3] Downloading VGGSound..."
    local vgg_dir="${RAW_DIR}/vggsound"
    mkdir -p "${vgg_dir}"

    # Download CSV annotation
    local csv_file="${vgg_dir}/vggsound.csv"
    if [[ -f "${csv_file}" ]]; then
        echo "  [skip] vggsound.csv already exists"
    else
        echo "  Downloading vggsound.csv..."
        curl -sL "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv" -o "${csv_file}"
        echo "  [done] vggsound.csv ($(wc -l < "${csv_file}") lines)"
    fi

    # Download video clips
    echo "  Downloading video clips (this may take many hours)..."
    python3 -c "
import csv
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

vgg_dir = Path('${vgg_dir}')
video_dir = Path('${VIDEO_DIR}')
video_dir.mkdir(parents=True, exist_ok=True)

def download_clip(row):
    \"\"\"Download a single VGGSound clip.\"\"\"
    yt_id = row[0].strip()
    start = int(row[1].strip())
    out_file = video_dir / f'{yt_id}_{start}.mp4'
    if out_file.exists():
        return 'skip'
    url = f'https://www.youtube.com/watch?v={yt_id}'
    cmd = [
        'yt-dlp', '-q', '--no-warnings',
        '-f', 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        '--merge-output-format', 'mp4',
        '--download-sections', f'*{start}-{start+10}',
        '-o', str(out_file),
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
        return 'ok' if out_file.exists() else 'fail'
    except Exception:
        return 'fail'

csv_path = vgg_dir / 'vggsound.csv'
with open(csv_path) as f:
    reader = csv.reader(f)
    rows = list(reader)

print(f'  Total VGGSound clips: {len(rows)}')
total, ok, skip, fail = 0, 0, 0, 0
with ThreadPoolExecutor(max_workers=${MAX_WORKERS}) as pool:
    futures = {pool.submit(download_clip, r): r for r in rows}
    for fut in as_completed(futures):
        total += 1
        result = fut.result()
        if result == 'ok': ok += 1
        elif result == 'skip': skip += 1
        else: fail += 1
        if total % 1000 == 0:
            print(f'    Progress: {total}/{len(rows)} (ok={ok}, skip={skip}, fail={fail})')

print(f'  VGGSound done: {ok} downloaded, {skip} skipped, {fail} failed out of {total}')
"
    echo "  [done] VGGSound"
}

# ---------------------------------------------------------------------------
# AVQA
# ---------------------------------------------------------------------------

download_avqa() {
    echo ">>> [3/3] Downloading AVQA annotations..."
    local avqa_dir="${RAW_DIR}/avqa"
    mkdir -p "${avqa_dir}"

    # AVQA annotations are hosted on OneDrive/Baidu — provide manual instructions
    # and attempt HuggingFace download as alternative
    echo "  Attempting HuggingFace download..."
    python3 -c "
from pathlib import Path
import json

avqa_dir = Path('${avqa_dir}')

try:
    from datasets import load_dataset

    print('  Loading AVQA from HuggingFace (Joysw909/AVQA)...')
    ds = load_dataset('Joysw909/AVQA')

    # Save train split
    train_path = avqa_dir / 'train_qa.json'
    if not train_path.exists():
        train_data = [dict(row) for row in ds['train']]
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f'  [done] train_qa.json ({len(train_data)} samples)')
    else:
        print(f'  [skip] train_qa.json already exists')

    # Save test/validation split
    for split_name in ['test', 'validation']:
        if split_name in ds:
            out_path = avqa_dir / 'val_qa.json'
            if not out_path.exists():
                val_data = [dict(row) for row in ds[split_name]]
                with open(out_path, 'w') as f:
                    json.dump(val_data, f, indent=2)
                print(f'  [done] val_qa.json ({len(val_data)} samples)')
            break

except ImportError:
    print('  WARNING: datasets library not installed. Install with: pip install datasets')
    print('  Alternative: Download AVQA manually from:')
    print('    OneDrive: https://tsinghuaeducn-my.sharepoint.com/:u:/g/personal/xin_wang_tsinghua_edu_cn/...')
    print('    Baidu: https://pan.baidu.com/s/1nftHgOAGYSCF6j8MxCjPqA?pwd=awd7')
    print('  Place train_qa.json and val_qa.json in: ${avqa_dir}/')

except Exception as e:
    print(f'  WARNING: HuggingFace download failed: {e}')
    print('  Download AVQA manually and place JSON files in: ${avqa_dir}/')
"

    echo "  NOTE: AVQA videos come from VGGSound. Run --dataset vggsound first."
    echo "  [done] AVQA"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

mkdir -p "${RAW_DIR}" "${VIDEO_DIR}"

case "${DATASET}" in
    audiocaps)  download_audiocaps ;;
    vggsound)   download_vggsound ;;
    avqa)       download_avqa ;;
    all)
        download_audiocaps
        download_vggsound
        download_avqa
        ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'. Choose: audiocaps, vggsound, avqa, all"
        exit 1
        ;;
esac

echo ""
echo "=== Download complete ==="
echo "Raw annotations: ${RAW_DIR}/"
echo "Videos:          ${VIDEO_DIR}/"
echo ""
echo "Next step: python scripts/prepare_data.py --data-root ${DATA_ROOT}"
