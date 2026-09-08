"""Download video clips from YouTube for AudioCaps and VGGSound datasets.

Runs yt-dlp in parallel threads with progress tracking.
Designed to be run in background via tmux/nohup.

Usage::

    python scripts/download_videos.py --dataset audiocaps --max-clips 1000 --workers 8
    python scripts/download_videos.py --dataset vggsound --max-clips 2000 --workers 8
    python scripts/download_videos.py --dataset all --max-clips 1000 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data/download.log")],
)
logger = logging.getLogger(__name__)

FFMPEG_PATH = None


def find_ffmpeg() -> str:
    """Locate ffmpeg binary."""
    import shutil

    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def download_clip(yt_id: str, start: int, duration: int, out_path: Path) -> str:
    """Download a YouTube video, then extract a clip with ffmpeg.

    Strategy: download full video to temp file, extract [start, start+duration]
    segment with ffmpeg -ss/-t, then delete the full file. This avoids
    --download-sections which requires specific ffmpeg builds.

    Returns 'ok', 'skip', or 'fail'.
    """
    if out_path.exists() and out_path.stat().st_size > 1000:
        return "skip"

    url = f"https://www.youtube.com/watch?v={yt_id}"
    tmp_full = out_path.with_suffix(".full.mp4")

    import shutil

    yt_dlp_bin = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")
    dl_cmd = [
        yt_dlp_bin,
        "-q",
        "--no-warnings",
        "-f",
        "best[height<=480]",
        "-o",
        str(tmp_full),
        url,
    ]

    try:
        subprocess.run(dl_cmd, capture_output=True, timeout=120)
        if not tmp_full.exists() or tmp_full.stat().st_size < 1000:
            tmp_full.unlink(missing_ok=True)
            return "fail"

        ffmpeg_bin = FFMPEG_PATH or "ffmpeg"
        cut_cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(tmp_full),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(out_path),
        ]
        subprocess.run(cut_cmd, capture_output=True, timeout=30)

        tmp_full.unlink(missing_ok=True)

        if out_path.exists() and out_path.stat().st_size > 1000:
            return "ok"
        return "fail"
    except subprocess.TimeoutExpired:
        tmp_full.unlink(missing_ok=True)
        return "fail"
    except Exception:
        tmp_full.unlink(missing_ok=True)
        return "fail"


def download_audiocaps(data_root: Path, max_clips: int, workers: int) -> dict[str, int]:
    """Download AudioCaps video clips."""
    csv_path = data_root / "raw" / "audiocaps" / "train.csv"
    video_dir = data_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error("AudioCaps train.csv not found at %s", csv_path)
        return {"ok": 0, "skip": 0, "fail": 0}

    clips: list[tuple[str, int, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yt_id = row["youtube_id"]
            start = int(row["start_time"])
            video_file = f"{yt_id}_{start}.mp4"
            clips.append((yt_id, start, video_file))

    if max_clips > 0:
        clips = clips[:max_clips]

    logger.info("AudioCaps: downloading %d clips with %d workers", len(clips), workers)
    return _parallel_download(clips, video_dir, workers, duration=10)


def download_vggsound(data_root: Path, max_clips: int, workers: int) -> dict[str, int]:
    """Download VGGSound video clips."""
    csv_path = data_root / "raw" / "vggsound" / "vggsound.csv"
    video_dir = data_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error("VGGSound CSV not found at %s", csv_path)
        return {"ok": 0, "skip": 0, "fail": 0}

    clips: list[tuple[str, int, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            yt_id = row[0].strip()
            start = int(row[1].strip())
            video_file = f"{yt_id}_{start}.mp4"
            clips.append((yt_id, start, video_file))

    if max_clips > 0:
        clips = clips[:max_clips]

    logger.info("VGGSound: downloading %d clips with %d workers", len(clips), workers)
    return _parallel_download(clips, video_dir, workers, duration=10)


def download_avqa(data_root: Path, max_clips: int, workers: int) -> dict[str, int]:
    """Download AVQA video clips from YouTube."""
    json_path = data_root / "raw" / "avqa" / "train_qa.json"
    video_dir = data_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        logger.error("AVQA train_qa.json not found at %s", json_path)
        return {"ok": 0, "skip": 0, "fail": 0}

    with open(json_path) as f:
        raw_data = json.load(f)

    existing = {p.stem for p in video_dir.glob("*.mp4")}
    seen: set[str] = set()
    clips: list[tuple[str, int, str]] = []

    for item in raw_data:
        vname = item.get("video_name", "")
        parts = vname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        yt_id, start_padded = parts
        try:
            start = int(start_padded)
        except ValueError:
            continue
        video_file = f"{yt_id}_{start}.mp4"
        if video_file[:-4] in existing or video_file in seen:
            continue
        seen.add(video_file)
        clips.append((yt_id, start, video_file))
        if max_clips > 0 and len(clips) >= max_clips:
            break

    logger.info("AVQA: downloading %d clips with %d workers", len(clips), workers)
    return _parallel_download(clips, video_dir, workers, duration=10)


def _parallel_download(
    clips: list[tuple[str, int, str]],
    video_dir: Path,
    workers: int,
    duration: int = 10,
) -> dict[str, int]:
    """Run parallel downloads with progress tracking."""
    stats = {"ok": 0, "skip": 0, "fail": 0}
    total = len(clips)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for yt_id, start, video_file in clips:
            out_path = video_dir / video_file
            fut = pool.submit(download_clip, yt_id, start, duration, out_path)
            futures[fut] = video_file

        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            stats[result] += 1

            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] ok=%d skip=%d fail=%d | %.1f clips/s | ETA %.0fs",
                    i,
                    total,
                    stats["ok"],
                    stats["skip"],
                    stats["fail"],
                    rate,
                    eta,
                )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d ok, %d skip, %d fail out of %d (%.1fs)",
        stats["ok"],
        stats["skip"],
        stats["fail"],
        total,
        elapsed,
    )
    return stats


def main() -> None:
    """Run video downloads."""
    global FFMPEG_PATH
    FFMPEG_PATH = find_ffmpeg()
    logger.info("ffmpeg: %s", FFMPEG_PATH)
    logger.info("yt-dlp: %s", subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True).stdout.strip())

    parser = argparse.ArgumentParser(description="Download video clips for VideoLLM")
    parser.add_argument("--dataset", type=str, default="all", choices=["audiocaps", "vggsound", "avqa", "all"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--max-clips", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.dataset in ("audiocaps", "all"):
        stats = download_audiocaps(data_root, args.max_clips, args.workers)
        logger.info("AudioCaps results: %s", stats)

    if args.dataset in ("vggsound", "all"):
        stats = download_vggsound(data_root, args.max_clips, args.workers)
        logger.info("VGGSound results: %s", stats)

    if args.dataset in ("avqa", "all"):
        stats = download_avqa(data_root, args.max_clips, args.workers)
        logger.info("AVQA results: %s", stats)

    # Count total videos
    video_dir = data_root / "videos"
    n_videos = len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0
    logger.info("Total videos in %s: %d", video_dir, n_videos)


if __name__ == "__main__":
    main()
