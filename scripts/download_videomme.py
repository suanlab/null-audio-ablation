"""Download Video-MME videos from YouTube.

Reads the Video-MME parquet file to extract YouTube URLs and downloads
videos using yt-dlp. Handles rate limiting, retries, and skips already-
downloaded videos.

Usage::

    # Install yt-dlp first
    pip install yt-dlp

    # Download all videos (default: 720p max)
    python scripts/download_videomme.py

    # Download with custom resolution and workers
    python scripts/download_videomme.py --max_height 480 --workers 4

    # Resume from specific index
    python scripts/download_videomme.py --start_idx 200

    # Dry run — show what would be downloaded
    python scripts/download_videomme.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/download_videomme.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

PARQUET_PATH = "data/Video-MME/videomme/test-00000-of-00001.parquet"
OUTPUT_DIR = "data/Video-MME/videos"


def parse_args() -> argparse.Namespace:
    """Parse download arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Download Video-MME YouTube videos")
    parser.add_argument("--parquet", type=str, default=PARQUET_PATH, help="Path to Video-MME parquet")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory for videos")
    parser.add_argument("--max_height", type=int, default=720, help="Max video height (360/480/720)")
    parser.add_argument(
        "--workers", type=int, default=2, help="Parallel download workers (keep low to avoid rate limiting)"
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Start from this video index")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per video")
    parser.add_argument("--dry_run", action="store_true", help="Print URLs without downloading")
    parser.add_argument("--cookies", type=str, default=None, help="Path to cookies.txt for age-restricted videos")
    return parser.parse_args()


def download_video(
    video_id: str,
    url: str,
    output_dir: Path,
    max_height: int,
    max_retries: int,
    cookies_path: str | None = None,
) -> tuple[str, bool, str]:
    """Download a single YouTube video using yt-dlp.

    Args:
        video_id: YouTube video ID.
        url: Full YouTube URL.
        output_dir: Directory to save the video.
        max_height: Maximum video height.
        max_retries: Number of retries on failure.
        cookies_path: Optional path to cookies file.

    Returns:
        Tuple of (video_id, success, message).
    """
    output_path = output_dir / f"{video_id}.mp4"

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 10000:
        return video_id, True, "already exists"

    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--quiet",
        "-f",
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        "--socket-timeout",
        "30",
        "--retries",
        "3",
        "--no-overwrites",
    ]

    if cookies_path is not None:
        cmd.extend(["--cookies", cookies_path])

    cmd.append(url)

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0 and output_path.exists():
                return video_id, True, "downloaded"

            error_msg = result.stderr.strip()[:200] if result.stderr else "unknown error"

            # Check for known unrecoverable errors
            if any(msg in error_msg.lower() for msg in ["private video", "removed", "not available", "copyright"]):
                return video_id, False, f"unavailable: {error_msg}"

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Backoff: 10s, 20s, 30s
                logger.debug("Retry %d/%d for %s in %ds", attempt + 1, max_retries, video_id, wait_time)
                time.sleep(wait_time)

        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                continue
            return video_id, False, "timeout"

        except Exception as e:
            return video_id, False, f"exception: {e!s}"

    return video_id, False, f"failed after {max_retries} retries"


def main() -> None:
    """Run Video-MME download pipeline."""
    args = parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Check yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)

    # Load Video-MME metadata
    df = pd.read_parquet(args.parquet)
    videos = df[["videoID", "url"]].drop_duplicates(subset="videoID").reset_index(drop=True)
    logger.info("Found %d unique videos in Video-MME", len(videos))

    # Apply start index
    videos = videos.iloc[args.start_idx :]
    logger.info("Processing %d videos (starting from index %d)", len(videos), args.start_idx)

    # Check already downloaded
    existing = set(p.stem for p in output_dir.glob("*.mp4") if p.stat().st_size > 10000)
    remaining = videos[~videos["videoID"].isin(existing)]
    logger.info("Already downloaded: %d, remaining: %d", len(existing), len(remaining))

    if args.dry_run:
        for _, row in remaining.iterrows():
            print(f"{row['videoID']}: {row['url']}")
        logger.info("Dry run complete. %d videos would be downloaded.", len(remaining))
        return

    # Download with thread pool
    success_count = 0
    fail_count = 0
    failed_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_video,
                row["videoID"],
                row["url"],
                output_dir,
                args.max_height,
                args.max_retries,
                args.cookies,
            ): row["videoID"]
            for _, row in remaining.iterrows()
        }

        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            video_id, success, message = future.result()
            if success:
                success_count += 1
                if message != "already exists":
                    logger.info("[%d/%d] ✓ %s: %s", i, total, video_id, message)
            else:
                fail_count += 1
                failed_ids.append(video_id)
                logger.warning("[%d/%d] ✗ %s: %s", i, total, video_id, message)

            # Progress every 50 videos
            if i % 50 == 0:
                logger.info("Progress: %d/%d (success=%d, fail=%d)", i, total, success_count, fail_count)

            # Rate limiting: small delay between downloads
            time.sleep(1.0)

    # Final summary
    total_existing = len(list(output_dir.glob("*.mp4")))
    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("  New downloads: %d", success_count)
    logger.info("  Failed: %d", fail_count)
    logger.info("  Total videos on disk: %d / %d", total_existing, len(df["videoID"].unique()))

    if failed_ids:
        failed_path = output_dir / "failed_downloads.txt"
        with open(failed_path, "w") as f:
            for vid in failed_ids:
                f.write(f"{vid}\n")
        logger.info("  Failed IDs saved to: %s", failed_path)


if __name__ == "__main__":
    main()
