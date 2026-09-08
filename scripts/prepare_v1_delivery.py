"""Build the media and manifests video-SALMONN v1 needs for the delivery control.

v1 takes video and audio as *separate* files (``image_name: [clip.mp4, clip.wav]``), which
makes the blank-video delivery control a matter of preparing media rather than patching the
model: point every item at one black video and vary only the waveform. That keeps this
comparison's intervention identical in spirit to the ones applied to the other two systems,
without a second implementation of it inside an unfamiliar codebase.

Why v1 at all: it is Whisper **+ BEATs** where video-SALMONN 2+ is Whisper-only (confirmed
at the weight level -- no BEATs tensors in the 2+ shard index). Same lab, same lineage, one
encoder added. With one model per front end elsewhere in this paper, every architectural
difference is equally consistent with the observed crossover; this pair narrows that.

Usage::

    python scripts/prepare_v1_delivery.py
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT = REPO_ROOT / "data" / "instruct" / "music_avqa_pilot_300.jsonl"
VIDEOS = REPO_ROOT / "data" / "raw" / "music_avqa" / "videos" / "MUSIC-AVQA-videos-Real"
OUT = Path("$VIDEOSALMONN_V1_DIR/delivery_data")
SR = 16000
DURATION = 60  # every MUSIC-AVQA clip is 60 s


def extract_wav(video: str) -> str | None:
    """Decode one clip's audio to a 16 kHz mono wav; return the path, or None on failure."""
    out = OUT / "wav" / f"{Path(video).stem}.wav"
    if out.exists():
        return str(out)
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(VIDEOS / video),
           "-ac", "1", "-ar", str(SR), "-vn", str(out)]
    done = subprocess.run(cmd, capture_output=True, timeout=120)
    return str(out) if done.returncode == 0 and out.exists() else None


def main() -> None:
    """Write the black video, the silent waveform, per-clip audio, and both manifests."""
    for sub in ("wav",):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    black = OUT / "black.mp4"
    if not black.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
             f"color=c=black:s=224x224:d={DURATION}:r=25", "-pix_fmt", "yuv420p", str(black)],
            capture_output=True, check=True, timeout=300,
        )
    silent = OUT / "silent.wav"
    if not silent.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
             f"anullsrc=r={SR}:cl=mono", "-t", str(DURATION), str(silent)],
            capture_output=True, check=True, timeout=300,
        )

    rows = [json.loads(line) for line in PILOT.read_text().splitlines()]
    videos = [str(r["video"]) for r in rows]
    with ThreadPoolExecutor(16) as ex:
        wavs = list(ex.map(extract_wav, videos))
    missing = sum(1 for w in wavs if w is None)

    for condition, audio_for in (("real", lambda i: wavs[i]), ("silent", lambda _i: str(silent))):
        items = []
        for i, row in enumerate(rows):
            if wavs[i] is None:
                continue  # a clip whose audio will not decode enters neither condition
            prompt = str(row["conversations"][0]["value"]).replace("<video>", "").replace("<audio>", "").strip()
            items.append({
                "image_name": [str(black), audio_for(i)],
                "conversation": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": "None"},
                ],
                "gt": str(row["conversations"][1]["value"]),
                "video": str(row["video"]),
            })
        path = OUT / f"delivery_{condition}.json"
        path.write_text(json.dumps(items, indent=1))
        print(f"wrote {path} ({len(items)} items)")

    print(f"black video : {black}")
    print(f"silent wav  : {silent}")
    print(f"clip wavs   : {sum(1 for w in wavs if w)} extracted, {missing} failed")


if __name__ == "__main__":
    main()
