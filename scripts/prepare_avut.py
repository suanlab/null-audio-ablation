"""Convert AVUT into the eval JSONL format and pick a video-disjoint pilot split.

AVUT (Audio-centric Video Understanding Benchmark, EMNLP 2025) is the second CMSS
domain. It is the right choice because M1 cannot be tested on MUSIC-AVQA at all: there,
either channel alone nearly solves the task, so "both modalities" has no headroom over
the better single channel. AVUT is audio-centric by construction and text-shortcut
filtered, so a utilisation gap can exist.

Two properties matter downstream:

* ~2.5 questions per video, so the preregistered **video-cluster bootstrap does real
  work** here — on the MUSIC-AVQA pilot each video contributed one item, which made
  clustering equivalent to an item bootstrap.
* Same 4-way multiple-choice shape as MUSIC-AVQA, so the existing answer parser and the
  25% chance floor carry over unchanged.

The split is held out **by whole video**, matching `preregistration.md` §6: no video may
appear in both the pilot and the final test set.

Usage::

    python scripts/prepare_avut.py --n_pilot 300
    python scripts/prepare_avut.py --n_pilot 300 --list_videos   # download manifest only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AVUT_DIR = REPO_ROOT / "data" / "avut"
SOURCE = AVUT_DIR / "AV_Human_filtered_data.json"
PILOT_SEED = 2027  # same seed as the preregistered bootstrap; fixed, not tuned


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare an AVUT pilot split")
    parser.add_argument("--n_pilot", type=int, default=300, help="Target pilot item count")
    parser.add_argument("--source", type=str, default=str(SOURCE), help="AVUT annotation JSON")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "data" / "instruct"))
    parser.add_argument(
        "--list_videos",
        action="store_true",
        help="Only print the videos the pilot needs (a download manifest), write nothing",
    )
    return parser.parse_args()


def to_conversation(record: dict[str, object]) -> dict[str, object]:
    """Render one AVUT record in the repo's eval JSONL shape.

    The prompt mirrors the MUSIC-AVQA wording exactly so that answer parsing, the
    ``<video>``/``<audio>`` placeholders, and the letter-extraction regex behave
    identically across domains — a prompt difference would confound any cross-domain
    comparison of `Delta_A`.
    """
    options = "\n".join(f"{letter}. {record[f'option_{letter}']}" for letter in "ABCD")
    answer = str(record["answer"]).strip().upper()
    question = str(record["question"]).strip()
    return {
        "video": Path(str(record["video_path"])).name,
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n<audio>\n{question}\nChoices:\n{options}\nAnswer with the letter.",
            },
            {"from": "gpt", "value": f"The answer is {answer}. {record[f'option_{answer}']}"},
        ],
        "task_type": record.get("task_type"),
        "video_type": record.get("video_type"),
        "qa_id": record.get("QA_id"),
    }


def main() -> None:
    """Build the pilot split (and the complementary held-out pool)."""
    args = parse_args()
    records = json.loads(Path(args.source).read_text())

    by_video: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_video.setdefault(Path(str(record["video_path"])).name, []).append(record)

    videos = sorted(by_video)
    random.Random(PILOT_SEED).shuffle(videos)

    pilot_videos: list[str] = []
    pilot_items: list[dict[str, object]] = []
    for video in videos:
        if len(pilot_items) >= args.n_pilot:
            break
        pilot_videos.append(video)
        pilot_items.extend(to_conversation(r) for r in by_video[video])
    pilot_items = pilot_items[: args.n_pilot]
    # Keep only videos actually represented after truncation, so the manifest is exact.
    kept = {str(item["video"]) for item in pilot_items}
    pilot_videos = [v for v in pilot_videos if v in kept]

    if args.list_videos:
        for video in pilot_videos:
            print(video)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = out_dir / "avut_pilot_300.jsonl"
    pilot_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pilot_items) + "\n")

    heldout_videos = [v for v in videos if v not in kept]
    manifest = {
        "source": str(args.source),
        "seed": PILOT_SEED,
        "pilot_items": len(pilot_items),
        "pilot_videos": len(pilot_videos),
        "heldout_videos": len(heldout_videos),
        "heldout_items": sum(len(by_video[v]) for v in heldout_videos),
        "videos": pilot_videos,
    }
    (out_dir / "avut_pilot_300_manifest.json").write_text(json.dumps(manifest, indent=2))

    overlap = kept & set(heldout_videos)
    assert not overlap, f"pilot/held-out video overlap: {sorted(overlap)[:5]}"

    print(f"wrote {pilot_path}: {len(pilot_items)} items / {len(pilot_videos)} videos")
    print(f"held out for the final test split: {len(heldout_videos)} videos, {manifest['heldout_items']} items")
    print("pilot/held-out video overlap: 0 (required by preregistration §6)")
    print(f"questions per video in the pilot: {len(pilot_items) / max(len(pilot_videos), 1):.2f}")


if __name__ == "__main__":
    main()
