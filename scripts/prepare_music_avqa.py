"""Build a MUSIC-AVQA pilot split in the repo's eval JSONL shape.

This replaces the previous anchor domain. That anchor was labelled "MUSIC-AVQA" throughout
the codebase but was in fact a VGGSound-derived audio-event set: five templated question
stems over VGGSound class labels with three random distractors, in which audio-sufficiency
holds by construction. An adversarial review caught the misattribution; this module exists
so the anchor is the real Li et al. (CVPR 2022) benchmark and so its provenance is
checkable from the manifest rather than taken on trust.

Three properties of the real benchmark shape what this script does:

* **The answer set is closed.** Every answer in the official test split is one of 41 words,
  and the benchmark's own baselines classify over that vocabulary. We therefore present the
  full vocabulary as candidates and score by normalised exact match. This keeps scoring
  deterministic, which the recent Video-LLM convention (open generation judged by GPT-3.5)
  does not: our estimand is a *paired difference* between audio conditions on the same
  item, so judge noise would enter every cell of every interval.
* **The chance floor is not uniform.** Comparative and Existential questions are binary
  (majority-class baseline 51-54%) while Location and Temporal span 24-27 answers. No
  single chance floor applies, which is why the protocol measures the no-information floor
  empirically with the blank-video + silent control instead of assuming one.
* **Prompts cannot match AVUT's.** AVUT is 4-way multiple choice; this is 41-way. Making
  the two identical would distort one of them. This is safe for the estimand -- Delta_A is
  a within-benchmark paired difference over identical prompts, so prompt format cancels --
  but it does mean absolute accuracies are not comparable across domains, and the manifest
  records that.

The split is held out by whole video (preregistration section 6) and is restricted to
videos actually present in the official real-video archive, so the synthetic subset never
enters.

Usage::

    python scripts/prepare_music_avqa.py --n_pilot 300
    python scripts/prepare_music_avqa.py --n_pilot 300 --dry_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "raw" / "music_avqa" / "avqa-test-update.json"
VIDEO_DIR = REPO_ROOT / "data" / "raw" / "music_avqa" / "videos" / "MUSIC-AVQA-videos-Real"
PILOT_SEED = 2027  # same seed as the preregistered bootstrap; fixed, not tuned
PLACEHOLDER = re.compile(r"<[^>]+>")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare a MUSIC-AVQA pilot split")
    parser.add_argument("--n_pilot", type=int, default=300, help="Target pilot item count")
    parser.add_argument("--source", type=str, default=str(SOURCE), help="Official test annotations")
    parser.add_argument("--video_dir", type=str, default=str(VIDEO_DIR), help="Extracted real videos")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "data" / "instruct"))
    parser.add_argument("--dry_run", action="store_true", help="Report the split, write nothing")
    return parser.parse_args()


def fill_template(question: str, templ_values: str) -> str:
    """Substitute a record's template values into its question, left to right.

    Args:
        question: ``question_content``, e.g. ``"Is the <Object> always playing?"``.
        templ_values: The record's ``templ_values`` JSON string, e.g. ``'["violin"]'``.

    Returns:
        The question with every placeholder replaced.

    Raises:
        ValueError: If the placeholder count does not match the value count, which would
            silently produce a malformed question.
    """
    values = json.loads(templ_values)
    holes = PLACEHOLDER.findall(question)
    if len(holes) != len(values):
        raise ValueError(f"{len(holes)} placeholders but {len(values)} values in {question!r}")
    out = question
    for value in values:
        out = PLACEHOLDER.sub(str(value).replace("_", " "), out, count=1)
    return out


def to_conversation(record: dict[str, object], vocabulary: list[str]) -> dict[str, object]:
    """Render one MUSIC-AVQA record in the repo's eval JSONL shape."""
    question = fill_template(str(record["question_content"]), str(record["templ_values"]))
    answer = str(record["anser"])
    options = ", ".join(v.replace("_", " ") for v in vocabulary)
    return {
        "video": f"{record['video_id']}.mp4",
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"<video>\n<audio>\n{question}\n"
                    f"Answer with exactly one of the following words:\n{options}\n"
                    f"Answer:"
                ),
            },
            {"from": "gpt", "value": answer.replace("_", " ")},
        ],
        "type": record["type"],
        "question_id": record["question_id"],
        "answer_raw": answer,
    }


def main() -> None:
    """Build the pilot split and a provenance manifest."""
    args = parse_args()
    source = Path(args.source)
    records = [r for r in json.loads(source.read_text()) if not r.get("question_deleted")]

    video_dir = Path(args.video_dir)
    available = {p.stem for p in video_dir.glob("*.mp4")} if video_dir.exists() else set()
    if not available:
        raise SystemExit(f"no videos found under {video_dir} -- extract the real archive first")
    usable = [r for r in records if str(r["video_id"]) in available]

    vocabulary = sorted({str(r["anser"]) for r in records})
    by_video: dict[str, list[dict[str, object]]] = {}
    for record in usable:
        by_video.setdefault(str(record["video_id"]), []).append(record)

    videos = sorted(by_video)
    random.Random(PILOT_SEED).shuffle(videos)

    pilot_videos: list[str] = []
    pilot_items: list[dict[str, object]] = []
    for video in videos:
        if len(pilot_items) >= args.n_pilot:
            break
        pilot_videos.append(video)
        pilot_items.extend(to_conversation(r, vocabulary) for r in by_video[video])
    pilot_items = pilot_items[: args.n_pilot]
    kept = {str(item["video"]).removesuffix(".mp4") for item in pilot_items}
    pilot_videos = [v for v in pilot_videos if v in kept]
    heldout = [v for v in videos if v not in kept]

    print(f"source              : {source} ({len(records)} active records)")
    print(f"videos in archive   : {len(available)}")
    print(f"records with a video: {len(usable)} over {len(by_video)} videos")
    print(f"answer vocabulary   : {len(vocabulary)} words")
    print(f"pilot               : {len(pilot_items)} items / {len(pilot_videos)} videos")
    print(f"held out for test   : {len(heldout)} videos, {sum(len(by_video[v]) for v in heldout)} items")
    print(f"questions per video : {len(pilot_items) / max(len(pilot_videos), 1):.2f}")
    assert not (kept & set(heldout)), "pilot/held-out video overlap"
    print("pilot/held-out video overlap: 0 (required by preregistration section 6)")
    if args.dry_run:
        print("\n[dry run] nothing written. Example item:")
        print(json.dumps(pilot_items[0], ensure_ascii=False, indent=2)[:900])
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = out_dir / "music_avqa_pilot_300.jsonl"
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in pilot_items) + "\n"
    pilot_path.write_text(payload)

    manifest = {
        "dataset": "MUSIC-AVQA (Li et al., CVPR 2022)",
        "annotations": "GeWu-Lab/MUSIC-AVQA data/json_update/avqa-test.json",
        "annotations_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "videos": "official real-video archive only; synthetic subset excluded",
        "scoring": "closed 41-word vocabulary presented as candidates; normalised exact match",
        "prompt_note": (
            "Prompts are NOT comparable to the AVUT domain (4-way multiple choice vs 41-way "
            "closed vocabulary). Delta_A is a within-benchmark paired difference so prompt "
            "format cancels, but absolute accuracies are not cross-domain comparable."
        ),
        "seed": PILOT_SEED,
        "pilot_items": len(pilot_items),
        "pilot_videos": len(pilot_videos),
        "heldout_videos": len(heldout),
        "answer_vocabulary": vocabulary,
        "pilot_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "videos_list": pilot_videos,
    }
    (out_dir / "music_avqa_pilot_300_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {pilot_path}")
    print(f"wrote {out_dir / 'music_avqa_pilot_300_manifest.json'}")


if __name__ == "__main__":
    main()
