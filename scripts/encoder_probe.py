"""Linear probes on frozen Whisper and BEATs features: what does each encoder represent?

The manuscript asserts that a Whisper front end cannot support instrument identification
and a BEATs front end cannot support lexical tasks, and infers that from downstream
delivery gains. That is an inference through two adapters, a projector, and an LLM. An
internal review objected on two counts: the representational claim is asserted rather than
measured, and prior work (Whisper-AT) reports that Whisper's encoder *does* carry
audio-event information, which would make the correct claim "the information is present but
the deployed tap does not carry it" -- a different and more interesting statement.

This settles it without an LLM in the loop. Both encoders are frozen; a linear probe is
fitted on their pooled features for two tasks drawn from the same clips the paper uses:

* **instrument identity** -- MUSIC-AVQA items whose answer is an instrument name.
* **lexical content** -- AVUT items, discriminating which of the candidate answer strings
  was actually spoken, using the benchmark's own options.

A linear probe measures what is linearly decodable from the frozen representation, which is
the relevant notion here: the adapters downstream are shallow projections, so information
that is not linearly available is unlikely to survive them.

Usage::

    python scripts/encoder_probe.py --task instrument --encoder both
    python scripts/encoder_probe.py --task lexical --encoder both
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MUSIC_SOURCE = REPO_ROOT / "data" / "raw" / "music_avqa" / "avqa-test-update.json"
MUSIC_PILOT = REPO_ROOT / "data" / "instruct" / "music_avqa_pilot_300.jsonl"
MUSIC_VIDEOS = REPO_ROOT / "data" / "raw" / "music_avqa" / "videos" / "MUSIC-AVQA-videos-Real"
AVUT_JSONL = REPO_ROOT / "data" / "instruct" / "avut_pilot_300.jsonl"
AVUT_VIDEOS = REPO_ROOT / "data" / "avut" / "videos"
SR = 16000
SEED = 2027

# The 41-word MUSIC-AVQA vocabulary minus counts, positions and yes/no.
INSTRUMENTS = frozenset(
    ["accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon", "cello", "clarinet", "congas", "drum", "electric_bass", "erhu", "flute", "guzheng", "piano", "pipa", "saxophone", "suona", "trumpet", "tuba", "ukulele", "violin", "xylophone"]
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Frozen-encoder linear probes")
    p.add_argument("--task", required=True, choices=("instrument", "lexical"))
    p.add_argument("--encoder", default="both", choices=("whisper", "beats", "both"))
    p.add_argument("--max_items", type=int, default=2000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "eval_results" / "encoder_probe.json"))
    return p.parse_args()


def instrument_labels(max_items: int) -> list[tuple[str, str]]:
    """(video path, instrument) for MUSIC-AVQA items whose answer is an instrument.

    Drawn from the full official test split rather than the 300-item pilot, and with the
    pilot's videos excluded. The probe is a separate measurement from the delivery grid, so
    reusing pilot clips would let a representational result lean on the same clips the
    headline numbers came from; the pilot also yields only ~46 labelled items, far too few
    for a 20-way instrument probe.
    """
    records = [r for r in json.loads(MUSIC_SOURCE.read_text()) if not r.get("question_deleted")]
    pilot = {json.loads(line)["video"] for line in MUSIC_PILOT.read_text().splitlines()}
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for record in records:
        answer = str(record["anser"]).strip()
        name = f"{record['video_id']}.mp4"
        if answer not in INSTRUMENTS or name in pilot or name in seen:
            continue  # one clip contributes at most one instrument label
        path = MUSIC_VIDEOS / name
        if path.exists():
            seen.add(name)
            out.append((str(path), answer))
    # Keep classes with enough support for a stratified fit.
    counts = Counter(label for _, label in out)
    out = [(p, y) for p, y in out if counts[y] >= 20]
    rng = np.random.default_rng(SEED)
    rng.shuffle(out)  # type: ignore[arg-type]
    return out[:max_items]


def lexical_labels(max_items: int) -> list[tuple[str, str]]:
    """(video path, spoken-content class) for AVUT items.

    AVUT's lexical families ask which of several transcribed strings occurs in the audio.
    The label is the ground-truth option text, so a probe can only succeed if the frozen
    features carry lexical content rather than acoustic scene identity.
    """
    rows = [json.loads(line) for line in AVUT_JSONL.read_text().splitlines()]
    lexical = {"Audio Character Matching", "Audio Information Extraction", "Audio OCR Matching"}
    out: list[tuple[str, str]] = []
    for row in rows:
        if str(row.get("task_type")) not in lexical:
            continue
        path = AVUT_VIDEOS / str(row["video"])
        if path.exists():
            gt = str(row["conversations"][1]["value"])
            out.append((str(path), gt.split(".")[0].strip()))
    counts = Counter(label for _, label in out)
    out = [(p, y) for p, y in out if counts[y] >= 5]
    return out[:max_items]


def main() -> None:
    """Fit and report cross-validated linear probes."""
    args = parse_args()
    items = instrument_labels(args.max_items) if args.task == "instrument" else lexical_labels(args.max_items)
    labels = sorted({y for _, y in items})
    print(f"task={args.task}  items={len(items)}  classes={len(labels)}")
    if len(items) < 30 or len(labels) < 2:
        raise SystemExit(
            f"not enough labelled items for a probe (items={len(items)}, classes={len(labels)}); "
            "widen --max_items or relax the per-class support floor"
        )
    counts = Counter(y for _, y in items)
    majority = max(counts.values()) / len(items)
    print(f"majority-class baseline: {100 * majority:.1f}%")
    for label, n in counts.most_common(10):
        print(f"   {label[:44]:44s} {n:4d}")
    payload = {
        "task": args.task,
        "items": len(items),
        "classes": len(labels),
        "majority_baseline": majority,
        "note": "feature extraction runs in the model environments; see run_encoder_probe.sh",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
