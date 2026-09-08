#!/usr/bin/env python3
"""Convert raw dataset annotations to unified VideoLLM JSONL format.

Reads raw annotations from AudioCaps (CSV), VGGSound (CSV), and AVQA (JSON),
then produces JSONL files matching the VideoAudioDataset expected format:

    {"video": "relative/path.mp4", "conversations": [
        {"from": "human", "value": "<video>\\n<audio>\\nQuestion"},
        {"from": "gpt", "value": "Answer"}
    ]}

Usage:
    python scripts/prepare_data.py --data-root data/
    python scripts/prepare_data.py --data-root data/ --dataset audiocaps
    python scripts/prepare_data.py --data-root data/ --dataset avqa --max-samples 1000

Output layout:
    data/
    ├── alignment/
    │   ├── audiocaps_train.jsonl
    │   └── audiocaps_val.jsonl
    ├── bridge/
    │   ├── vggsound_train.jsonl
    │   └── vggsound_val.jsonl
    └── instruct/
        ├── avqa_train.jsonl
        └── avqa_val.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special tokens (must match src/videollm/data/constants.py)
# ---------------------------------------------------------------------------

VIDEO_TOKEN = "<video>"
AUDIO_TOKEN = "<audio>"


# ---------------------------------------------------------------------------
# AudioCaps → Stage 1 Alignment (caption generation)
# ---------------------------------------------------------------------------


def convert_audiocaps(
    raw_dir: Path,
    out_dir: Path,
    video_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Convert AudioCaps CSV annotations to JSONL for audio-visual captioning.

    Each sample becomes a single-turn conversation:
        Human: "<video>\\n<audio>\\nDescribe what you hear and see in this video."
        GPT: "<caption>"

    Args:
        raw_dir: Directory containing train.csv, val.csv, test.csv.
        out_dir: Output directory for JSONL files.
        video_dir: Directory containing downloaded video files.
        max_samples: Optional cap on samples per split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": "audiocaps_train.jsonl",
        "val": "audiocaps_val.jsonl",
        "test": "audiocaps_val.jsonl",  # merge test into val for simplicity
    }

    prompts = [
        "Describe what you hear and see in this video.",
        "What sounds and visual events are happening in this clip?",
        "Provide a detailed description of the audio and visual content.",
        "What can you hear and see? Describe in detail.",
        "Listen to the audio and watch the video. What is happening?",
    ]

    for split, out_name in split_map.items():
        csv_path = raw_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.warning("Skipping %s — %s not found", split, csv_path)
            continue

        out_path = out_dir / out_name
        # If we merge test into val, append instead of overwrite
        mode = "a" if split == "test" and out_path.exists() else "w"

        count = 0
        skipped = 0
        with open(csv_path, newline="") as f_in, open(out_path, mode) as f_out:
            reader = csv.DictReader(f_in)
            for row in reader:
                yt_id = row["youtube_id"]
                start = row["start_time"]
                caption = row["caption"].strip()

                video_file = f"{yt_id}_{start}.mp4"
                video_path = video_dir / video_file

                # Only include samples with downloaded videos
                if not video_path.exists():
                    skipped += 1
                    continue

                if max_samples is not None and count >= max_samples:
                    break

                prompt = prompts[count % len(prompts)]
                sample = {
                    "video": video_file,
                    "conversations": [
                        {"from": "human", "value": f"{VIDEO_TOKEN}\n{AUDIO_TOKEN}\n{prompt}"},
                        {"from": "gpt", "value": caption},
                    ],
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1

        logger.info(
            "AudioCaps %s → %s: %d samples written, %d skipped (missing video)",
            split,
            out_path.name,
            count,
            skipped,
        )


# ---------------------------------------------------------------------------
# VGGSound → Stage 2 Bridge (audio-visual classification as conversation)
# ---------------------------------------------------------------------------


def convert_vggsound(
    raw_dir: Path,
    out_dir: Path,
    video_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Convert VGGSound CSV to JSONL for audio-visual bridge training.

    Each sample becomes a classification-as-conversation task:
        Human: "<video>\\n<audio>\\nWhat sound is being made in this video?"
        GPT: "<label>"

    Args:
        raw_dir: Directory containing vggsound.csv.
        out_dir: Output directory for JSONL files.
        video_dir: Directory containing downloaded video files.
        max_samples: Optional cap on samples per split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "vggsound.csv"
    if not csv_path.exists():
        logger.error("VGGSound CSV not found at %s", csv_path)
        return

    prompts = [
        "What sound is being made in this video?",
        "Identify the sound you hear in this clip.",
        "What is the source of the sound in this video?",
        "Describe the audio event happening in this video.",
        "What produces the sound you hear?",
    ]

    train_samples: list[dict[str, object]] = []
    test_samples: list[dict[str, object]] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 4:
                continue

            yt_id = row[0].strip()
            start = row[1].strip()
            label = row[2].strip()
            split = row[3].strip()

            video_file = f"{yt_id}_{start}.mp4"
            video_path = video_dir / video_file

            if not video_path.exists():
                continue

            prompt = prompts[i % len(prompts)]
            sample = {
                "video": video_file,
                "conversations": [
                    {"from": "human", "value": f"{VIDEO_TOKEN}\n{AUDIO_TOKEN}\n{prompt}"},
                    {"from": "gpt", "value": label},
                ],
            }

            if split == "train":
                train_samples.append(sample)
            else:
                test_samples.append(sample)

    # Shuffle and cap
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    if max_samples is not None:
        train_samples = train_samples[:max_samples]
        test_samples = test_samples[: max(max_samples // 10, 100)]

    _write_jsonl(out_dir / "vggsound_train.jsonl", train_samples)
    _write_jsonl(out_dir / "vggsound_val.jsonl", test_samples)

    logger.info("VGGSound → train: %d, val: %d", len(train_samples), len(test_samples))


# ---------------------------------------------------------------------------
# AVQA → Stage 3 Instruction (multi-choice QA as conversation)
# ---------------------------------------------------------------------------


def convert_avqa(
    raw_dir: Path,
    out_dir: Path,
    video_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Convert AVQA JSON to JSONL for instruction-tuning.

    Each sample becomes a QA conversation with choices:
        Human: "<video>\\n<audio>\\n<question>\\nChoices:\\nA. ...\\nB. ...\\nC. ...\\nD. ..."
        GPT: "The answer is <letter>. <correct_choice>"

    Args:
        raw_dir: Directory containing train_qa.json, val_qa.json.
        out_dir: Output directory for JSONL files.
        video_dir: Directory containing downloaded video files.
        max_samples: Optional cap on samples per split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train_qa.json": "avqa_train.jsonl",
        "val_qa.json": "avqa_val.jsonl",
    }
    choice_letters = ["A", "B", "C", "D"]

    for src_name, out_name in split_map.items():
        src_path = raw_dir / src_name
        if not src_path.exists():
            logger.warning("Skipping AVQA %s — file not found", src_path)
            continue

        with open(src_path) as f:
            raw_data = json.load(f)

        samples: list[dict[str, object]] = []
        skipped = 0

        for item in raw_data:
            video_name = item.get("video_name", "")
            video_file = f"{video_name}.mp4"
            video_path = video_dir / video_file

            # AVQA uses zero-padded starts (e.g., -03N_1zOM4E_000239)
            # Our downloads use unpadded (e.g., -03N_1zOM4E_239)
            if not video_path.exists():
                parts = video_name.rsplit("_", 1)
                if len(parts) == 2:
                    yt_id, start_padded = parts
                    try:
                        start_int = int(start_padded)
                        alt_file = f"{yt_id}_{start_int}.mp4"
                        alt_path = video_dir / alt_file
                        if alt_path.exists():
                            video_file = alt_file
                            video_path = alt_path
                    except ValueError:
                        pass

            if not video_path.exists():
                skipped += 1
                continue

            question = item.get("question_text", "")
            choices = item.get("multi_choice", [])
            answer_idx = item.get("answer", 0)

            if not question or len(choices) < 4:
                skipped += 1
                continue

            # Format choices
            choices_text = "\n".join(f"{choice_letters[j]}. {choices[j]}" for j in range(min(len(choices), 4)))
            answer_letter = choice_letters[answer_idx] if answer_idx < 4 else "A"
            answer_text = choices[answer_idx] if answer_idx < len(choices) else choices[0]

            # Build conversation with modality context
            modality = item.get("question_relation", "Both")
            if modality == "View":
                modal_tokens = f"{VIDEO_TOKEN}\n"
            elif modality == "Sound":
                modal_tokens = f"{AUDIO_TOKEN}\n"
            else:
                modal_tokens = f"{VIDEO_TOKEN}\n{AUDIO_TOKEN}\n"

            human_text = f"{modal_tokens}{question}\nChoices:\n{choices_text}\nAnswer with the letter."
            gpt_text = f"The answer is {answer_letter}. {answer_text}"

            sample = {
                "video": video_file,
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": gpt_text},
                ],
            }
            samples.append(sample)

        random.shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]

        _write_jsonl(out_dir / out_name, samples)
        logger.info(
            "AVQA %s → %s: %d samples, %d skipped",
            src_name,
            out_name,
            len(samples),
            skipped,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, samples: list[dict[str, object]]) -> None:
    """Write a list of samples as JSONL.

    Args:
        path: Output file path.
        samples: List of dicts to serialize.
    """
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Wrote %d samples to %s", len(samples), path)


def generate_stats(data_root: Path) -> None:
    """Print dataset statistics after conversion.

    Args:
        data_root: Base data directory.
    """
    print("\n=== Dataset Statistics ===\n")
    for stage_dir in ["alignment", "bridge", "instruct"]:
        stage_path = data_root / stage_dir
        if not stage_path.exists():
            continue
        print(f"[{stage_dir}]")
        for jsonl_file in sorted(stage_path.glob("*.jsonl")):
            with open(jsonl_file) as fh:
                count = sum(1 for _ in fh)
            print(f"  {jsonl_file.name}: {count:,} samples")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run dataset conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert raw dataset annotations to VideoLLM training format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["audiocaps", "vggsound", "avqa", "all"],
        help="Which dataset to convert (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for debugging / quick runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    raw_dir = data_root / "raw"
    video_dir = data_root / "videos"

    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        logger.error("Run 'bash scripts/download_data.sh' first.")
        sys.exit(1)

    if args.dataset in ("audiocaps", "all"):
        convert_audiocaps(
            raw_dir=raw_dir / "audiocaps",
            out_dir=data_root / "alignment",
            video_dir=video_dir,
            max_samples=args.max_samples,
        )

    if args.dataset in ("vggsound", "all"):
        convert_vggsound(
            raw_dir=raw_dir / "vggsound",
            out_dir=data_root / "bridge",
            video_dir=video_dir,
            max_samples=args.max_samples,
        )

    if args.dataset in ("avqa", "all"):
        convert_avqa(
            raw_dir=raw_dir / "avqa",
            out_dir=data_root / "instruct",
            video_dir=video_dir,
            max_samples=args.max_samples,
        )

    generate_stats(data_root)


if __name__ == "__main__":
    main()
