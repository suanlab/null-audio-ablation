"""Generate small dummy video files and JSONL annotations for pipeline testing.

Creates synthetic MP4 videos (with audio) using torchvision and torchaudio,
plus JSONL annotation files for each training stage. Designed to verify the
full train.py pipeline without downloading real datasets.

Usage::

    python scripts/create_dummy_data.py --output_dir data/dummy --num_samples 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torchvision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_NUM_SAMPLES = 8
DEFAULT_FPS = 8
DEFAULT_DURATION_S = 2
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_SAMPLE_RATE = 48000


def create_dummy_video(
    path: Path,
    num_frames: int = 16,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    fps: int = DEFAULT_FPS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Write a small synthetic MP4 with random video + sine-wave audio.

    Args:
        path: Output .mp4 file path.
        num_frames: Number of video frames.
        height: Frame height in pixels.
        width: Frame width in pixels.
        fps: Frames per second.
        sample_rate: Audio sample rate.
    """
    # Random RGB frames: (T, H, W, C) uint8
    frames = torch.randint(0, 256, (num_frames, height, width, 3), dtype=torch.uint8)

    duration_s = num_frames / fps
    num_audio_samples = int(duration_s * sample_rate)
    freq = 440.0 + torch.randint(0, 400, (1,)).item()
    t = torch.linspace(0, duration_s, num_audio_samples)
    waveform = (torch.sin(2 * torch.pi * freq * t) * 0.5).unsqueeze(0)  # (1, samples)

    path.parent.mkdir(parents=True, exist_ok=True)

    torchvision.io.write_video(
        str(path),
        frames,
        fps=fps,
        audio_array=waveform,
        audio_fps=sample_rate,
        audio_codec="aac",
    )


def create_sft_annotations(
    video_names: list[str],
    output_path: Path,
) -> None:
    """Create JSONL for SFT training (stages 1-3).

    Args:
        video_names: List of video filenames (relative to video_dir).
        output_path: Output JSONL path.
    """
    prompts = [
        "Describe what happens in this video.",
        "What sounds can you hear in the video?",
        "Summarize the visual and audio content.",
        "What actions are shown in this clip?",
    ]
    responses = [
        "The video shows a colorful scene with various objects moving across the frame while ambient sounds play.",
        "There are sounds of music and environmental noise accompanying the visual content in the video.",
        "The clip features dynamic visual elements with synchronized audio creating an engaging experience.",
        "Several actions are depicted including movement and interaction between objects in the scene.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, name in enumerate(video_names):
            ann = {
                "video": name,
                "conversations": [
                    {"from": "human", "value": f"<video>\n<audio>\n{prompts[i % len(prompts)]}"},
                    {"from": "gpt", "value": responses[i % len(responses)]},
                ],
            }
            f.write(json.dumps(ann) + "\n")
    logger.info("Wrote %d SFT annotations to %s", len(video_names), output_path)


def create_dpo_annotations(
    video_names: list[str],
    output_path: Path,
) -> None:
    """Create JSONL for DPO training (stage 4).

    Args:
        video_names: List of video filenames (relative to video_dir).
        output_path: Output JSONL path.
    """
    chosen_responses = [
        "The video displays a vivid scene with multiple objects interacting dynamically. "
        "The audio features a clear melodic tone accompanying the visual action.",
        "A colorful sequence unfolds showing movement and visual complexity, "
        "paired with distinct environmental audio that enhances the viewing experience.",
    ]
    rejected_responses = [
        "A video.",
        "Some stuff happens.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, name in enumerate(video_names):
            ann = {
                "video": name,
                "prompt": "<video>\n<audio>\nDescribe this video in detail.",
                "chosen": chosen_responses[i % len(chosen_responses)],
                "rejected": rejected_responses[i % len(rejected_responses)],
            }
            f.write(json.dumps(ann) + "\n")
    logger.info("Wrote %d DPO annotations to %s", len(video_names), output_path)


def main() -> None:
    """Generate dummy data for pipeline testing."""
    parser = argparse.ArgumentParser(description="Create dummy VideoLLM training data")
    parser.add_argument("--output_dir", type=str, default="data/dummy")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_names: list[str] = []
    for i in range(args.num_samples):
        name = f"dummy_{i:04d}.mp4"
        video_path = video_dir / name
        logger.info("Creating %s (%d/%d)", name, i + 1, args.num_samples)
        create_dummy_video(
            video_path,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
        )
        video_names.append(name)

    create_sft_annotations(video_names, output_dir / "sft_train.jsonl")
    create_dpo_annotations(video_names, output_dir / "dpo_train.jsonl")

    logger.info("Done! Dummy data in %s", output_dir)
    logger.info("  Videos: %s/ (%d files)", video_dir, len(video_names))
    logger.info("  SFT:    %s", output_dir / "sft_train.jsonl")
    logger.info("  DPO:    %s", output_dir / "dpo_train.jsonl")


if __name__ == "__main__":
    main()
