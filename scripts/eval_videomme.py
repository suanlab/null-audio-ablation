"""Evaluate VideoLLM on Video-MME benchmark.

Video-MME is a comprehensive video understanding benchmark with 900 videos
and 2700 multi-choice QA pairs across multiple domains and durations.

Supports filtering by duration (short/medium/long) and domain, and reports
accuracy per domain, per duration, and overall.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_videomme.py \
        --checkpoint checkpoints/full_v6/stage3_instruction \
        --parquet data/Video-MME/videomme/test-00000-of-00001.parquet \
        --video_dir data/Video-MME/videos \
        --output eval_results/videomme_v6.json

    # Only evaluate on a specific domain
    python scripts/eval_videomme.py \
        --checkpoint checkpoints/stage3_instruction \
        --domain "Sports Competition"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer

from videollm.data.constants import (
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_DURATION,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_NUM_FRAMES,
    VIDEO_TOKEN_INDEX,
)
from videollm.data.dataset import uniform_frame_sample
from videollm.data.transforms import AudioTransform, VideoTransform
from videollm.model.videollm import ModelConfig, VideoLLM
from videollm.utils import load_audio_from_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHOICE_RE = re.compile(r"[A-D]")


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments.

    Returns:
        Parsed CLI namespace with model config, eval paths, and filter options.
    """
    parser = argparse.ArgumentParser(description="Evaluate VideoLLM on Video-MME")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/Video-MME/videomme/test-00000-of-00001.parquet",
        help="Path to Video-MME parquet file",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data/Video-MME/videos",
        help="Directory containing downloaded MP4 files",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_temporal_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm_projector_type", type=str, default="stc", choices=["stc", "mlp", "linear"])
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="Filter by video duration",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain (e.g., 'Knowledge', 'Sports Competition')",
    )
    return parser.parse_args()


def load_videollm(args: argparse.Namespace, device: torch.device) -> tuple[VideoLLM, AutoTokenizer]:
    """Build VideoLLM and load trained weights from checkpoint.

    Args:
        args: CLI arguments with model config flags.
        device: Target device.

    Returns:
        Tuple of (model, tokenizer).
    """
    config = ModelConfig(
        mm_projector_type=args.mm_projector_type,
        use_audio=args.use_audio,
        use_temporal_bridge=args.use_temporal_bridge,
        use_lora=args.use_lora,
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=not args.use_lora,
    )
    model = VideoLLM(config)

    ckpt_path = Path(args.checkpoint) / "model.safetensors"
    if ckpt_path.exists():
        load_model(model, str(ckpt_path), strict=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)
    else:
        logger.warning("No model.safetensors found at %s, using fresh weights", ckpt_path)

    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.llm_path, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_video_frames(video_path: str, num_frames: int, transform: VideoTransform) -> torch.Tensor:
    """Load and preprocess video frames.

    Args:
        video_path: Path to MP4 file.
        num_frames: Number of frames to sample.
        transform: Video transform pipeline.

    Returns:
        Tensor of shape ``(T, C, H, W)``.
    """
    import decord

    decord.bridge.set_bridge("torch")
    try:
        vr = decord.VideoReader(video_path, num_threads=1)
        indices = uniform_frame_sample(len(vr), num_frames)
        frames = vr.get_batch(indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
        return transform(frames)
    except Exception:
        logger.warning("Failed to load video %s, using black frames", video_path)
        return transform(torch.zeros(num_frames, 3, 384, 384))


def load_audio_waveform(video_path: str, transform: AudioTransform) -> torch.Tensor:
    """Extract audio waveform from video.

    Args:
        video_path: Path to MP4 file.
        transform: Audio transform pipeline.

    Returns:
        Waveform tensor of shape ``(samples,)``.
    """
    try:
        waveform, sr = load_audio_from_video(video_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        return transform(waveform, sr)
    except Exception:
        duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
        return torch.zeros(duration_samples)


def build_videomme_prompt(question: str, options: list[str]) -> str:
    """Build a multi-choice prompt for Video-MME.

    Args:
        question: The question text.
        options: List of option strings (e.g., ['A. Apples.', 'B. Candles.', ...]).

    Returns:
        Formatted prompt string with <video> token.
    """
    choices_text = "\n".join(f"{opt}" for opt in options)
    return f"<video>\nQuestion: {question}\nChoices:\n{choices_text}\nAnswer with the letter."


def extract_answer_letter(text: str) -> str:
    """Extract the predicted choice letter (A-D) from model output.

    Args:
        text: Raw model output text.

    Returns:
        Single letter A-D, or empty string if not found.
    """
    match = CHOICE_RE.search(text.upper())
    return match.group(0) if match else ""


def run_evaluation(
    model: VideoLLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    video_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Run Video-MME evaluation loop.

    Args:
        model: Trained VideoLLM model.
        tokenizer: Tokenizer.
        df: Filtered Video-MME DataFrame.
        video_dir: Directory with video MP4 files.
        args: CLI arguments.
        device: Inference device.

    Returns:
        Dictionary with accuracy, per-domain/duration breakdowns, and per-sample predictions.
    """
    video_transform = VideoTransform(is_train=False)
    audio_transform = AudioTransform(is_train=False)

    correct = 0
    total = 0
    predictions: list[dict[str, str]] = []

    # Track per-domain and per-duration stats
    domain_stats: dict[str, dict[str, int]] = {}
    duration_stats: dict[str, dict[str, int]] = {}

    for _idx, row in df.iterrows():
        video_id = str(row["videoID"])
        question = str(row["question"])
        options_raw = row["options"]
        answer = str(row["answer"]).strip().upper()
        domain = str(row["domain"])
        duration = str(row["duration"])

        # Parse options: they come as a single string with newlines
        if isinstance(options_raw, str):
            options = [line.strip() for line in options_raw.strip().split("\n") if line.strip()]
        elif isinstance(options_raw, list):
            options = [str(o).strip() for o in options_raw]
        else:
            options = []

        # Validate ground truth
        if answer not in ("A", "B", "C", "D"):
            continue

        video_path = video_dir / f"{video_id}.mp4"

        # Load video
        pixel_values: torch.Tensor | None = None
        if video_path.exists():
            frames = load_video_frames(str(video_path), args.num_frames, video_transform)
            pixel_values = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)

        # Load audio
        waveforms: torch.Tensor | None = None
        if args.use_audio and video_path.exists():
            wf = load_audio_waveform(str(video_path), audio_transform)
            waveforms = wf.unsqueeze(0).to(device)  # (1, samples)

        # Build prompt
        prompt = build_videomme_prompt(question, options)
        clean_text = prompt.replace("<video>", "").replace("<audio>", "").strip()
        encoding = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoding["input_ids"].to(device)  # (1, S)
        attention_mask = encoding["attention_mask"].to(device)  # (1, S)

        # Prepend modal token indices
        modal_ids: list[int] = []
        if pixel_values is not None:
            modal_ids.append(VIDEO_TOKEN_INDEX)
        if waveforms is not None:
            modal_ids.append(AUDIO_TOKEN_INDEX)

        if modal_ids:
            modal_tensor = torch.tensor([modal_ids], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat([modal_tensor, input_ids], dim=1)
            modal_mask = torch.ones(1, len(modal_ids), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([modal_mask, attention_mask], dim=1)

        # Generate
        pred_letter = ""
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    waveforms=waveforms,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_letter = extract_answer_letter(decoded)
        except Exception as e:
            logger.warning("Inference failed for %s: %s", video_id, e)

        is_correct = pred_letter == answer
        if is_correct:
            correct += 1
        total += 1

        # Update domain stats
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1

        # Update duration stats
        if duration not in duration_stats:
            duration_stats[duration] = {"correct": 0, "total": 0}
        duration_stats[duration]["total"] += 1
        if is_correct:
            duration_stats[duration]["correct"] += 1

        predictions.append(
            {
                "question_id": str(row["question_id"]),
                "videoID": video_id,
                "domain": domain,
                "duration": duration,
                "sub_category": str(row.get("sub_category", "")),
                "gt": answer,
                "pred": pred_letter,
                "correct": str(is_correct),
                "video_loaded": str(video_path.exists()),
            }
        )

        if (total) % 10 == 0:
            acc = correct / total * 100 if total > 0 else 0.0
            logger.info("[%d/%d] accuracy: %.1f%% (%d/%d)", total, len(df), acc, correct, total)

    accuracy = correct / total * 100 if total > 0 else 0.0

    # Build per-domain breakdown
    domain_breakdown: dict[str, object] = {}
    for dom, stats in sorted(domain_stats.items()):
        domain_breakdown[dom] = {
            "accuracy": round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # Build per-duration breakdown
    duration_breakdown: dict[str, object] = {}
    for dur, stats in sorted(duration_stats.items()):
        duration_breakdown[dur] = {
            "accuracy": round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped_missing_video": len(df) - total,
        "per_domain": domain_breakdown,
        "per_duration": duration_breakdown,
        "predictions": predictions,
    }


def main() -> None:
    """Run Video-MME evaluation."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and filter data
    df = pd.read_parquet(args.parquet)
    logger.info("Loaded %d QA pairs (%d unique videos) from %s", len(df), df["videoID"].nunique(), args.parquet)

    # Check available videos
    video_dir = Path(args.video_dir)
    downloaded_ids = set(f.stem for f in video_dir.glob("*.mp4"))
    df = df[df["videoID"].isin(downloaded_ids)]
    logger.info("Filtered to %d QA pairs with downloaded videos", len(df))

    # Apply filters
    if args.duration:
        before = len(df)
        df = df[df["duration"] == args.duration]
        logger.info("Duration filter '%s': %d -> %d QA pairs", args.duration, before, len(df))

    if args.domain:
        before = len(df)
        df = df[df["domain"] == args.domain]
        logger.info("Domain filter '%s': %d -> %d QA pairs", args.domain, before, len(df))

    logger.info(
        "Final eval set: %d QA pairs, %d videos, domains: %s",
        len(df),
        df["videoID"].nunique(),
        ", ".join(sorted(df["domain"].unique())),
    )

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model, tokenizer = load_videollm(args, device)

    # Run evaluation
    t0 = time.time()
    results = run_evaluation(model, tokenizer, df, video_dir, args, device)
    elapsed = time.time() - t0

    results["elapsed_seconds"] = round(elapsed, 1)
    results["checkpoint"] = args.checkpoint
    results["config"] = {
        "use_audio": args.use_audio,
        "use_temporal_bridge": args.use_temporal_bridge,
        "mm_projector_type": args.mm_projector_type,
        "total_videos_in_benchmark": 900,
        "downloaded_videos": len(downloaded_ids),
        "filters": {"duration": args.duration, "domain": args.domain},
    }

    logger.info("=" * 60)
    logger.info(
        "Video-MME Evaluation: %.1f%% (%d/%d) in %.1fs",
        results["accuracy"],
        results["correct"],
        results["total"],
        elapsed,
    )
    logger.info("Per-duration breakdown:")
    for dur, stats in results["per_duration"].items():
        logger.info("  %s: %.1f%% (%d/%d)", dur, stats["accuracy"], stats["correct"], stats["total"])
    logger.info("Per-domain breakdown:")
    for dom, stats in results["per_domain"].items():
        logger.info("  %s: %.1f%% (%d/%d)", dom, stats["accuracy"], stats["correct"], stats["total"])

    output_path = args.output
    if output_path is None:
        ckpt_name = Path(args.checkpoint).name
        output_path = f"eval_results/videomme_{ckpt_name}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
