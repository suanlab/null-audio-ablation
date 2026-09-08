"""Fast evaluation for VideoLLM on AVQA with logit-based scoring.

Instead of autoregressive generation (slow, ~17s/sample), this script does a
single forward pass and picks the highest-logit choice token (A/B/C/D).
This is ~5-8x faster than the generate-based eval_avqa.py (~2-3s/sample).

Supports all audio ablation modes: real, shuffled, shifted, noise, silent.

Supports all audio ablation modes: real, shuffled, shifted, noise, silent.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_avqa_fast.py \\
        --checkpoint checkpoints/audio_quick_bridge_v2/stage3_instruction \\
        --eval_data data/instruct/avqa_train_quick5k.jsonl \\
        --audio_mode real \\
        --output eval_results/train_utility_real.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Token IDs for A, B, C, D in Qwen2 tokenizer
CHOICE_TOKEN_IDS = [32, 33, 34, 35]  # A=32, B=33, C=34, D=35
CHOICE_LETTERS = ["A", "B", "C", "D"]

AUDIO_MODES = ("real", "shuffled", "shifted", "noise", "silent")


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments.

    Returns:
        Parsed CLI namespace with model config, eval paths, and audio ablation settings.
    """
    parser = argparse.ArgumentParser(description="Fast AVQA evaluation (logit-based)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--eval_data", type=str, default="data/instruct/avqa_val.jsonl")
    parser.add_argument("--video_dir", type=str, default="data/videos")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_temporal_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm_projector_type", type=str, default="stc", choices=["stc", "mlp", "linear"])
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Ignored (kept for CLI compat with eval_avqa.py)")
    parser.add_argument("--audio_mode", type=str, default="real", choices=list(AUDIO_MODES))
    parser.add_argument("--shift_seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
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
    """Load and preprocess video frames with thread-based timeout.

    Uses ctypes.pythonapi.PyThreadState_SetAsyncExc to force-kill stuck decord
    threads that block inside C extensions (where signal handlers can't fire).

    Args:
        video_path: Path to MP4 file.
        num_frames: Number of frames to sample.
        transform: Video transform pipeline.

    Returns:
        Tensor of shape ``(T, C, H, W)``.
    """
    import ctypes
    import threading

    import decord

    result_holder: list[torch.Tensor] = []
    error_holder: list[str] = []

    def _worker() -> None:
        try:
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(video_path, num_threads=1)
            indices = uniform_frame_sample(len(vr), num_frames)
            frames = vr.get_batch(indices)  # (T, H, W, C)
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
            result_holder.append(frames)
        except Exception as exc:
            error_holder.append(str(exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=30)

    if thread.is_alive():
        tid = thread.ident
        if tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), ctypes.py_object(SystemExit)
            )
        logger.warning("Video load TIMED OUT for %s after 30s — using black frames", video_path)
        return transform(torch.zeros(num_frames, 3, 384, 384))

    if error_holder:
        logger.warning("Video load failed for %s: %s — using black frames", video_path, error_holder[0])
        return transform(torch.zeros(num_frames, 3, 384, 384))

    if result_holder:
        return transform(result_holder[0])

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

def get_ablated_audio(
    audio_mode: str,
    video_path: Path,
    audio_transform: AudioTransform,
    eval_data: list[dict[str, object]],
    video_dir: Path,
    sample_idx: int,
    shuffled_mapping: list[int],
    shift_seconds: float,
) -> torch.Tensor:
    """Load audio waveform according to the specified ablation mode.

    Args:
        audio_mode: One of ``real``, ``shuffled``, ``shifted``, ``noise``, ``silent``.
        video_path: Path to the current sample's video file.
        audio_transform: Audio transform pipeline.
        eval_data: Full list of evaluation samples (for shuffled mode).
        video_dir: Root directory containing video files.
        sample_idx: Current sample index in eval_data.
        shuffled_mapping: Pre-computed shuffled index mapping.
        shift_seconds: Number of seconds to shift audio (for shifted mode).

    Returns:
        Waveform tensor of shape ``(samples,)``.
    """
    duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)  # 480000

    if audio_mode == "real":
        return load_audio_waveform(str(video_path), audio_transform)
    if audio_mode == "silent":
        return torch.zeros(duration_samples)
    if audio_mode == "noise":
        return torch.randn(duration_samples)
    if audio_mode == "shifted":
        waveform = load_audio_waveform(str(video_path), audio_transform)
        shift_samples = int(shift_seconds * DEFAULT_AUDIO_SAMPLE_RATE)
        return torch.roll(waveform, shifts=shift_samples, dims=0)
    if audio_mode == "shuffled":
        donor_idx = shuffled_mapping[sample_idx]
        donor_sample = eval_data[donor_idx]
        donor_video = str(donor_sample.get("video", ""))
        donor_path = video_dir / donor_video
        if donor_path.exists():
            return load_audio_waveform(str(donor_path), audio_transform)
        return torch.zeros(duration_samples)
    raise ValueError(f"Unknown audio_mode: {audio_mode}")


def build_shuffled_mapping(num_samples: int, seed: int) -> list[int]:
    """Build a deterministic rejection-sampled derangement (no fixed points).

    Matches the contract used by the external adapters so that cross-model
    shuffled results are on the IDENTICAL donor mapping for the same
    (num_samples, seed). CHECK.md B.1 / L.1.
    """
    rng = random.Random(seed)
    idx = list(range(num_samples))
    for _ in range(100):
        perm = idx[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(num_samples)):
            return perm
    return [(i + 1) % num_samples for i in range(num_samples)]


def extract_ground_truth(gpt_response: str) -> str:
    """Extract ground truth letter from GPT response like 'The answer is A. ...'

    Args:
        gpt_response: The GPT conversation turn text.

    Returns:
        Ground truth letter A-D.
    """
    match = re.search(r"answer is ([A-D])", gpt_response, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def prepare_sample_data(
    sample: dict[str, object],
    sample_idx: int,
    args: argparse.Namespace,
    video_dir: Path,
    video_transform: VideoTransform,
    audio_transform: AudioTransform,
    eval_data: list[dict[str, object]],
    shuffled_mapping: list[int],
) -> dict[str, object] | None:
    """Prepare video/audio tensors for a single sample (CPU-side, thread-safe).

    Args:
        sample: Raw AVQA sample dict.
        sample_idx: Index in eval_data.
        args: CLI arguments.
        video_dir: Video directory path.
        video_transform: Video preprocessing.
        audio_transform: Audio preprocessing.
        eval_data: Full dataset (for shuffled mode).
        shuffled_mapping: Shuffled audio mapping.

    Returns:
        Dict with pre-loaded tensors, or None if sample is invalid.
    """
    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    human_turn = conversations[0]
    gpt_turn = conversations[1]
    if not isinstance(human_turn, dict) or not isinstance(gpt_turn, dict):
        return None

    question_text = str(human_turn.get("value", ""))
    gt_text = str(gpt_turn.get("value", ""))
    gt_letter = extract_ground_truth(gt_text)
    if not gt_letter:
        return None

    video_file = str(sample.get("video", ""))
    video_path = video_dir / video_file

    # Load video frames
    pixel_values = None
    if video_path.exists():
        pixel_values = load_video_frames(str(video_path), args.num_frames, video_transform)  # (T, C, H, W)

    # Load audio
    waveform = None
    if args.use_audio and video_path.exists():
        waveform = get_ablated_audio(
            audio_mode=args.audio_mode,
            video_path=video_path,
            audio_transform=audio_transform,
            eval_data=eval_data,
            video_dir=video_dir,
            sample_idx=sample_idx,
            shuffled_mapping=shuffled_mapping,
            shift_seconds=args.shift_seconds,
        )

    return {
        "pixel_values": pixel_values,
        "waveform": waveform,
        "question_text": question_text,
        "gt_letter": gt_letter,
        "video_file": video_file,
        "sample_idx": sample_idx,
    }


def run_fast_evaluation(
    model: VideoLLM,
    tokenizer: AutoTokenizer,
    eval_data: list[dict[str, object]],
    video_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Run fast AVQA evaluation using single forward pass + logit extraction.

    Processes samples sequentially (batch_size=1) with a single forward pass
    instead of autoregressive generation. This avoids decord thread-safety
    issues while still being ~5-8x faster than generate-based evaluation.

    Args:
        model: Trained VideoLLM model.
        tokenizer: Tokenizer.
        eval_data: List of AVQA samples.
        video_dir: Directory with video files.
        args: CLI arguments.
        device: Inference device.

    Returns:
        Dictionary with accuracy, correct count, total count, and per-sample results.
    """
    video_transform = VideoTransform(is_train=False)
    audio_transform = AudioTransform(is_train=False)
    shuffled_mapping = build_shuffled_mapping(len(eval_data), args.seed)

    logger.info("Audio ablation mode: %s", args.audio_mode)
    choice_ids = torch.tensor(CHOICE_TOKEN_IDS, device=device)  # (4,)

    correct = 0
    total = 0
    predictions: list[dict[str, str]] = []

    for i, sample in enumerate(eval_data):
        prepared = prepare_sample_data(
            sample, i, args, video_dir, video_transform,
            audio_transform, eval_data, shuffled_mapping,
        )
        if prepared is None:
            continue

        question_text = prepared["question_text"]
        gt_letter = prepared["gt_letter"]
        video_file = prepared["video_file"]


        # Build input_ids with modal tokens
        clean_text = question_text.replace("<video>", "").replace("<audio>", "").strip()
        encoding = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoding["input_ids"]  # (1, S)
        attention_mask = encoding["attention_mask"]  # (1, S)

        # Prepare pixel_values and waveforms
        pixel_values = None
        if prepared["pixel_values"] is not None:
            pixel_values = prepared["pixel_values"].unsqueeze(0).to(device)  # (1, T, C, H, W)

        waveforms = None
        if prepared["waveform"] is not None:
            waveforms = prepared["waveform"].unsqueeze(0).to(device)  # (1, samples)

        # Prepend modal token indices
        modal_ids: list[int] = []
        if pixel_values is not None:
            modal_ids.append(VIDEO_TOKEN_INDEX)
        if waveforms is not None:
            modal_ids.append(AUDIO_TOKEN_INDEX)

        if modal_ids:
            modal_tensor = torch.tensor([modal_ids], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat([modal_tensor, input_ids.to(device)], dim=1)
            modal_mask = torch.ones(1, len(modal_ids), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([modal_mask, attention_mask.to(device)], dim=1)
        else:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        # Single forward pass — no autoregressive generation
        try:
            with torch.no_grad():
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    waveforms=waveforms,
                )

            # For batch_size=1, last position in logits is always the prediction position
            last_logits = outputs.logits[0, -1, :]  # (V,)
            choice_logits = last_logits[choice_ids]  # (4,)
            pred_idx = choice_logits.argmax().item()
            pred_letter = CHOICE_LETTERS[pred_idx]
        except Exception as e:
            logger.warning("Inference failed for sample %d: %s", i, e)
            pred_letter = ""
            choice_logits = torch.zeros(4)

        is_correct = pred_letter == gt_letter
        if is_correct:
            correct += 1
        total += 1

        predictions.append(
            {
                "index": str(i),
                "video": video_file,
                "gt": gt_letter,
                "pred": pred_letter,
                "correct": str(is_correct),
            }
        )

        if total % 10 == 0:
            acc = correct / total * 100
            logger.info("[%d/%d] accuracy: %.1f%% (%d/%d)", total, len(eval_data), acc, correct, total)



    accuracy = correct / total * 100 if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def main() -> None:
    """Run fast AVQA evaluation."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s", args.checkpoint)
    model, tokenizer = load_videollm(args, device)

    logger.info("Loading eval data from %s", args.eval_data)
    eval_data: list[dict[str, object]] = []
    with open(args.eval_data) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                eval_data.append(json.loads(stripped))
    logger.info("Loaded %d evaluation samples", len(eval_data))

    t0 = time.time()
    results = run_fast_evaluation(model, tokenizer, eval_data, Path(args.video_dir), args, device)
    elapsed = time.time() - t0

    results["elapsed_seconds"] = round(elapsed, 1)
    results["checkpoint"] = args.checkpoint
    results["config"] = {
        "use_audio": args.use_audio,
        "use_temporal_bridge": args.use_temporal_bridge,
        "mm_projector_type": args.mm_projector_type,
        "audio_mode": args.audio_mode,
    }

    logger.info(
        "Fast AVQA Eval [%s]: %.1f%% accuracy (%d/%d) in %.1fs (%.2fs/sample)",
        args.audio_mode,
        results["accuracy"],
        results["correct"],
        results["total"],
        elapsed,
        elapsed / max(results["total"], 1),
    )

    output_path = args.output
    if output_path is None:
        ckpt_name = Path(args.checkpoint).name
        output_path = f"eval_results/avqa_{ckpt_name}_{args.audio_mode}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
