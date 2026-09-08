"""Generate DPO preference data by running stage3 model inference on AVQA train split.

For each sample:
- chosen = ground truth answer
- rejected = model's prediction (if wrong) or random wrong choice (if model is correct)

Usage::

    CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 python scripts/generate_dpo_data.py \
        --checkpoint checkpoints/stage3_instruction \
        --train_data data/instruct/avqa_train_split.jsonl \
        --video_dir data/videos \
        --output data/dpo/avqa_preferences.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHOICE_RE = re.compile(r"[A-D]")
CHOICES_PATTERN = re.compile(r"([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)", re.DOTALL)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DPO data generation."""
    parser = argparse.ArgumentParser(description="Generate DPO preference data from AVQA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage3 checkpoint")
    parser.add_argument("--train_data", type=str, default="data/instruct/avqa_train_split.jsonl")
    parser.add_argument("--video_dir", type=str, default="data/videos")
    parser.add_argument("--output", type=str, default="data/dpo/avqa_preferences.jsonl")
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_temporal_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm_projector_type", type=str, default="stc")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_videollm(args: argparse.Namespace, device: torch.device) -> tuple[VideoLLM, AutoTokenizer]:
    """Build VideoLLM and load trained weights.

    Args:
        args: CLI arguments.
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
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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
        return transform(frames)
    except Exception:
        logger.warning("Failed to load video %s, using black frames", video_path)
        return torch.zeros(num_frames, 3, 384, 384)


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


def extract_answer_letter(text: str) -> str:
    """Extract the predicted choice letter (A-D) from model output.

    Args:
        text: Raw model output text.

    Returns:
        Single letter A-D, or empty string if not found.
    """
    match = CHOICE_RE.search(text.upper())
    return match.group(0) if match else ""


def extract_ground_truth(gpt_response: str) -> str:
    """Extract ground truth letter from GPT response like 'The answer is A. ...'

    Args:
        gpt_response: The GPT conversation turn text.

    Returns:
        Ground truth letter A-D.
    """
    match = re.search(r"answer is ([A-D])", gpt_response, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def extract_choices(question_text: str) -> dict[str, str]:
    """Parse choice options from the question text.

    Args:
        question_text: The full question including choices.

    Returns:
        Dictionary mapping letter (A-D) to choice text.
    """
    matches = CHOICES_PATTERN.findall(question_text)
    return {letter: text.strip() for letter, text in matches}


def build_rejected_response(
    gt_letter: str,
    pred_letter: str,
    choices: dict[str, str],
    rng: random.Random,
) -> str:
    """Build a rejected response for DPO.

    If the model predicted wrong, use the model's prediction.
    If the model was correct (or no valid prediction), pick a random wrong choice.

    Args:
        gt_letter: Ground truth letter.
        pred_letter: Model's predicted letter.
        choices: All available choices.
        rng: Random number generator.

    Returns:
        Rejected response string in 'The answer is X. {explanation}' format.
    """
    if pred_letter and pred_letter != gt_letter and pred_letter in choices:
        reject_letter = pred_letter
    else:
        wrong_letters = [letter for letter in choices if letter != gt_letter]
        # Fallback to gt_letter should not happen (implies no wrong choice exists)
        reject_letter = rng.choice(wrong_letters) if wrong_letters else gt_letter

    reject_text = choices.get(reject_letter, reject_letter)
    return f"The answer is {reject_letter}. {reject_text}"


def main() -> None:
    """Generate DPO preference data."""
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s", args.checkpoint)
    model, tokenizer = load_videollm(args, device)

    logger.info("Loading training data from %s", args.train_data)
    train_data: list[dict[str, object]] = []
    with open(args.train_data) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                train_data.append(json.loads(stripped))
    logger.info("Loaded %d training samples", len(train_data))

    video_transform = VideoTransform(is_train=False)
    audio_transform = AudioTransform(is_train=False)
    video_dir = Path(args.video_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preference_pairs: list[dict[str, str]] = []
    correct_count = 0
    total_count = 0
    skipped = 0

    t0 = time.time()

    for i, sample in enumerate(train_data):
        video_file = str(sample.get("video", ""))
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            skipped += 1
            continue

        human_turn = conversations[0]
        gpt_turn = conversations[1]
        if not isinstance(human_turn, dict) or not isinstance(gpt_turn, dict):
            skipped += 1
            continue

        question_text = str(human_turn.get("value", ""))
        gt_text = str(gpt_turn.get("value", ""))
        gt_letter = extract_ground_truth(gt_text)
        if not gt_letter:
            skipped += 1
            continue

        choices = extract_choices(question_text)
        if len(choices) < 2:
            skipped += 1
            continue

        video_path = video_dir / video_file

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

        # Strip modal tokens from text and tokenize
        clean_text = question_text.replace("<video>", "").replace("<audio>", "").strip()
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

        # Generate model prediction
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
            logger.warning("Inference failed for sample %d: %s", i, e)

        is_correct = pred_letter == gt_letter
        if is_correct:
            correct_count += 1
        total_count += 1

        # Build preference pair
        # Prompt includes modal tokens as the PreferenceDataset expects
        prompt = question_text
        if not prompt.startswith("<video>"):
            prompt = "<video>\n<audio>\n" + prompt

        chosen = gt_text  # "The answer is X. explanation"
        rejected = build_rejected_response(gt_letter, pred_letter, choices, rng)

        preference_pairs.append(
            {
                "video": video_file,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

        if (i + 1) % 20 == 0:
            acc = correct_count / total_count * 100 if total_count > 0 else 0.0
            elapsed = time.time() - t0
            logger.info(
                "[%d/%d] model accuracy: %.1f%% | pairs: %d | elapsed: %.0fs",
                i + 1,
                len(train_data),
                acc,
                len(preference_pairs),
                elapsed,
            )

    elapsed = time.time() - t0

    # Write output
    with open(output_path, "w") as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    acc = correct_count / total_count * 100 if total_count > 0 else 0.0
    logger.info("=" * 60)
    logger.info("DPO data generation complete")
    logger.info(
        "Total samples: %d | Skipped: %d | Pairs generated: %d", len(train_data), skipped, len(preference_pairs)
    )
    logger.info("Model accuracy on train split: %.1f%% (%d/%d)", acc, correct_count, total_count)
    logger.info("Output: %s", output_path)
    logger.info("Elapsed: %.1fs", elapsed)


if __name__ == "__main__":
    main()
