"""Evaluate VideoLLM on MVBench (20-task multi-choice video QA).

Loads a trained VideoLLM checkpoint, runs inference on all 20 MVBench tasks,
and reports per-task and overall accuracy.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_mvbench.py \
        --checkpoint checkpoints/stage3_instruction \
        --mvbench_dir data/MVBench \
        --use_audio --use_temporal_bridge --mm_projector_type stc
"""

from __future__ import annotations

import argparse
import json
import logging
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

TASK_VIDEO_DIRS: dict[str, str] = {
    "action_antonym": "ssv2_video",
    "action_count": "star/Charades_v1_480",
    "action_localization": "sta/sta_video",
    "action_prediction": "star/Charades_v1_480",
    "action_sequence": "star/Charades_v1_480",
    "character_order": "perception/videos",
    "counterfactual_inference": "clevrer/video_validation",
    "egocentric_navigation": "vlnqa",
    "episodic_reasoning": "tvqa/frames_fps3_hq",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "fine_grained_pose": "nturgbd",
    "moving_attribute": "clevrer/video_validation",
    "moving_count": "clevrer/video_validation",
    "moving_direction": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "object_interaction": "star/Charades_v1_480",
    "object_shuffle": "perception/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "unexpected_action": "FunQA_test/test",
}


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VideoLLM on MVBench")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mvbench_dir", type=str, default="data/MVBench")
    parser.add_argument("--output", type=str, default="eval_results/mvbench_results.json")
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_temporal_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm_projector_type", type=str, default="stc")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def load_videollm(args: argparse.Namespace, device: torch.device) -> tuple[VideoLLM, AutoTokenizer]:
    """Build VideoLLM and load trained weights."""
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
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_video_frames(video_path: str, num_frames: int, transform: VideoTransform) -> torch.Tensor | None:
    """Load and preprocess video frames, returns None on failure."""
    import decord

    decord.bridge.set_bridge("torch")
    try:
        vr = decord.VideoReader(video_path, num_threads=1)
        indices = uniform_frame_sample(len(vr), num_frames)
        frames = vr.get_batch(indices)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
        return transform(frames)
    except Exception:
        return None


def load_audio_waveform(video_path: str, transform: AudioTransform) -> torch.Tensor:
    """Extract audio waveform from video."""
    try:
        waveform, sr = load_audio_from_video(video_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        return transform(waveform, sr)
    except (RuntimeError, OSError, ImportError):
        duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
        return torch.zeros(duration_samples)


def build_mvbench_prompt(question: str, candidates: list[str]) -> str:
    """Build a multi-choice prompt for MVBench."""
    letters = "ABCDEFGHIJ"
    choices = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(candidates))
    return f"<video>\nQuestion: {question}\nChoices:\n{choices}\nAnswer with the letter."


def extract_answer(text: str, num_choices: int) -> str:
    """Extract predicted choice letter from model output."""
    valid = set("ABCDEFGHIJ"[:num_choices])
    for ch in text.upper():
        if ch in valid:
            return ch
    return ""


def evaluate_task(
    task_name: str,
    samples: list[dict[str, object]],
    model: VideoLLM,
    tokenizer: AutoTokenizer,
    video_base: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a single MVBench task."""
    video_transform = VideoTransform(is_train=False)
    audio_transform = AudioTransform(is_train=False)
    video_subdir = TASK_VIDEO_DIRS.get(task_name, "")
    video_dir = video_base / "video" / video_subdir

    correct = 0
    total = 0
    predictions: list[dict[str, str]] = []

    for sample in samples:
        video_file = str(sample.get("video", ""))
        question = str(sample.get("question", ""))
        candidates = sample.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        answer = str(sample.get("answer", ""))

        gt_idx = -1
        for i, c in enumerate(candidates):
            if str(c).strip() == answer.strip():
                gt_idx = i
                break
        if gt_idx < 0:
            continue
        gt_letter = "ABCDEFGHIJ"[gt_idx]

        video_path = video_dir / video_file
        if not video_path.exists():
            video_path = video_base / "video" / video_file
        if not video_path.exists():
            for subdir in video_base.glob("video/*"):
                candidate = subdir / video_file
                if candidate.exists():
                    video_path = candidate
                    break

        pixel_values: torch.Tensor | None = None
        if video_path.exists():
            frames = load_video_frames(str(video_path), args.num_frames, video_transform)
            if frames is not None:
                pixel_values = frames.unsqueeze(0).to(device)

        waveforms: torch.Tensor | None = None
        if args.use_audio and video_path.exists():
            wf = load_audio_waveform(str(video_path), audio_transform)
            waveforms = wf.unsqueeze(0).to(device)

        prompt = build_mvbench_prompt(question, [str(c) for c in candidates])
        clean_text = prompt.replace("<video>", "").replace("<audio>", "").strip()
        encoding = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

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
            pred_letter = extract_answer(decoded, len(candidates))
        except Exception as e:
            logger.warning("Inference failed for %s/%s: %s", task_name, video_file, e)

        is_correct = pred_letter == gt_letter
        if is_correct:
            correct += 1
        total += 1

        predictions.append(
            {
                "video": video_file,
                "gt": gt_letter,
                "pred": pred_letter,
                "correct": str(is_correct),
            }
        )

    accuracy = correct / total * 100 if total > 0 else 0.0
    return {
        "task": task_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def main() -> None:
    """Run MVBench evaluation."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mvbench_dir = Path(args.mvbench_dir)

    logger.info("Loading model from %s", args.checkpoint)
    model, tokenizer = load_videollm(args, device)

    json_dir = mvbench_dir / "json"
    task_files = sorted(json_dir.glob("*.json"))
    logger.info("Found %d MVBench tasks", len(task_files))

    all_results: dict[str, object] = {}
    total_correct = 0
    total_samples = 0

    t0 = time.time()
    for task_file in task_files:
        task_name = task_file.stem
        with open(task_file) as f:
            samples = json.load(f)
        logger.info("Evaluating %s (%d samples)...", task_name, len(samples))

        result = evaluate_task(task_name, samples, model, tokenizer, mvbench_dir, args, device)
        all_results[task_name] = result
        total_correct += result["correct"]
        total_samples += result["total"]

        logger.info("  %s: %.1f%% (%d/%d)", task_name, result["accuracy"], result["correct"], result["total"])

    elapsed = time.time() - t0
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0.0

    output = {
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 1),
        "checkpoint": args.checkpoint,
        "per_task": all_results,
    }

    logger.info("=" * 60)
    logger.info("MVBench Overall: %.1f%% (%d/%d) in %.1fs", overall_acc, total_correct, total_samples, elapsed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
