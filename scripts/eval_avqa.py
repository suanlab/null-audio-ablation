"""Evaluate VideoLLM on AVQA validation set with audio ablation support.

Loads a trained VideoLLM checkpoint, runs inference on AVQA multi-choice QA,
and reports accuracy. Supports all ablation variants and audio sanity checks.

Audio ablation modes:
    - ``real``: Use actual audio from the video (default).
    - ``shuffled``: Use audio from a random different video in the eval set.
    - ``shifted``: Use audio from the same video, time-shifted by N seconds.
    - ``noise``: Replace audio with random Gaussian noise.
    - ``silent``: Replace audio with zeros (silence).

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_avqa.py \
        --checkpoint checkpoints/stage3_instruction \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --video_dir data/videos \
        --use_audio --use_temporal_bridge --mm_projector_type stc \
        --audio_mode real

    # Shuffled audio ablation
    python scripts/eval_avqa.py \
        --checkpoint checkpoints/stage3_instruction \
        --audio_mode shuffled

    # Time-shifted audio ablation (5 second shift)
    python scripts/eval_avqa.py \
        --checkpoint checkpoints/stage3_instruction \
        --audio_mode shifted --shift_seconds 5.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import signal
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
from videollm.data.video_decode import GuardedDecoder
from videollm.data.video_degradation import VISUAL_MODES, DegradationConfig, apply_visual_degradation
from videollm.model.videollm import ModelConfig, VideoLLM
from videollm.utils import load_audio_from_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHOICE_RE = re.compile(r"[A-D]")


AUDIO_MODES = ("real", "shuffled", "shifted", "noise", "silent")


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments.

    Returns:
        Parsed CLI namespace with model config, eval paths, and audio ablation settings.
    """
    parser = argparse.ArgumentParser(description="Evaluate VideoLLM on AVQA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--eval_data", type=str, default="data/instruct/avqa_val.jsonl")
    parser.add_argument("--video_dir", type=str, default="data/videos")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_temporal_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm_projector_type", type=str, default="stc", choices=["stc", "mlp", "linear"])
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    # Audio ablation arguments
    parser.add_argument(
        "--audio_mode",
        type=str,
        default="real",
        choices=list(AUDIO_MODES),
        help="Audio ablation mode: real (default), shuffled, shifted, noise, silent",
    )
    parser.add_argument(
        "--shift_seconds",
        type=float,
        default=3.0,
        help="Seconds to shift audio for 'shifted' mode (default: 3.0)",
    )
    # Visual degradation arguments (CMSS visual axis, symmetric to --audio_mode)
    parser.add_argument(
        "--visual_mode",
        type=str,
        default="clean",
        choices=list(VISUAL_MODES),
        help="Visual degradation family: clean (default), motion_blur, occlusion, frame_drop, downscale",
    )
    parser.add_argument(
        "--visual_severity",
        type=float,
        default=1.0,
        help="Visual degradation severity in [0, 1] (ignored when --visual_mode clean)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffled audio mapping")
    parser.add_argument("--skip_to", type=int, default=0, help="Skip to this sample index (0-based, for resuming)")
    parser.add_argument(
        "--decode_timeout",
        type=float,
        default=0.0,
        help="Hard per-video decode budget in seconds, enforced in a killable child "
             "process (0 = decode in-process, legacy behaviour). Guards against decord "
             "hangs that SIGALRM cannot preempt.",
    )
    parser.add_argument("--sample_timeout", type=int, default=120, help="Per-sample timeout in seconds (default: 120)")
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


def load_video_frames(
    video_path: str,
    num_frames: int,
    transform: VideoTransform,
    degradation: DegradationConfig | None = None,
) -> torch.Tensor:
    """Load and preprocess video frames.

    Args:
        video_path: Path to MP4 file.
        num_frames: Number of frames to sample.
        transform: Video transform pipeline.
        degradation: Optional visual-degradation intervention applied to the raw
            ``[0, 1]`` frames *before* ``transform`` (symmetric to the audio-mode
            injection point). ``None`` or ``clean`` mode is a no-op.

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
        if degradation is not None:
            frames = apply_visual_degradation(frames, degradation)  # (T, C, H, W)
        return transform(frames)
    except Exception:
        logger.warning("Failed to load video %s, using black frames", video_path)
        return transform(torch.zeros(num_frames, 3, 384, 384))


_DECODER: GuardedDecoder | None = None


def _get_decoder(timeout_s: float) -> GuardedDecoder:
    """Return the process-wide decode worker, starting it on first use."""
    global _DECODER  # noqa: PLW0603 - one worker per eval process, reused across samples
    if _DECODER is None:
        _DECODER = GuardedDecoder(timeout_s)
    return _DECODER


def load_video_frames_guarded(
    video_path: str,
    num_frames: int,
    transform: VideoTransform,
    degradation: DegradationConfig | None,
    timeout_s: float,
) -> torch.Tensor | None:
    """Load frames with an enforceable timeout, or ``None`` if decoding times out/fails.

    decord can hang indefinitely on some clips and SIGALRM cannot preempt native code,
    so with ``timeout_s > 0`` the decode runs in a reusable spawned worker that is
    killed and replaced when it hangs (see :mod:`videollm.data.video_decode`).
    ``timeout_s <= 0`` keeps the original in-process path.

    Returning ``None`` lets the caller record a failed inference rather than dropping
    the item, so the evaluated item set stays matched across models.
    """
    if timeout_s <= 0:
        return load_video_frames(video_path, num_frames, transform, degradation)
    return _get_decoder(timeout_s).decode(video_path, num_frames, transform.image_size, degradation)


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
    seed: int = 42,
) -> torch.Tensor:
    """Load audio waveform according to the specified ablation mode.

    Args:
        audio_mode: One of ``real``, ``shuffled``, ``shifted``, ``noise``, ``silent``.
        video_path: Path to the current sample's video file.
        audio_transform: Audio transform pipeline.
        eval_data: Full list of evaluation samples (for shuffled mode).
        video_dir: Root directory containing video files.
        sample_idx: Current sample index in eval_data.
        shuffled_mapping: Pre-computed shuffled index mapping (sample_idx -> donor_idx).
        shift_seconds: Number of seconds to shift audio (for shifted mode).
        seed: Base seed; ``noise`` is drawn from a per-item generator seeded with
            ``seed + sample_idx`` so a given item gets the same waveform regardless of
            iteration order or ``--skip_to`` resumption (replay-determinism gate).

    Returns:
        Waveform tensor of shape ``(samples,)``.
    """
    duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)  # 480000

    if audio_mode == "real":
        return load_audio_waveform(str(video_path), audio_transform)

    if audio_mode == "silent":
        return torch.zeros(duration_samples)  # (samples,)

    if audio_mode == "noise":
        generator = torch.Generator(device="cpu").manual_seed(seed + sample_idx)
        return torch.randn(duration_samples, generator=generator)  # (samples,)

    if audio_mode == "shifted":
        waveform = load_audio_waveform(str(video_path), audio_transform)  # (samples,)
        shift_samples = int(shift_seconds * DEFAULT_AUDIO_SAMPLE_RATE)
        return torch.roll(waveform, shifts=shift_samples, dims=0)  # (samples,)

    if audio_mode == "shuffled":
        donor_idx = shuffled_mapping[sample_idx]
        donor_sample = eval_data[donor_idx]
        donor_video = str(donor_sample.get("video", ""))
        donor_path = video_dir / donor_video
        if donor_path.exists():
            return load_audio_waveform(str(donor_path), audio_transform)
        return torch.zeros(duration_samples)  # (samples,)

    raise ValueError(f"Unknown audio_mode: {audio_mode}")


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


def build_shuffled_mapping(num_samples: int, seed: int) -> list[int]:
    """Build a deterministic rejection-sampled derangement (no fixed points).

    Matches the contract used by the external adapters (VideoLLaMA2,
    Qwen2.5-Omni, video-SALMONN 2+) so that cross-model shuffled results are
    on the IDENTICAL donor mapping for the same (num_samples, seed). CHECK.md
    B.1 / L.1: this replaces an earlier cyclic-offset variant whose mapping
    differed from the externals'.

    Args:
        num_samples: Total number of evaluation samples.
        seed: Random seed for reproducibility.

    Returns:
        List where ``mapping[i]`` is the donor index for sample ``i``.
    """
    rng = random.Random(seed)
    idx = list(range(num_samples))
    for _ in range(100):
        perm = idx[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(num_samples)):
            return perm
    return [(i + 1) % num_samples for i in range(num_samples)]


def _run_single_sample(
    sample_idx: int,
    sample: dict[str, object],
    video_dir: Path,
    video_file: str,
    question_text: str,
    video_transform: VideoTransform,
    audio_transform: AudioTransform,
    audio_mode: str,
    eval_data: list[dict[str, object]],
    shuffled_mapping: list[int],
    shift_seconds: float,
    model: VideoLLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, str]:
    """Run inference on a single sample.

    Args:
        sample_idx: Index in eval_data.
        sample: The evaluation sample dict.
        video_dir: Directory containing video files.
        video_file: Video filename.
        question_text: Clean question text.
        video_transform: Video transform pipeline.
        audio_transform: Audio transform pipeline.
        audio_mode: Audio ablation mode.
        eval_data: Full eval dataset (for shuffled mode).
        shuffled_mapping: Pre-computed shuffled index mapping.
        shift_seconds: Time shift for shifted mode.
        model: VideoLLM model.
        tokenizer: Tokenizer.
        args: CLI arguments.
        device: Inference device.

    Returns:
        Tuple of (pred_letter, decoded_text).
    """
    video_path = video_dir / video_file

    pixel_values: torch.Tensor | None = None
    if video_path.exists():
        degradation = DegradationConfig(
            mode=args.visual_mode,
            severity=args.visual_severity,
            seed=args.seed + sample_idx,  # per-video variation, still deterministic
        )
        frames = load_video_frames_guarded(
            str(video_path),
            args.num_frames,
            video_transform,
            degradation,
            getattr(args, "decode_timeout", 0.0),
        )
        if frames is None:
            # Do NOT fall through to inference: with pixel_values unset the model would
            # answer from audio alone, silently turning this item into the audio-only
            # condition the protocol is measuring and inflating apparent audio reliance.
            # Report it as a failed inference instead, matching the external adapters.
            logger.warning("Undecodable video, recording sample %d as failed: %s", sample_idx, video_file)
            return "", ""
        pixel_values = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)
        # Positive control: blank the video stream (env-gated, no behavior change otherwise).
        import os as _os

        if _os.environ.get("POS_BLANK") == "1":
            pixel_values = torch.zeros_like(pixel_values)

    waveforms: torch.Tensor | None = None
    if args.use_audio and video_path.exists():
        wf = get_ablated_audio(
            audio_mode=audio_mode,
            video_path=video_path,
            audio_transform=audio_transform,
            eval_data=eval_data,
            video_dir=video_dir,
            sample_idx=sample_idx,
            shuffled_mapping=shuffled_mapping,
            shift_seconds=shift_seconds,
            seed=args.seed,
        )
        waveforms = wf.unsqueeze(0).to(device)  # (1, samples)

    clean_text = question_text.replace("<video>", "").replace("<audio>", "").strip()
    encoding = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoding["input_ids"].to(device)  # (1, S)
    attention_mask = encoding["attention_mask"].to(device)  # (1, S)

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
        logger.warning("Inference failed for sample %d: %s", sample_idx, e)
        decoded = ""
        pred_letter = ""

    return pred_letter, decoded


def run_evaluation(
    model: VideoLLM,
    tokenizer: AutoTokenizer,
    eval_data: list[dict[str, object]],
    video_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Run AVQA evaluation loop with optional audio ablation.

    Args:
        model: Trained VideoLLM model.
        tokenizer: Tokenizer.
        eval_data: List of AVQA samples.
        video_dir: Directory with video files.
        args: CLI arguments including ``audio_mode`` and ``shift_seconds``.
        device: Inference device.

    Returns:
        Dictionary with accuracy, correct count, total count, and per-sample results.
    """
    video_transform = VideoTransform(is_train=False)
    audio_transform = AudioTransform(is_train=False)

    audio_mode: str = getattr(args, "audio_mode", "real")
    shift_seconds: float = getattr(args, "shift_seconds", 3.0)
    seed: int = getattr(args, "seed", 42)

    # Pre-compute shuffled mapping for deterministic audio donor assignment
    shuffled_mapping = build_shuffled_mapping(len(eval_data), seed)

    logger.info("Audio ablation mode: %s", audio_mode)
    if audio_mode == "shifted":
        logger.info("Audio shift: %.1f seconds", shift_seconds)

    skip_to: int = getattr(args, "skip_to", 0)
    sample_timeout: int = getattr(args, "sample_timeout", 120)
    if skip_to > 0:
        logger.info("Skipping to sample %d", skip_to)

    correct = 0
    total = 0
    predictions: list[dict[str, str]] = []

    import os as _os

    _pos_idx_path = _os.environ.get("POS_IDX")
    _pos_idx: set[int] | None = None
    if _pos_idx_path:
        with open(_pos_idx_path) as _pf:
            _pos_idx = set(json.loads(_pf.read()))

    for i, sample in enumerate(eval_data):
        if i < skip_to:
            continue
        if _pos_idx is not None and i not in _pos_idx:
            continue
        video_file = str(sample.get("video", ""))
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue

        human_turn = conversations[0]
        gpt_turn = conversations[1]
        if not isinstance(human_turn, dict) or not isinstance(gpt_turn, dict):
            continue

        question_text = str(human_turn.get("value", ""))
        gt_text = str(gpt_turn.get("value", ""))
        gt_letter = extract_ground_truth(gt_text)
        if not gt_letter:
            continue

        # Per-sample timeout to prevent hangs on bad video files
        def _timeout_handler(signum, frame, _i=i):  # noqa: ANN001,ARG001  bind loop var
            raise TimeoutError(f"Sample {_i} timed out after {sample_timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(sample_timeout)
        try:
            pred_letter, decoded = _run_single_sample(
                i,
                sample,
                video_dir,
                video_file,
                question_text,
                video_transform,
                audio_transform,
                audio_mode,
                eval_data,
                shuffled_mapping,
                shift_seconds,
                model,
                tokenizer,
                args,
                device,
            )
        except TimeoutError as e:
            logger.warning(str(e))
            pred_letter, decoded = "", ""
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

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
                "output": decoded[:200],
            }
        )

        if (i + 1) % 10 == 0:
            acc = correct / total * 100 if total > 0 else 0.0
            logger.info("[%d/%d] accuracy: %.1f%% (%d/%d)", i + 1, len(eval_data), acc, correct, total)

    accuracy = correct / total * 100 if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def main() -> None:
    """Run AVQA evaluation."""
    args = parse_args()
    # Deterministic seeding (CHECK.md B.6): make noise mode and any RNG use reproducible.
    # (eval_avqa.py noise uses torch.randn; no numpy needed here.)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    results = run_evaluation(model, tokenizer, eval_data, Path(args.video_dir), args, device)
    elapsed = time.time() - t0

    results["elapsed_seconds"] = round(elapsed, 1)
    results["checkpoint"] = args.checkpoint
    results["config"] = {
        "use_audio": args.use_audio,
        "use_temporal_bridge": args.use_temporal_bridge,
        "mm_projector_type": args.mm_projector_type,
        "audio_mode": args.audio_mode,
        "shift_seconds": args.shift_seconds if args.audio_mode == "shifted" else None,
        "visual_mode": args.visual_mode,
        "visual_severity": args.visual_severity if args.visual_mode != "clean" else 0.0,
    }

    logger.info(
        "AVQA Evaluation [%s]: %.1f%% accuracy (%d/%d) in %.1fs",
        args.audio_mode,
        results["accuracy"],
        results["correct"],
        results["total"],
        elapsed,
    )

    output_path = args.output
    if output_path is None:
        ckpt_name = Path(args.checkpoint).name
        vis_tag = "" if args.visual_mode == "clean" else f"_vis-{args.visual_mode}-s{args.visual_severity:g}"
        output_path = f"eval_results/avqa_{ckpt_name}_{args.audio_mode}{vis_tag}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    if _DECODER is not None:
        _DECODER.close()


if __name__ == "__main__":
    main()
