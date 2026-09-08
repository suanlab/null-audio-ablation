"""Dry-run test for train.py pipeline on GPU.

Verifies that HuggingFace Trainer integration works for both SFT and DPO
modes with dummy data. Runs 2 training steps with batch_size=1.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/test_train_dryrun.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torchvision
from transformers import AutoTokenizer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def create_tiny_videos(video_dir: Path, n: int = 4) -> list[str]:
    """Create tiny dummy MP4 files for testing."""
    video_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i in range(n):
        name = f"tiny_{i:04d}.mp4"
        frames = torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.uint8)
        t = torch.linspace(0, 1.0, 48000)
        audio = (torch.sin(2 * torch.pi * 440 * t) * 0.3).unsqueeze(0)
        torchvision.io.write_video(
            str(video_dir / name), frames, fps=8, audio_array=audio, audio_fps=48000, audio_codec="aac"
        )
        names.append(name)
    return names


def create_sft_jsonl(path: Path, video_names: list[str]) -> None:
    """Write SFT annotations."""
    with open(path, "w") as f:
        for name in video_names:
            ann = {
                "video": name,
                "conversations": [
                    {"from": "human", "value": "<video>\n<audio>\nDescribe this video."},
                    {"from": "gpt", "value": "A short video with colorful content and audio."},
                ],
            }
            f.write(json.dumps(ann) + "\n")


def create_dpo_jsonl(path: Path, video_names: list[str]) -> None:
    """Write DPO annotations."""
    with open(path, "w") as f:
        for name in video_names:
            ann = {
                "video": name,
                "prompt": "<video>\n<audio>\nDescribe this video.",
                "chosen": "A detailed video description with rich audio-visual content.",
                "rejected": "A video.",
            }
            f.write(json.dumps(ann) + "\n")


def test_sft_training(tmp_dir: Path, video_names: list[str]) -> bool:
    """Test SFT training with HuggingFace Trainer."""
    from videollm.data.dataset import VideoAudioDataset, collate_fn
    from videollm.data.transforms import AudioTransform, VideoTransform
    from videollm.model.videollm import ModelConfig, VideoLLM

    logger.info("=" * 60)
    logger.info("TEST: SFT Training Dry-Run")
    logger.info("=" * 60)

    sft_path = tmp_dir / "sft.jsonl"
    create_sft_jsonl(sft_path, video_names)
    output_dir = tmp_dir / "sft_out"

    t0 = time.time()

    model_config = ModelConfig(num_frames=8, use_lora=True, lora_r=8, lora_alpha=16)
    model = VideoLLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = VideoAudioDataset(
        data_path=str(sft_path),
        tokenizer=tokenizer,
        video_transform=VideoTransform(is_train=True),
        audio_transform=AudioTransform(is_train=True),
        num_frames=8,
        max_length=256,
        video_dir=str(tmp_dir / "videos"),
    )

    from transformers import Trainer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=2,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    elapsed = time.time() - t0

    logger.info("SFT training loss: %.4f", train_result.training_loss)
    logger.info("SFT dry-run completed in %.1fs", elapsed)
    logger.info("VRAM used: %.1f GB", torch.cuda.max_memory_allocated() / 1e9)

    del model, trainer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return True


def test_dpo_training(tmp_dir: Path, video_names: list[str]) -> bool:
    """Test DPO training with custom DPOTrainer."""
    from videollm.data.preference_dataset import PreferenceDataset, preference_collate_fn
    from videollm.data.transforms import AudioTransform, VideoTransform
    from videollm.dpo_trainer import DPOConfig, DPOTrainer
    from videollm.model.videollm import ModelConfig, VideoLLM

    logger.info("=" * 60)
    logger.info("TEST: DPO Training Dry-Run")
    logger.info("=" * 60)

    dpo_path = tmp_dir / "dpo.jsonl"
    create_dpo_jsonl(dpo_path, video_names)
    output_dir = tmp_dir / "dpo_out"

    t0 = time.time()

    model_config = ModelConfig(num_frames=8, use_lora=True, lora_r=8, lora_alpha=16)
    model = VideoLLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PreferenceDataset(
        data_path=str(dpo_path),
        tokenizer=tokenizer,
        video_transform=VideoTransform(is_train=True),
        audio_transform=AudioTransform(is_train=True),
        num_frames=8,
        max_length=256,
        video_dir=str(tmp_dir / "videos"),
    )

    dpo_config = DPOConfig(beta=0.1, reference_free=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=2,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        dpo_config=dpo_config,
        args=training_args,
        train_dataset=dataset,
        data_collator=preference_collate_fn,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    elapsed = time.time() - t0

    logger.info("DPO training loss: %.4f", train_result.training_loss)
    logger.info("DPO dry-run completed in %.1fs", elapsed)
    logger.info("VRAM used: %.1f GB", torch.cuda.max_memory_allocated() / 1e9)

    del model, trainer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return True


def main() -> None:
    """Run SFT and DPO dry-run tests."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot run dry-run.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    logger.info("GPU: %s", device_name)
    logger.info("VRAM: %.1f GB free", torch.cuda.mem_get_info(0)[0] / 1e9)

    tmp_dir = Path(tempfile.mkdtemp(prefix="videollm_dryrun_"))
    logger.info("Temp dir: %s", tmp_dir)

    try:
        video_names = create_tiny_videos(tmp_dir / "videos", n=4)
        logger.info("Created %d tiny videos", len(video_names))

        sft_ok = test_sft_training(tmp_dir, video_names)
        dpo_ok = test_dpo_training(tmp_dir, video_names)

        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("  SFT: %s", "PASS" if sft_ok else "FAIL")
        logger.info("  DPO: %s", "PASS" if dpo_ok else "FAIL")
        logger.info("=" * 60)

        if not (sft_ok and dpo_ok):
            sys.exit(1)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp dir")


if __name__ == "__main__":
    main()
