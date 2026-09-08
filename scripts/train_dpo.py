"""Train VideoLLM with DPO (Direct Preference Optimization).

Loads a stage3 checkpoint as both policy and reference model, then
trains on preference pairs using the DPO objective.

Usage::

    CUDA_HOME=/tmp/fake_cuda CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo.py \
        --checkpoint checkpoints/stage3_instruction \
        --data_path data/dpo/avqa_preferences.jsonl \
        --video_dir data/videos \
        --output_dir checkpoints/stage4_dpo \
        --num_epochs 1 --batch_size 2 --learning_rate 5e-6
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model
from transformers import AutoTokenizer, TrainingArguments

from videollm.data.constants import DEFAULT_NUM_FRAMES
from videollm.data.preference_dataset import PreferenceDataset, preference_collate_fn
from videollm.data.transforms import AudioTransform, VideoTransform
from videollm.dpo_trainer import DPOConfig, DPOTrainer
from videollm.model.videollm import ModelConfig, VideoLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse DPO training arguments."""
    parser = argparse.ArgumentParser(description="Train VideoLLM with DPO")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 3 checkpoint path")
    parser.add_argument("--data_path", type=str, default="data/dpo/avqa_preferences.jsonl")
    parser.add_argument("--video_dir", type=str, default="data/videos")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage4_dpo")
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_temporal_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm_projector_type", type=str, default="stc")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_loss_type", type=str, default="sigmoid", choices=["sigmoid", "hinge"])
    parser.add_argument("--reference_free", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config JSON path")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[VideoLLM, AutoTokenizer]:
    """Build and load VideoLLM from stage3 checkpoint.

    Args:
        args: Training arguments.
        device: Target device (used for dtype selection only; Trainer handles placement).

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
        logger.info("Loaded stage3 checkpoint from %s", ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.llm_path, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer


def main() -> None:
    """Run DPO training."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Stage 4: DPO Training")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Data: %s", args.data_path)
    logger.info("Output: %s", args.output_dir)
    logger.info("DPO beta: %s, loss_type: %s", args.dpo_beta, args.dpo_loss_type)
    logger.info(
        "Batch size: %d, Grad accum: %d, LR: %s", args.batch_size, args.gradient_accumulation_steps, args.learning_rate
    )
    logger.info("Reference free: %s", args.reference_free)
    logger.info("=" * 60)

    # Build model
    logger.info("Building policy model...")
    model, tokenizer = build_model(args, device)

    # Build dataset
    logger.info("Loading preference dataset...")
    video_transform = VideoTransform(is_train=True)
    audio_transform = AudioTransform(is_train=True)

    train_dataset = PreferenceDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        video_transform=video_transform,
        audio_transform=audio_transform,
        num_frames=args.num_frames,
        max_length=args.max_length,
        video_dir=args.video_dir,
    )
    logger.info("Dataset loaded: %d preference pairs", len(train_dataset))

    # DPO config
    dpo_config = DPOConfig(
        beta=args.dpo_beta,
        loss_type=args.dpo_loss_type,
        reference_free=args.reference_free,
    )

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    # Build DPO trainer
    logger.info("Initializing DPO trainer (reference_free=%s)...", args.reference_free)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Deep copy created internally if not reference_free
        dpo_config=dpo_config,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=preference_collate_fn,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting DPO training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info("DPO training completed in %.1fs", elapsed)

    # Save final model
    logger.info("Saving model to %s", output_dir)
    save_model(model, str(output_dir / "model.safetensors"))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("=" * 60)
    logger.info("Stage 4 DPO complete. Checkpoint: %s", output_dir)
    logger.info("Elapsed: %.1fs", elapsed)


if __name__ == "__main__":
    main()
