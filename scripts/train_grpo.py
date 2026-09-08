"""GRPO reinforcement learning training for AGTA (Phase 3).

Trains the AGTA model with Group Relative Policy Optimization to improve
audio utilization quality. Generates rollouts with mixed audio conditions
(real + silent) and rewards correct answers under both conditions.

Design inspired by Video-R1's temporal augmentation approach:
  - Half of rollouts use real audio
  - Half of rollouts use silent/noise audio
  - GRPO naturally learns: use audio when it helps, be robust when it doesn't

Prerequisites:
  - Completed Phase 2 (Reward-Weighted SFT) checkpoint
  - trl >= 0.29.0 installed

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
        --model_path checkpoints/audio_quick_bridge_v4/stage3_instruction \
        --data_path data/instruct/avqa_train_quick5k.jsonl \
        --output_dir checkpoints/audio_quick_bridge_v5_grpo

Note:
    This script requires significant engineering for multimodal GRPO.
    The standard trl GRPOTrainer handles text-only generation.
    For multimodal models (video+audio), we need a custom rollout/generation
    pipeline that feeds pixel_values and waveforms during RL rollouts.

    Current status: SCAFFOLD — needs custom GRPOTrainer subclass for
    multimodal generation. See TODO markers below.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse GRPO training CLI arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="GRPO RL training for AGTA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Training data JSONL path")
    parser.add_argument("--video_dir", type=str, default="data/videos", help="Video directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of RL epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--num_generations", type=int, default=8, help="Rollouts per prompt")
    parser.add_argument("--max_completion_length", type=int, default=64, help="Max generated tokens")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty (0=no ref model)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--audio_dropout_ratio", type=float, default=0.5, help="Fraction of rollouts with silent audio")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load LLM in 4-bit (QLoRA)")
    return parser.parse_args()


def prepare_grpo_dataset(data_path: str, video_dir: str) -> list[dict[str, object]]:
    """Load and format training data for GRPO.

    The GRPO dataset needs a ``prompt`` column (the question) and additional columns
    for reward computation (``ground_truth``, ``video``, etc.).

    Args:
        data_path: Path to AVQA training JSONL.
        video_dir: Base directory for video files.

    Returns:
        List of formatted samples for GRPO training.
    """
    samples: list[dict[str, object]] = []
    with open(data_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            conversations = item.get("conversations", [])
            if len(conversations) < 2:
                continue

            human_turn = conversations[0]
            gpt_turn = conversations[1]
            question = human_turn.get("value", "")
            ground_truth = gpt_turn.get("value", "")
            video = item.get("video", "")

            # Strip modal tokens for the text prompt
            clean_prompt = question.replace("<video>", "").replace("<audio>", "").strip()

            samples.append(
                {
                    "prompt": clean_prompt,
                    "ground_truth": ground_truth,
                    "video": video,
                    "video_path": str(Path(video_dir) / video),
                    "has_audio_token": "<audio>" in question,
                    "has_video_token": "<video>" in question,
                }
            )

    logger.info("Prepared %d GRPO training samples from %s", len(samples), data_path)
    return samples


def main() -> None:
    """Run GRPO training for AGTA."""
    args = parse_args()

    # Step 1: Prepare dataset
    samples = prepare_grpo_dataset(args.data_path, args.video_dir)

    # Step 2: Import components
    from safetensors.torch import load_model as _load_safetensors
    from transformers import AutoTokenizer

    from videollm.grpo_reward import combined_audio_reward
    from videollm.grpo_trainer import AudioVisualGRPOTrainer, GRPOTrainingConfig
    from videollm.model.videollm import ModelConfig, VideoLLM

    # Step 3: Build model
    logger.info("Loading model from %s", args.model_path)
    model_config = ModelConfig(
        mm_projector_type="stc",
        use_audio=True,
        use_temporal_bridge=True,
        use_lora=args.use_lora,
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=not args.use_lora,
        load_in_4bit=args.load_in_4bit,
    )
    model = VideoLLM(model_config)

    ckpt_path = Path(args.model_path) / "model.safetensors"
    if ckpt_path.exists():
        if args.load_in_4bit:
            # 4-bit: base LLM weights are quantized, can't load bf16 checkpoint for them.
            # Load only LoRA adapters + projectors + bridge (skip base LLM weights).
            from safetensors.torch import load_file
            state_dict = load_file(str(ckpt_path))
            filtered = {
                k: v for k, v in state_dict.items()
                if not ("llm" in k and "base_layer" in k)
                and not ("llm" in k and "lora" not in k and "base_model" not in k)
            }
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            logger.info(
                "4-bit checkpoint load: %d params loaded, %d skipped (base LLM), %d unexpected",
                len(filtered), len(missing), len(unexpected),
            )
        else:
            _load_safetensors(model, str(ckpt_path), strict=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_in_4bit:
        # 4-bit: LLM stays on GPU from quantization, move non-LLM modules
        for name, module in model.named_children():
            if name != "llm":
                module.to(device=device, dtype=torch.bfloat16)
    elif args.bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 4: Configure GRPO
    config = GRPOTrainingConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        audio_dropout_ratio=args.audio_dropout_ratio,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=0.95,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        beta=args.beta,
        epsilon=args.epsilon,
        bf16=args.bf16,
        video_dir=args.video_dir,
    )

    # Step 5: Train
    trainer = AudioVisualGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=samples,
        reward_fn=combined_audio_reward,
        config=config,
    )

    logger.info("="  * 60)
    logger.info("GRPO Training Configuration:")
    logger.info("  Model: %s", args.model_path)
    logger.info("  Data: %s (%d samples)", args.data_path, len(samples))
    logger.info("  Generations per prompt: %d", args.num_generations)
    logger.info("  Audio dropout ratio: %.1f", args.audio_dropout_ratio)
    logger.info("  Learning rate: %.1e", args.learning_rate)
    logger.info("  KL beta: %.2f", args.beta)
    logger.info("  Output: %s", args.output_dir)
    logger.info("=" * 60)

    result = trainer.train()
    logger.info("GRPO training complete: %s", result)


if __name__ == "__main__":
    main()
