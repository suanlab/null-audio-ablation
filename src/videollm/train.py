"""Training entry point for VideoLLM."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

from videollm.data.constants import DEFAULT_AUDIO_ENCODER, DEFAULT_LLM, DEFAULT_NUM_FRAMES, DEFAULT_VISION_ENCODER
from videollm.data.dataset import VideoAudioDataset
from videollm.model.videollm import ModelConfig, VideoLLM
from videollm.utils import is_main_process, setup_logging

logger = logging.getLogger(__name__)
torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for VideoLLM training.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(description="Train VideoLLM with single GPU, DDP, or DeepSpeed")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model_path", type=str, default=DEFAULT_LLM, help="LLM checkpoint path")
    parser.add_argument("--vision_encoder", type=str, default=DEFAULT_VISION_ENCODER, help="Vision encoder identifier")
    parser.add_argument("--audio_encoder", type=str, default=DEFAULT_AUDIO_ENCODER, help="Audio encoder identifier")
    parser.add_argument("--mm_projector_type", type=str, default="stc", help="Multimodal projector type")
    parser.add_argument("--data_path", type=str, default="data/train.jsonl", help="Training data JSON/JSONL path")
    parser.add_argument("--video_dir", type=str, default="data/videos", help="Base directory for video files")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/videollm-7b", help="Checkpoint output directory"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of sampled frames per video")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True, help="Enable LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Enable bf16 training")
    parser.add_argument("--report_to", type=str, default="wandb", help="Integrations for Trainer logging")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument(
        "--freeze_vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze vision encoder weights",
    )
    parser.add_argument(
        "--freeze_audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze audio encoder weights",
    )
    parser.add_argument(
        "--freeze_llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze LLM backbone weights",
    )
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True, help="Enable audio branch")
    parser.add_argument(
        "--use_temporal_bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable temporal bridge module",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of update steps to accumulate before backward",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval in optimizer steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of saved checkpoints")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader workers per process")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument(
        "--bridge_lr",
        type=float,
        default=None,
        help="Separate learning rate for temporal bridge (default: same as --learning_rate)",
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help="Load weights from a prior checkpoint (strict=False). Use for stage continuation.",
    )
    parser.add_argument(
        "--audio_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of replacing audio with silence during training (0.0 to 1.0)",
    )
    parser.add_argument(
        "--sample_weights_path",
        type=str,
        default=None,
        help="Path to JSON with per-sample loss weights (from compute_sample_weights.py)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from (e.g. output_dir/checkpoint-3172)",
    )
    return parser


def parse_args() -> tuple[argparse.Namespace, set[str]]:
    """Parse training arguments and track explicit CLI overrides.

    Returns:
        A tuple of parsed arguments and argument names explicitly set from CLI.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Detect which args were explicitly set on CLI by comparing against defaults.
    # NOTE: Python 3.12+ validates default types in set_defaults, so we cannot
    # use argparse.SUPPRESS for type-constrained arguments.
    defaults = _build_parser().parse_args([])
    overrides: set[str] = set()
    for key, value in vars(args).items():
        if key == "config":
            continue
        default_value = getattr(defaults, key, None)
        if value != default_value:
            overrides.add(key)
    return args, overrides


def _load_yaml_config(config_path: str | None) -> dict[str, object]:
    """Load training YAML configuration.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Flattened dictionary of config values from model/training/data sections.
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        loaded_obj: object = yaml.safe_load(file)
    raw: object = loaded_obj if loaded_obj is not None else {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML config must be a mapping")
    raw_map = cast("dict[str, object]", raw)

    flattened: dict[str, object] = {}
    for section_name in ("model", "training", "data"):
        section_value_obj: object = raw_map.get(section_name, {})
        if isinstance(section_value_obj, dict):
            section_map = cast("dict[object, object]", section_value_obj)
            for key_obj, value_obj in section_map.items():
                if isinstance(key_obj, str):
                    flattened[key_obj] = value_obj
    return flattened


def _merge_args(args: argparse.Namespace, overrides: set[str]) -> argparse.Namespace:
    """Merge YAML values into parsed arguments while preserving explicit CLI overrides.

    Args:
        args: Parsed CLI arguments.
        overrides: CLI argument names explicitly provided by user.

    Returns:
        Merged argument namespace.
    """
    yaml_values = _load_yaml_config(args.config)
    for key, value in yaml_values.items():
        if hasattr(args, key) and key not in overrides:
            setattr(args, key, value)
    return args


def _build_dataset(
    data_path: str,
    tokenizer: object,
    image_processor: object,
    num_frames: int,
    max_length: int,
    video_dir: str | None = None,
    audio_dropout_prob: float = 0.0,
    sample_weights_path: str | None = None,
) -> VideoAudioDataset:
    """Build dataset while supporting the evolving constructor interface.

    Args:
        data_path: Training data path.
        tokenizer: HuggingFace tokenizer.
        image_processor: Image processor for frame preprocessing.
        num_frames: Number of video frames.
        max_length: Maximum tokenized length.
        video_dir: Base directory for video files.
        audio_dropout_prob: Probability of replacing audio with silence.
        sample_weights_path: Path to per-sample loss weights JSON.

    Returns:
        Instantiated VideoAudioDataset.
    """
    init_fn = cast("Callable[..., object]", VideoAudioDataset.__init__)
    signature = inspect.signature(init_fn)
    parameter_names = set(signature.parameters.keys())

    constructor = cast("Callable[..., VideoAudioDataset]", VideoAudioDataset)

    # Build kwargs for sample_weights_path if the constructor supports it
    weight_kwargs: dict[str, str | None] = {}
    if "sample_weights_path" in parameter_names and sample_weights_path is not None:
        weight_kwargs["sample_weights_path"] = sample_weights_path

    if "image_processor" in parameter_names:
        return constructor(
            data_path=data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_frames=num_frames,
            max_length=max_length,
            video_dir=video_dir,
            audio_dropout_prob=audio_dropout_prob,
            **weight_kwargs,
        )

    if "video_transform" in parameter_names:
        return VideoAudioDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            video_transform=None,
            audio_transform=None,
            num_frames=num_frames,
            max_length=max_length,
            video_dir=video_dir,
            audio_dropout_prob=audio_dropout_prob,
            **weight_kwargs,
        )

    raise RuntimeError("Unsupported VideoAudioDataset constructor signature")


def _stack_or_pad_1d(tensors: list[object], padding_value: int | float) -> object:
    """Stack equal-length tensors or pad 1D tensors to equal length.

    Args:
        tensors: Tensor list.
        padding_value: Value used for padding.

    Returns:
        Batched tensor.
    """
    if len(tensors) == 0:
        raise ValueError("Expected at least one tensor in collator")

    try:
        return torch.stack(tensors, dim=0)
    except RuntimeError:
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def video_audio_collator(features: list[dict[str, object]]) -> dict[str, object]:
    """Collate multimodal training features for Trainer.

    Args:
        features: Batch feature dictionaries.

    Returns:
        Collated batch dictionary.
    """
    if len(features) == 0:
        raise ValueError("Empty features list in data collator")

    batch: dict[str, object] = {}
    keys = set(features[0].keys())
    for key in keys:
        tensors = [feature[key] for feature in features]
        if key in {"input_ids", "attention_mask"}:
            batch[key] = _stack_or_pad_1d(tensors, padding_value=0)
        elif key == "labels":
            batch[key] = _stack_or_pad_1d(tensors, padding_value=-100)
        elif key in {"pixel_values", "waveforms"}:
            batch[key] = _stack_or_pad_1d(tensors, padding_value=0.0)
        else:
            batch[key] = torch.stack(tensors, dim=0)
    return batch


class WeightedLossTrainer(transformers.Trainer):
    """Trainer subclass that applies per-sample loss weights for reward-weighted SFT.

    When the batch includes ``sample_weight`` tensors, the standard cross-entropy loss
    for each sample is multiplied by its weight before averaging across the batch.
    This enables emphasizing samples where audio is genuinely useful (audio_helps=2.0)
    and de-emphasizing samples where audio hurts (audio_hurts=0.5).
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, object],
        return_outputs: bool = False,
        **kwargs: object,
    ) -> object:
        """Compute weighted cross-entropy loss.

        If ``sample_weight`` is present in inputs, applies per-sample weighting.
        Otherwise falls back to standard loss computation.

        Args:
            model: The model being trained.
            inputs: Batch dictionary with model inputs and optional ``sample_weight``.
            return_outputs: Whether to return model outputs alongside loss.
            **kwargs: Additional keyword arguments passed by Trainer.

        Returns:
            Weighted loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        # Extract sample weights before forward pass (model doesn't expect them)
        sample_weights = inputs.pop("sample_weight", None)

        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss  # (scalar) — standard CE averaged across batch

        if sample_weights is not None:
            # With batch_size=1 (gradient_accumulation handles effective batch),
            # each forward pass processes exactly 1 sample.
            # Scale the standard loss by the sample weight.
            weights = sample_weights.to(loss.device)  # (B,)
            loss = loss * weights.mean()  # scalar

        return (loss, outputs) if return_outputs else loss


def train() -> None:
    """Run VideoLLM training with HuggingFace Trainer + optional DeepSpeed/LoRA."""
    setup_logging()
    parsed_args, overrides = parse_args()
    args = _merge_args(parsed_args, overrides)

    if is_main_process():
        logger.info("Starting VideoLLM training")
        logger.info("Arguments: %s", args)

    model_config = ModelConfig(
        llm_path=args.model_path,
        vision_encoder=args.vision_encoder,
        audio_encoder=args.audio_encoder,
        mm_projector_type=args.mm_projector_type,
        num_frames=args.num_frames,
        freeze_vision=args.freeze_vision,
        freeze_audio=args.freeze_audio,
        freeze_llm=args.freeze_llm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_audio=args.use_audio,
        use_temporal_bridge=args.use_temporal_bridge,
    )
    model = VideoLLM(model_config)

    # Load pre-trained weights from a prior stage (strict=False for architecture changes)
    load_from = getattr(args, "load_from", None)
    if load_from is not None:
        from safetensors.torch import load_model as _load_safetensors

        ckpt_file = Path(load_from) / "model.safetensors"
        if ckpt_file.exists():
            _load_safetensors(model, str(ckpt_file), strict=False)
            if is_main_process():
                logger.info("Loaded pre-trained weights from %s (strict=False)", ckpt_file)
        else:
            if is_main_process():
                logger.warning("--load_from specified but %s not found, starting fresh", ckpt_file)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = transformers.AutoImageProcessor.from_pretrained(args.vision_encoder)
    train_dataset = _build_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_frames=args.num_frames,
        max_length=args.max_length,
        video_dir=args.video_dir,
        audio_dropout_prob=getattr(args, "audio_dropout_prob", 0.0),
        sample_weights_path=getattr(args, "sample_weights_path", None),
    )

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        bf16=args.bf16,
        report_to=args.report_to,
        local_rank=args.local_rank,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
    )

    # Use custom optimizer with separate LR for bridge parameters if bridge_lr is set
    bridge_lr = getattr(args, "bridge_lr", None)
    optimizer = None
    if bridge_lr is not None and model_config.use_temporal_bridge and model.temporal_bridge is not None:
        bridge_param_ids = set(id(p) for p in model.temporal_bridge.parameters())
        bridge_params = [p for p in model.parameters() if id(p) in bridge_param_ids and p.requires_grad]
        other_params = [p for p in model.parameters() if id(p) not in bridge_param_ids and p.requires_grad]

        if is_main_process():
            bridge_count = sum(p.numel() for p in bridge_params)
            other_count = sum(p.numel() for p in other_params)
            logger.info(
                "Param groups: bridge=%d params (lr=%.1e), other=%d params (lr=%.1e)",
                bridge_count,
                bridge_lr,
                other_count,
                args.learning_rate,
            )

        optim_cls = torch.optim.AdamW
        optimizer = optim_cls(
            [
                {"params": other_params, "lr": args.learning_rate},
                {"params": bridge_params, "lr": bridge_lr},
            ],
            weight_decay=args.weight_decay,
        )

    # Choose Trainer: weighted loss if sample weights are provided, else standard
    use_weighted = getattr(args, "sample_weights_path", None) is not None
    trainer_cls = WeightedLossTrainer if use_weighted else transformers.Trainer
    if use_weighted and is_main_process():
        logger.info("Using WeightedLossTrainer for reward-weighted SFT")

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=video_audio_collator,
        processing_class=tokenizer,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt is not None and is_main_process():
        logger.info("Resuming training from checkpoint: %s", resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if is_main_process():
        logger.info("Saved model and tokenizer to %s", output_dir)


if __name__ == "__main__":
    train()
