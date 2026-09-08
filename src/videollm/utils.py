"""Common utilities for VideoLLM."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the videollm package."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_rank() -> int:
    """Get the current distributed rank, or 0 if not distributed."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    """Get the total number of distributed processes."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """Check if the current process is the main (rank 0) process."""
    return get_rank() == 0


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total number of (trainable) parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(count: int) -> str:
    """Format parameter count for display (e.g. '7.2B', '350M')."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    if count >= 1e6:
        return f"{count / 1e6:.1f}M"
    if count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> dict[str, object]:
    """Load a model checkpoint with informative logging.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        strict: Whether to strictly enforce matching keys.

    Returns:
        Dictionary with 'missing_keys' and 'unexpected_keys'.
    """
    checkpoint_path = str(checkpoint_path)
    logger.info("Loading checkpoint from %s", checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    result = model.load_state_dict(state_dict, strict=strict)

    if result.missing_keys:
        logger.warning("Missing keys: %s", result.missing_keys[:10])
    if result.unexpected_keys:
        logger.warning("Unexpected keys: %s", result.unexpected_keys[:10])

    total_params = count_parameters(model, trainable_only=False)
    logger.info("Model loaded: %s parameters", format_param_count(total_params))

    return {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys}


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze all parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    """Unfreeze all parameters of a module."""
    for param in module.parameters():
        param.requires_grad = True


def load_audio_from_video(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio waveform from a video/audio file using PyAV.

    Falls back to torchaudio if PyAV fails, then to silence.

    Args:
        path: Path to media file.

    Returns:
        Tuple of (waveform, sample_rate). Waveform has shape ``(channels, samples)``.
    """
    path = str(path)
    try:
        import av  # noqa: F811

        container = av.open(path)
        audio_stream = None
        for stream in container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break
        if audio_stream is None:
            logger.warning("No audio stream in %s, returning silence", path)
            from videollm.data.constants import DEFAULT_AUDIO_DURATION, DEFAULT_AUDIO_SAMPLE_RATE

            samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
            return torch.zeros(1, samples), DEFAULT_AUDIO_SAMPLE_RATE

        sample_rate = audio_stream.rate
        frames = []
        for frame in container.decode(audio=0):
            frames.append(frame.to_ndarray())
        container.close()

        import numpy as np

        audio_np = np.concatenate(frames, axis=1)  # (channels, samples)
        waveform = torch.from_numpy(audio_np).float()
        return waveform, sample_rate
    except Exception as exc:
        logger.warning("PyAV audio load failed for %s: %s, returning silence", path, exc)
        from videollm.data.constants import DEFAULT_AUDIO_DURATION, DEFAULT_AUDIO_SAMPLE_RATE

        samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
        return torch.zeros(1, samples), DEFAULT_AUDIO_SAMPLE_RATE
