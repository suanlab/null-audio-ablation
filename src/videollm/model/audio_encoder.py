"""Audio encoder tower for VideoLLM using CLAP."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from transformers import ClapAudioModelWithProjection, ClapProcessor

from videollm.data.constants import DEFAULT_AUDIO_ENCODER, DEFAULT_AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioTower(nn.Module):
    """CLAP-based audio encoder wrapper that returns temporal features.

    Args:
        model_name: HuggingFace CLAP checkpoint identifier.
        freeze: Whether to freeze encoder parameters.
        sample_rate: Input waveform sample rate in Hz.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_AUDIO_ENCODER,
        freeze: bool = True,
        sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        self.model_name = model_name
        self.sample_rate = sample_rate

        self._model = ClapAudioModelWithProjection.from_pretrained(model_name)
        self._processor = ClapProcessor.from_pretrained(model_name)

        if freeze:
            self.requires_grad_(False)

        logger.info(
            "Loaded audio tower: %s hidden=%d sample_rate=%d",
            model_name,
            self.hidden_size,
            sample_rate,
        )

    @property
    def hidden_size(self) -> int:
        """Return hidden feature dimension for sequence-level audio states."""
        return int(self._model.config.hidden_size)

    @property
    def dtype(self) -> torch.dtype:
        """Return model parameter dtype."""
        return next(self._model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Return model parameter device."""
        return next(self._model.parameters()).device

    @property
    def processor(self) -> ClapProcessor:
        """Return CLAP processor used for waveform preprocessing."""
        return self._processor

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode waveforms into temporal audio tokens.

        Args:
            waveforms: Raw waveform tensor of shape ``(B, S)`` at ``sample_rate``.

        Returns:
            Sequence features of shape ``(B, T_a, D)``.
        """
        input_features = self._prepare_input_features(waveforms)  # (B, F, T) or (B, 1, F, T)
        outputs = self._model.audio_model(input_features=input_features)  # backbone outputs
        hidden = outputs.last_hidden_state

        if hidden.ndim == 4:
            bsz, dim, h_freq, w_time = hidden.shape
            hidden = hidden.reshape(bsz, dim, h_freq * w_time)  # (B, D, T_a)
            hidden = hidden.transpose(1, 2)  # (B, T_a, D)
            return hidden
        if hidden.ndim == 3:
            return hidden  # (B, T_a, D)

        raise ValueError(f"Unexpected CLAP hidden state rank {hidden.ndim}, expected 3 or 4")

    def _prepare_input_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Convert raw audio waveforms to CLAP audio features.

        Args:
            waveforms: Raw waveform tensor of shape ``(B, S)``.

        Returns:
            CLAP-ready features of shape ``(B, F, T)`` or ``(B, 1, F, T)``.
        """
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError(f"waveforms must be torch.Tensor, got {type(waveforms)!r}")
        if waveforms.ndim != 2:
            raise ValueError(f"waveforms must have shape (B, S), got {tuple(waveforms.shape)}")
        if waveforms.shape[1] <= 0:
            raise ValueError("waveforms must include at least one sample")

        waveforms_cpu = waveforms.detach().to(dtype=torch.float32, device="cpu")  # (B, S)
        waveform_list = [np.asarray(row.numpy(), dtype=np.float32) for row in waveforms_cpu]  # list[(S,)]

        processed = self._processor(
            audio=waveform_list,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        input_features = processed.get("input_features")
        if input_features is None:
            raise ValueError("CLAP processor did not return 'input_features'")

        return input_features.to(device=self.device, dtype=self.dtype)


def build_audio_tower(
    model_name: str = DEFAULT_AUDIO_ENCODER,
    freeze: bool = True,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
) -> AudioTower:
    """Build an ``AudioTower`` instance.

    Args:
        model_name: HuggingFace CLAP checkpoint identifier.
        freeze: Whether to freeze encoder parameters.
        sample_rate: Input waveform sample rate in Hz.

    Returns:
        Initialized CLAP ``AudioTower``.
    """
    return AudioTower(model_name=model_name, freeze=freeze, sample_rate=sample_rate)
