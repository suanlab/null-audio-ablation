"""Tests for the VideoLLM audio encoder tower."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from videollm.model.audio_encoder import AudioTower, build_audio_tower

HIDDEN_SIZE = 64
PROJECTION_DIM = 32
SAMPLE_RATE = 16000
AUDIO_STEPS = 10


class _DummyClapModel(nn.Module):
    """Lightweight CLAP model stub exposing the required interface."""

    def __init__(self) -> None:
        """Initialize tiny config and backend callable."""
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN_SIZE, projection_dim=PROJECTION_DIM)
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.audio_model = MagicMock(side_effect=self._audio_forward)

    def _audio_forward(self, input_features: torch.Tensor) -> MagicMock:
        """Return temporal hidden states for CLAP backbone calls.

        Args:
            input_features: CLAP features of shape ``(B, F, T)``.

        Returns:
            Mock object with ``last_hidden_state`` as ``(B, T_a, D)``.
        """
        batch = input_features.shape[0]
        hidden = torch.randn(batch, AUDIO_STEPS, HIDDEN_SIZE)  # (B, T_a, D)
        return MagicMock(last_hidden_state=hidden)


def _build_dummy_processor() -> MagicMock:
    """Create a processor stub returning CLAP ``input_features``."""

    def _process(*, audio: list[object], **_: object) -> dict[str, torch.Tensor]:
        """Convert waveform list into tiny random features.

        Args:
            audio: List of waveform arrays.

        Returns:
            Mapping with ``input_features`` tensor.
        """
        batch = len(audio)
        return {"input_features": torch.randn(batch, 8, 6)}  # (B, F, T)

    return MagicMock(side_effect=_process)


def test_audio_tower_forward_produces_temporal_features() -> None:
    """AudioTower forward should map ``(B, samples)`` to ``(B, T_a, D)``."""
    model = _DummyClapModel()
    processor = _build_dummy_processor()
    with (
        patch("videollm.model.audio_encoder.ClapAudioModelWithProjection.from_pretrained", return_value=model),
        patch("videollm.model.audio_encoder.ClapProcessor.from_pretrained", return_value=processor),
    ):
        tower = AudioTower(model_name="laion/mock-clap", freeze=False, sample_rate=SAMPLE_RATE)

    waveforms = torch.randn(2, SAMPLE_RATE)  # (B, samples)
    output = tower(waveforms)  # (B, T_a, D)
    assert output.shape == (2, AUDIO_STEPS, HIDDEN_SIZE)


def test_audio_tower_properties_report_hidden_projection_dtype_and_device() -> None:
    """AudioTower should expose hidden size, projection dim, dtype, and device."""
    model = _DummyClapModel()
    processor = _build_dummy_processor()
    with (
        patch("videollm.model.audio_encoder.ClapAudioModelWithProjection.from_pretrained", return_value=model),
        patch("videollm.model.audio_encoder.ClapProcessor.from_pretrained", return_value=processor),
    ):
        tower = AudioTower(model_name="laion/mock-clap", freeze=False, sample_rate=SAMPLE_RATE)

    assert tower.hidden_size == HIDDEN_SIZE
    assert tower._model.config.projection_dim == PROJECTION_DIM
    assert tower.dtype == torch.float32
    assert tower.device.type == "cpu"


def test_audio_tower_freeze_disables_gradients() -> None:
    """AudioTower with ``freeze=True`` should disable all trainable params."""
    model = _DummyClapModel()
    processor = _build_dummy_processor()
    with (
        patch("videollm.model.audio_encoder.ClapAudioModelWithProjection.from_pretrained", return_value=model),
        patch("videollm.model.audio_encoder.ClapProcessor.from_pretrained", return_value=processor),
    ):
        tower = AudioTower(model_name="laion/mock-clap", freeze=True, sample_rate=SAMPLE_RATE)

    assert all(not parameter.requires_grad for parameter in tower.parameters())


def test_build_audio_tower_factory_returns_audio_tower_instance() -> None:
    """build_audio_tower should return an ``AudioTower`` instance."""
    model = _DummyClapModel()
    processor = _build_dummy_processor()
    with (
        patch("videollm.model.audio_encoder.ClapAudioModelWithProjection.from_pretrained", return_value=model),
        patch("videollm.model.audio_encoder.ClapProcessor.from_pretrained", return_value=processor),
    ):
        tower = build_audio_tower(model_name="laion/mock-clap", freeze=False, sample_rate=SAMPLE_RATE)

    assert isinstance(tower, AudioTower)
