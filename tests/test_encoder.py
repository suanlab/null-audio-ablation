"""Tests for the VideoLLM vision encoder tower."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from videollm.model.encoder import VisionTower, build_vision_tower

HIDDEN_SIZE = 64
IMAGE_SIZE = 32
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


class _DummyVisionModel(nn.Module):
    """Lightweight mock-compatible vision model for encoder tests."""

    def __init__(self, has_cls_token: bool) -> None:
        """Initialize a tiny vision model with deterministic hidden states.

        Args:
            has_cls_token: Whether returned hidden states include CLS at index 0.
        """
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN_SIZE, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self._has_cls_token = has_cls_token
        self.last_hidden_state: torch.Tensor | None = None

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> MagicMock:
        """Return deterministic hidden states for a batch.

        Args:
            pixel_values: Input image batch of shape ``(B, C, H, W)``.
            output_hidden_states: Unused flag preserved for API parity.

        Returns:
            Mock output object with a ``hidden_states`` list.
        """
        del output_hidden_states
        batch = pixel_values.shape[0]
        token_count = NUM_PATCHES + int(self._has_cls_token)
        values = torch.arange(batch * token_count * HIDDEN_SIZE, dtype=pixel_values.dtype, device=pixel_values.device)
        hidden = values.reshape(batch, token_count, HIDDEN_SIZE)  # (B, N(+1), D)
        self.last_hidden_state = hidden
        return MagicMock(hidden_states=[hidden - 1.0, hidden, hidden + 1.0])


def test_vision_tower_forward_produces_patch_tokens() -> None:
    """VisionTower forward should map ``(B, C, H, W)`` to ``(B, N, D)``."""
    model = _DummyVisionModel(has_cls_token=False)
    with (
        patch("videollm.model.encoder.SiglipVisionModel.from_pretrained", return_value=model),
        patch("videollm.model.encoder.SiglipImageProcessor.from_pretrained", return_value=MagicMock()),
    ):
        tower = VisionTower(model_name="google/siglip-tiny", freeze=False, select_layer=-2)

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)  # (B, C, H, W)
    output = tower(images)  # (B, N, D)
    assert output.shape == (2, NUM_PATCHES, HIDDEN_SIZE)


def test_vision_tower_properties_report_config_and_tensor_metadata() -> None:
    """VisionTower properties should expose model config, dtype, and device."""
    model = _DummyVisionModel(has_cls_token=False)
    with (
        patch("videollm.model.encoder.SiglipVisionModel.from_pretrained", return_value=model),
        patch("videollm.model.encoder.SiglipImageProcessor.from_pretrained", return_value=MagicMock()),
    ):
        tower = VisionTower(model_name="google/siglip-tiny", freeze=False)

    assert tower.hidden_size == HIDDEN_SIZE
    assert tower.image_size == IMAGE_SIZE
    assert tower.patch_size == PATCH_SIZE
    assert tower.num_patches == NUM_PATCHES
    assert tower.dtype == torch.float32
    assert tower.device.type == "cpu"


def test_vision_tower_freeze_disables_gradients() -> None:
    """VisionTower with ``freeze=True`` should set all params to no-grad."""
    model = _DummyVisionModel(has_cls_token=False)
    with (
        patch("videollm.model.encoder.SiglipVisionModel.from_pretrained", return_value=model),
        patch("videollm.model.encoder.SiglipImageProcessor.from_pretrained", return_value=MagicMock()),
    ):
        tower = VisionTower(model_name="google/siglip-tiny", freeze=True)

    assert all(not parameter.requires_grad for parameter in tower.parameters())


def test_build_vision_tower_factory_returns_vision_tower_instance() -> None:
    """build_vision_tower should return a ``VisionTower`` instance."""
    model = _DummyVisionModel(has_cls_token=False)
    with (
        patch("videollm.model.encoder.SiglipVisionModel.from_pretrained", return_value=model),
        patch("videollm.model.encoder.SiglipImageProcessor.from_pretrained", return_value=MagicMock()),
    ):
        tower = build_vision_tower(model_name="google/siglip-tiny", freeze=False)

    assert isinstance(tower, VisionTower)


def test_clip_hidden_state_cls_token_is_removed() -> None:
    """CLIP hidden states should drop the CLS token at index 0."""
    model = _DummyVisionModel(has_cls_token=True)
    with (
        patch("videollm.model.encoder.CLIPVisionModel.from_pretrained", return_value=model),
        patch("videollm.model.encoder.CLIPImageProcessor.from_pretrained", return_value=MagicMock()),
    ):
        tower = VisionTower(model_name="openai/clip-vit-base-patch32", freeze=False, select_layer=-2)

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)  # (B, C, H, W)
    output = tower(images)  # (B, N, D)

    assert model.last_hidden_state is not None
    expected = model.last_hidden_state[:, 1:, :]  # (B, N, D)
    assert torch.equal(output, expected)
