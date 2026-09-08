"""Tests for VideoLLM projector modules and factories."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from videollm.model.projector import (
    AudioProjector,
    ProjectorConfig,
    STCConnector,
    build_audio_projector,
    build_mlp,
    build_vision_projector,
)

VISION_HIDDEN = 64
AUDIO_HIDDEN = 48
LLM_HIDDEN = 128


def test_projector_config_default_projector_type() -> None:
    """ProjectorConfig should default mm_projector_type to ``stc``."""
    config = ProjectorConfig(mm_hidden_size=VISION_HIDDEN, hidden_size=LLM_HIDDEN, mm_hidden_size_a=AUDIO_HIDDEN)
    assert config.mm_projector_type == "stc"


def test_build_mlp_depth_1_forward_shape() -> None:
    """MLP depth 1 should map input to output hidden size."""
    projector = build_mlp(depth=1, input_dim=VISION_HIDDEN, output_dim=LLM_HIDDEN)
    inputs = torch.randn(2, 5, VISION_HIDDEN)  # (B, T, D_in)
    outputs = projector(inputs)  # (B, T, D_out)
    assert outputs.shape == (2, 5, LLM_HIDDEN)
    assert len(projector) == 1


def test_build_mlp_depth_2_forward_shape() -> None:
    """MLP depth 2 should map input to output hidden size."""
    projector = build_mlp(depth=2, input_dim=VISION_HIDDEN, output_dim=LLM_HIDDEN)
    inputs = torch.randn(2, 5, VISION_HIDDEN)  # (B, T, D_in)
    outputs = projector(inputs)  # (B, T, D_out)
    assert outputs.shape == (2, 5, LLM_HIDDEN)
    assert isinstance(projector[0], nn.Linear)
    assert isinstance(projector[1], nn.GELU)
    assert isinstance(projector[2], nn.Linear)


def test_build_mlp_raises_for_invalid_depth() -> None:
    """MLP build should raise when depth is lower than 1."""
    with pytest.raises(ValueError, match="depth must be positive"):
        build_mlp(depth=0, input_dim=VISION_HIDDEN, output_dim=LLM_HIDDEN)


def test_stc_connector_forward_downsamples_and_projects() -> None:
    """STCConnector should downsample ``(T, H, W)`` by 2 and project channels."""
    config = ProjectorConfig(mm_hidden_size=VISION_HIDDEN, hidden_size=LLM_HIDDEN, mm_hidden_size_a=AUDIO_HIDDEN)
    connector = STCConnector(config)

    inputs = torch.randn(2, 4, 4, 4, VISION_HIDDEN)  # (B, T, H, W, D_v)
    outputs = connector(inputs)  # (B, L, D_llm)

    assert outputs.shape == (2, 8, LLM_HIDDEN)


def test_stc_connector_handles_odd_spatial_dims() -> None:
    """STCConnector should handle odd spatial dimensions via Conv3d truncation."""
    config = ProjectorConfig(mm_hidden_size=VISION_HIDDEN, hidden_size=LLM_HIDDEN, mm_hidden_size_a=AUDIO_HIDDEN)
    connector = STCConnector(config)

    odd_input = torch.randn(1, 4, 3, 5, VISION_HIDDEN)  # (B, T, H, W, D_v) — odd H and W
    output = connector(odd_input)  # Conv3d truncates: T=4->2, H=3->1, W=5->2
    assert output.ndim == 3
    assert output.shape[0] == 1
    assert output.shape[2] == LLM_HIDDEN


def test_audio_projector_forward_shape() -> None:
    """AudioProjector should map ``(B, T_a, D_audio)`` to ``(B, T_a, D_llm)``."""
    config = ProjectorConfig(mm_hidden_size=VISION_HIDDEN, hidden_size=LLM_HIDDEN, mm_hidden_size_a=AUDIO_HIDDEN)
    projector = AudioProjector(config)

    inputs = torch.randn(2, 7, AUDIO_HIDDEN)  # (B, T_a, D_audio)
    outputs = projector(inputs)  # (B, T_a, D_llm)
    assert outputs.shape == (2, 7, LLM_HIDDEN)


def test_build_vision_projector_factory_types() -> None:
    """Vision projector factory should construct stc, mlp, and linear modules."""
    stc_config = ProjectorConfig(
        mm_hidden_size=VISION_HIDDEN,
        hidden_size=LLM_HIDDEN,
        mm_hidden_size_a=AUDIO_HIDDEN,
        mm_projector_type="stc",
    )
    mlp_config = ProjectorConfig(
        mm_hidden_size=VISION_HIDDEN,
        hidden_size=LLM_HIDDEN,
        mm_hidden_size_a=AUDIO_HIDDEN,
        mm_projector_type="mlp",
    )
    linear_config = ProjectorConfig(
        mm_hidden_size=VISION_HIDDEN,
        hidden_size=LLM_HIDDEN,
        mm_hidden_size_a=AUDIO_HIDDEN,
        mm_projector_type="linear",
    )

    assert isinstance(build_vision_projector(stc_config), STCConnector)
    assert isinstance(build_vision_projector(mlp_config), nn.Sequential)
    assert isinstance(build_vision_projector(linear_config), nn.Linear)


def test_build_audio_projector_factory_types() -> None:
    """Audio projector factory should construct mlp-style and linear modules."""
    mlp_config = ProjectorConfig(
        mm_hidden_size=VISION_HIDDEN,
        hidden_size=LLM_HIDDEN,
        mm_hidden_size_a=AUDIO_HIDDEN,
        mm_projector_type="mlp",
    )
    linear_config = ProjectorConfig(
        mm_hidden_size=VISION_HIDDEN,
        hidden_size=LLM_HIDDEN,
        mm_hidden_size_a=AUDIO_HIDDEN,
        mm_projector_type="linear",
    )

    assert isinstance(build_audio_projector(mlp_config), AudioProjector)
    assert isinstance(build_audio_projector(linear_config), nn.Linear)


def test_build_vision_projector_raises_for_unknown_type() -> None:
    """Vision projector factory should raise for unsupported types."""
    config = ProjectorConfig(
        mm_hidden_size=VISION_HIDDEN,
        hidden_size=LLM_HIDDEN,
        mm_hidden_size_a=AUDIO_HIDDEN,
        mm_projector_type="unknown",
    )
    with pytest.raises(ValueError, match="Unsupported mm_projector_type"):
        build_vision_projector(config)
