"""Multi-modal projectors/connectors for VideoLLM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from timm.layers import LayerNorm2d
from timm.models.regnet import RegStage


@dataclass
class ProjectorConfig:
    """Configuration for vision/audio projectors.

    Attributes:
        mm_hidden_size: Vision encoder hidden size.
        hidden_size: LLM hidden size.
        mm_hidden_size_a: Audio encoder hidden size.
        mm_projector_type: Projector type, one of ``"stc"``, ``"mlp"``, ``"linear"``.
    """

    mm_hidden_size: int
    hidden_size: int
    mm_hidden_size_a: int
    mm_projector_type: str = "stc"


def build_mlp(depth: int, input_dim: int, output_dim: int) -> nn.Sequential:
    """Build a feed-forward MLP.

    Args:
        depth: Number of linear layers.
        input_dim: Input dimension.
        output_dim: Output dimension.

    Returns:
        MLP module.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If dimensions or depth are invalid.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError(f"input_dim/output_dim must be positive, got {input_dim}, {output_dim}")

    layers: list[nn.Module] = []
    for idx in range(depth):
        in_dim = input_dim if idx == 0 else output_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if idx < depth - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class STCConnector(nn.Module):
    """Spatial-Temporal Convolution connector for video tokens.

    Pipeline:
        1) ``s1``: RegStage block projecting from vision dim to LLM dim.
        2) ``sampler``: 3D convolution with kernel/stride ``(2, 2, 2)`` at LLM dim.
        3) ``s2``: Second RegStage block at LLM dim.
        4) ``readout``: MLP at LLM dim for refinement.
    """

    def __init__(self, config: ProjectorConfig) -> None:
        """Initialize the STC connector.

        Args:
            config: Projector configuration.
        """
        super().__init__()
        if config.mm_hidden_size <= 0:
            raise ValueError(f"mm_hidden_size must be positive, got {config.mm_hidden_size}")
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {config.hidden_size}")

        self.s1 = RegStage(
            depth=4,
            in_chs=config.mm_hidden_size,
            out_chs=config.hidden_size,
            stride=1,
            dilation=1,
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,
        )
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(2, 2, 2),
                padding=(0, 0, 0),
                bias=True,
            ),
        )
        self.s2 = RegStage(
            depth=4,
            in_chs=config.hidden_size,
            out_chs=config.hidden_size,
            stride=1,
            dilation=1,
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,
        )
        self.readout = build_mlp(depth=2, input_dim=config.hidden_size, output_dim=config.hidden_size)
        self.register_buffer("output_scale", torch.tensor(0.03))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project video grid features into LLM token space.

        Args:
            x: Vision tensor of shape ``(B, T, H, W, D_v)``.

        Returns:
            Projected tokens of shape ``(B, L, D_llm)`` where ``L = (T/2)*(H/2)*(W/2)``.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.Tensor, got {type(x)!r}")
        if x.ndim != 5:
            raise ValueError(f"x must have shape (B, T, H, W, D), got {tuple(x.shape)}")

        bsz, t_steps, h_size, w_size, channels = x.shape

        x2d = x.permute(0, 1, 4, 2, 3).reshape(bsz * t_steps, channels, h_size, w_size)  # (B*T, D_v, H, W)
        x2d = self.s1(x2d)  # (B*T, D_llm, H, W)

        out_chs = x2d.shape[1]  # D_llm after s1 projection
        x3d = x2d.reshape(bsz, t_steps, out_chs, h_size, w_size).permute(0, 2, 1, 3, 4)  # (B, D_llm, T, H, W)
        x3d = self.sampler(x3d)  # (B, D_llm, T/2, H/2, W/2)

        _, channels_ds, t_ds, h_ds, w_ds = x3d.shape
        x2d_ds = x3d.permute(0, 2, 1, 3, 4).reshape(bsz * t_ds, channels_ds, h_ds, w_ds)  # (B*T', D_llm, H', W')
        x2d_ds = self.s2(x2d_ds)  # (B*T', D_llm, H', W')

        x_tokens = x2d_ds.reshape(bsz, t_ds, channels_ds, h_ds, w_ds)  # (B, T', D_llm, H', W')
        x_tokens = x_tokens.permute(0, 1, 3, 4, 2).reshape(bsz, t_ds * h_ds * w_ds, channels_ds)  # (B, L, D_llm)
        x_tokens = self.readout(x_tokens) * self.output_scale  # (B, L, D_llm)
        return x_tokens


class AudioProjector(nn.Module):
    """Two-layer MLP projector for audio features."""

    def __init__(self, config: ProjectorConfig) -> None:
        """Initialize an audio projector.

        Args:
            config: Projector configuration.
        """
        super().__init__()
        if config.mm_hidden_size_a <= 0:
            raise ValueError(f"mm_hidden_size_a must be positive, got {config.mm_hidden_size_a}")
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {config.hidden_size}")

        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size_a, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.register_buffer("output_scale", torch.tensor(0.03))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features to LLM hidden space.

        Args:
            x: Audio features of shape ``(B, T_a, D_audio)``.

        Returns:
            Projected features of shape ``(B, T_a, D_llm)``.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.Tensor, got {type(x)!r}")
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T_a, D_audio), got {tuple(x.shape)}")
        return self.proj(x) * self.output_scale  # (B, T_a, D_llm)


def build_vision_projector(config: ProjectorConfig) -> nn.Module:
    """Build vision projector connector.

    Args:
        config: Projector configuration.

    Returns:
        Vision projector module.
    """
    projector_type = config.mm_projector_type.lower()
    if projector_type == "stc":
        return STCConnector(config)
    if projector_type == "mlp":
        return build_mlp(depth=2, input_dim=config.mm_hidden_size, output_dim=config.hidden_size)
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    raise ValueError(f"Unsupported mm_projector_type '{config.mm_projector_type}'")


def build_audio_projector(config: ProjectorConfig) -> nn.Module:
    """Build audio projector connector.

    Args:
        config: Projector configuration.

    Returns:
        Audio projector module.
    """
    projector_type = config.mm_projector_type.lower()
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size_a, config.hidden_size)
    return AudioProjector(config)
