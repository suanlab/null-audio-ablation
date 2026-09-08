"""Visual-degradation interventions for the audio-visual grounding protocol.

This module adds a **visual** intervention axis symmetric to the five-mode audio
protocol (``real / silent / noise / shuffled / shifted``). Where the audio modes
degrade the acoustic channel, these degrade the visual channel while leaving audio
intact, so the protocol can test the hypothesis "does a model lean on audio more as
visual SNR drops?" (ARR reviewer K323).

All degradations operate on **raw decoded frames** ``(T, C, H, W)`` with float values
in ``[0, 1]`` — i.e. *before* ``VideoTransform`` resize/normalize, mirroring where the
audio modes intervene on the raw waveform. Each degradation is deterministic given
``seed`` so eval runs are reproducible, and ``severity`` in ``[0, 1]`` maps linearly to
a documented physical range per mode. ``severity == 0`` and mode ``clean`` are both the
identity (the visual analogue of audio ``real``).

The extreme "all-black video" is intentionally *not* reachable here (frame_drop keeps
>=1 frame): the fully-blanked-video condition is the separate positive control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

# Visual intervention modes. "clean" is the identity baseline (analogue of audio "real").
VISUAL_MODES: tuple[str, ...] = ("clean", "motion_blur", "occlusion", "frame_drop", "downscale")

# Per-mode physical range that severity=1.0 maps to. Documented so paper numbers are
# interpretable and reproducible.
MAX_BLUR_FRAC = 0.12  # motion-blur kernel length as a fraction of frame width
MAX_OCC_AREA_FRAC = 0.50  # occlusion patch area as a fraction of frame area
MAX_DROP_FRAC = 0.90  # fraction of frames blacked out (capped so >=1 frame survives)
MIN_DOWNSCALE = 0.10  # smallest relative resolution reached at severity=1.0


@dataclass(frozen=True)
class DegradationConfig:
    """Configuration for a single visual-degradation intervention.

    Args:
        mode: One of :data:`VISUAL_MODES`.
        severity: Intervention strength in ``[0, 1]``; ``0`` is the identity.
        seed: Seed for the (deterministic) random choices in stochastic modes.
    """

    mode: str = "clean"
    severity: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in VISUAL_MODES:
            raise ValueError(f"mode must be one of {VISUAL_MODES}, got {self.mode!r}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")


def _validate_frames(frames: torch.Tensor) -> None:
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape (T, C, H, W), got {tuple(frames.shape)}")
    if frames.shape[1] != 3:
        raise ValueError(f"frames channel dimension must be 3, got {frames.shape[1]}")


def _motion_blur(frames: torch.Tensor, severity: float) -> torch.Tensor:
    """Directional (horizontal) averaging blur, a proxy for camera-pan motion blur."""
    t, c, h, w = frames.shape  # (T, C, H, W)
    max_len = max(1, round(MAX_BLUR_FRAC * w))
    kernel_len = 1 + round(severity * (max_len - 1))
    if kernel_len <= 1:
        return frames
    if kernel_len % 2 == 0:
        kernel_len += 1  # keep odd so padding is symmetric
    pad = kernel_len // 2
    kernel = torch.full((c, 1, 1, kernel_len), 1.0 / kernel_len, dtype=frames.dtype, device=frames.device)
    padded = F.pad(frames, (pad, pad, 0, 0), mode="replicate")  # (T, C, H, W+2*pad)
    blurred = F.conv2d(padded, kernel, groups=c)  # (T, C, H, W)
    return blurred


def _occlusion(frames: torch.Tensor, severity: float) -> torch.Tensor:
    """Zero out a centered rectangular patch covering ``severity * MAX_OCC_AREA_FRAC`` area."""
    t, c, h, w = frames.shape  # (T, C, H, W)
    side_frac = (severity * MAX_OCC_AREA_FRAC) ** 0.5
    box_h = round(side_frac * h)
    box_w = round(side_frac * w)
    if box_h <= 0 or box_w <= 0:
        return frames
    top = (h - box_h) // 2
    left = (w - box_w) // 2
    out = frames.clone()
    out[:, :, top : top + box_h, left : left + box_w] = 0.0  # (T, C, H, W)
    return out


def _frame_drop(frames: torch.Tensor, severity: float, seed: int) -> torch.Tensor:
    """Black out a seeded random ``severity * MAX_DROP_FRAC`` fraction of frames (>=1 kept)."""
    t, c, h, w = frames.shape  # (T, C, H, W)
    n_drop = round(severity * MAX_DROP_FRAC * t)
    n_drop = min(n_drop, t - 1)  # never blank the whole clip; that is the positive control
    if n_drop <= 0:
        return frames
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(t, generator=generator)  # (T,)
    drop_idx = perm[:n_drop]
    out = frames.clone()
    out[drop_idx] = 0.0  # (n_drop, C, H, W)
    return out


def _downscale(frames: torch.Tensor, severity: float) -> torch.Tensor:
    """Bilinear down- then up-sample to erase spatial detail while keeping the shape."""
    t, c, h, w = frames.shape  # (T, C, H, W)
    scale = 1.0 - severity * (1.0 - MIN_DOWNSCALE)
    small_h = max(1, round(scale * h))
    small_w = max(1, round(scale * w))
    if small_h >= h and small_w >= w:
        return frames
    small = F.interpolate(frames, size=(small_h, small_w), mode="bilinear", align_corners=False)  # (T, C, h', w')
    restored = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)  # (T, C, H, W)
    return restored


def apply_visual_degradation(frames: torch.Tensor, config: DegradationConfig) -> torch.Tensor:
    """Apply a visual-degradation intervention to raw decoded frames.

    Args:
        frames: Raw frames, shape ``(T, C, H, W)``, float in ``[0, 1]`` (pre-transform).
        config: The degradation to apply.

    Returns:
        Degraded frames, same shape ``(T, C, H, W)``, clamped to ``[0, 1]``.

    Raises:
        ValueError: If ``frames`` is not a 4-D 3-channel tensor.
    """
    _validate_frames(frames)
    if config.mode == "clean" or config.severity == 0.0:
        return frames

    if config.mode == "motion_blur":
        out = _motion_blur(frames, config.severity)
    elif config.mode == "occlusion":
        out = _occlusion(frames, config.severity)
    elif config.mode == "frame_drop":
        out = _frame_drop(frames, config.severity, config.seed)
    elif config.mode == "downscale":
        out = _downscale(frames, config.severity)
    else:  # unreachable: DegradationConfig validates mode at construction
        raise ValueError(f"unhandled mode {config.mode!r}")

    return out.clamp_(0.0, 1.0)  # (T, C, H, W)
