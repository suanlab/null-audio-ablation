"""Tests for the visual-degradation intervention axis."""

from __future__ import annotations

import pytest
import torch

from videollm.data.video_degradation import (
    VISUAL_MODES,
    DegradationConfig,
    apply_visual_degradation,
)

T, C, H, W = 16, 3, 48, 64


def _clip() -> torch.Tensor:
    """A deterministic non-degenerate clip in [0, 1], shape (T, C, H, W)."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(T, C, H, W, generator=generator)


def test_clean_is_identity() -> None:
    frames = _clip()
    out = apply_visual_degradation(frames, DegradationConfig(mode="clean", severity=1.0))
    assert torch.equal(out, frames)


@pytest.mark.parametrize("mode", VISUAL_MODES)
def test_zero_severity_is_identity(mode: str) -> None:
    frames = _clip()
    out = apply_visual_degradation(frames, DegradationConfig(mode=mode, severity=0.0))
    assert torch.equal(out, frames)


@pytest.mark.parametrize("mode", [m for m in VISUAL_MODES if m != "clean"])
def test_shape_and_range_preserved(mode: str) -> None:
    frames = _clip()
    out = apply_visual_degradation(frames, DegradationConfig(mode=mode, severity=0.7))
    assert out.shape == frames.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize("mode", [m for m in VISUAL_MODES if m != "clean"])
def test_degradation_changes_frames(mode: str) -> None:
    frames = _clip()
    out = apply_visual_degradation(frames, DegradationConfig(mode=mode, severity=1.0))
    assert not torch.equal(out, frames)


@pytest.mark.parametrize("mode", [m for m in VISUAL_MODES if m != "clean"])
def test_deterministic_given_seed(mode: str) -> None:
    frames = _clip()
    cfg = DegradationConfig(mode=mode, severity=0.6, seed=123)
    a = apply_visual_degradation(frames, cfg)
    b = apply_visual_degradation(frames, cfg)
    assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["motion_blur", "occlusion", "downscale"])
def test_severity_monotonic_distortion(mode: str) -> None:
    """Higher severity moves further from the original (monotone distortion)."""
    frames = _clip()
    dists = [
        float((apply_visual_degradation(frames, DegradationConfig(mode=mode, severity=s)) - frames).abs().mean())
        for s in (0.25, 0.5, 1.0)
    ]
    assert dists[0] < dists[1] < dists[2]


def test_occlusion_zeros_center() -> None:
    frames = torch.ones(T, C, H, W)
    out = apply_visual_degradation(frames, DegradationConfig(mode="occlusion", severity=1.0))
    # center pixel must be blacked out, a corner must survive
    assert float(out[:, :, H // 2, W // 2].max()) == 0.0
    assert float(out[:, :, 0, 0].min()) == 1.0


def test_frame_drop_blacks_expected_count_and_keeps_one() -> None:
    frames = torch.ones(T, C, H, W)
    out = apply_visual_degradation(frames, DegradationConfig(mode="frame_drop", severity=1.0, seed=7))
    per_frame_sum = out.flatten(1).sum(dim=1)  # (T,)
    n_black = int((per_frame_sum == 0.0).sum())
    assert n_black == round(1.0 * 0.90 * T)  # MAX_DROP_FRAC = 0.90
    assert n_black <= T - 1  # never blanks the whole clip (that is the positive control)


@pytest.mark.parametrize(
    ("mode", "severity"),
    [("not_a_mode", 0.5), ("clean", 1.5), ("occlusion", -0.1)],
)
def test_invalid_config_raises(mode: str, severity: float) -> None:
    with pytest.raises(ValueError):
        DegradationConfig(mode=mode, severity=severity)


def test_invalid_frames_shape_raises() -> None:
    with pytest.raises(ValueError):
        apply_visual_degradation(torch.rand(C, H, W), DegradationConfig(mode="occlusion"))
    with pytest.raises(ValueError):
        apply_visual_degradation(torch.rand(T, 1, H, W), DegradationConfig(mode="occlusion"))
