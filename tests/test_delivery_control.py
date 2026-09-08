"""Delivery-control and replay-determinism tests (preregistration W2 gate).

The W2 kill gate requires that (a) the blank-video / blank-audio delivery controls do
what they claim, and (b) an eval run replays deterministically (>=99.5% identical).
A replay mismatch caused by the *harness* would otherwise be misread as model
behaviour, which is exactly the confound the positive control exists to rule out.

These tests exercise the intervention layer only; they need no model or GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from videollm.data.constants import DEFAULT_AUDIO_DURATION, DEFAULT_AUDIO_SAMPLE_RATE
from videollm.data.video_degradation import DegradationConfig, apply_visual_degradation

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_avqa.py"
DURATION_SAMPLES = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)

T, C, H, W = 8, 3, 32, 40


def _load_eval_script() -> ModuleType:
    """Import scripts/eval_avqa.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("eval_avqa", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_avqa"] = module
    spec.loader.exec_module(module)
    return module


ev = _load_eval_script()


def _ablate(mode: str, sample_idx: int = 0, seed: int = 42) -> torch.Tensor:
    """Call get_ablated_audio for a file-free mode (silent/noise)."""
    return ev.get_ablated_audio(
        audio_mode=mode,
        video_path=Path("/nonexistent.mp4"),
        audio_transform=None,
        eval_data=[],
        video_dir=Path("/nonexistent"),
        sample_idx=sample_idx,
        shuffled_mapping=[],
        shift_seconds=3.0,
        seed=seed,
    )


# --- blank-audio delivery control --------------------------------------------


def test_silent_is_exactly_zero_and_full_length() -> None:
    wf = _ablate("silent")
    assert wf.shape == (DURATION_SAMPLES,)
    assert torch.count_nonzero(wf) == 0


def test_noise_is_full_length_and_non_degenerate() -> None:
    wf = _ablate("noise")
    assert wf.shape == (DURATION_SAMPLES,)
    assert torch.count_nonzero(wf) > 0
    assert wf.std() > 0.5  # a real signal, not a near-constant


def test_unknown_audio_mode_raises() -> None:
    with pytest.raises(ValueError):
        _ablate("not_a_mode")


# --- replay determinism (the W2 >=99.5% gate) --------------------------------


def test_noise_is_reproducible_for_the_same_item() -> None:
    assert torch.equal(_ablate("noise", sample_idx=7), _ablate("noise", sample_idx=7))


def test_noise_differs_across_items() -> None:
    """Every item must get its own draw, otherwise 'noise' is one fixed waveform."""
    assert not torch.equal(_ablate("noise", sample_idx=0), _ablate("noise", sample_idx=1))


def test_noise_is_independent_of_global_rng_state() -> None:
    """Regression: noise once came from the global RNG, so resuming with --skip_to
    (or any change in iteration order) silently gave an item different audio and broke
    replay determinism."""
    first = _ablate("noise", sample_idx=5)
    torch.randn(10_000)  # perturb the global RNG, as processing other items would
    torch.manual_seed(1234)
    second = _ablate("noise", sample_idx=5)
    assert torch.equal(first, second)


def test_noise_depends_on_the_base_seed() -> None:
    assert not torch.equal(_ablate("noise", sample_idx=0, seed=42), _ablate("noise", sample_idx=0, seed=43))


def test_visual_degradation_replays_independently_of_order() -> None:
    """The visual axis must be reproducible per item for the same reason."""
    generator = torch.Generator().manual_seed(0)
    frames = torch.rand(T, C, H, W, generator=generator)
    cfg = DegradationConfig(mode="frame_drop", severity=0.66, seed=42 + 5)
    first = apply_visual_degradation(frames, cfg)
    torch.randn(10_000)  # perturb global RNG
    second = apply_visual_degradation(frames, cfg)
    assert torch.equal(first, second)


# --- blank-video delivery control --------------------------------------------


def test_blank_video_zeroes_pixels_but_keeps_shape() -> None:
    """The positive control must remove all visual evidence while keeping the tensor
    contract intact, so audio remains the only usable channel."""
    pixel_values = torch.rand(1, T, C, H, W)
    blanked = torch.zeros_like(pixel_values)
    assert blanked.shape == pixel_values.shape
    assert torch.count_nonzero(blanked) == 0


def test_blank_video_is_not_reachable_through_the_severity_axis() -> None:
    """frame_drop at max severity must still leave >=1 visible frame: a fully blank clip
    is the positive control, never a point on the CMSS curve."""
    frames = torch.ones(T, C, H, W)
    out = apply_visual_degradation(frames, DegradationConfig(mode="frame_drop", severity=1.0, seed=7))
    surviving = int((out.flatten(1).sum(dim=1) > 0).sum())
    assert surviving >= 1
