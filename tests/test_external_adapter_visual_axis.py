"""CMSS visual axis in the external-model adapters.

The external adapters (video-SALMONN 2+, Qwen2.5-Omni) do not decode frames
themselves — they hand a path to upstream code. The visual intervention therefore
hooks the decoded frames just before the model's own preprocessing, mirroring where
each adapter already applies its audio intervention and its POS_BLANK control.

These tests exercise the adapters' real source (extracted without importing the heavy
upstream repos) so a drift in the hook contract fails here rather than silently
producing an un-degraded surface.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
VSALM_PATH = REPO_ROOT / "scripts" / "eval_avqa_videosalmonn2.py"
QWEN_PATH = REPO_ROOT / "scripts" / "eval_avqa_qwenomni.py"

T, C, H, W = 8, 3, 32, 40


def _exec_vd_block(src: str) -> dict[str, Any]:
    """Run an adapter's real `_VD` loader block (the path-based degradation import)."""
    start = src.index('_VD_PATH = Path(__file__)')
    end = src.index("VISUAL_MODES = _VD.VISUAL_MODES") + len("VISUAL_MODES = _VD.VISUAL_MODES")
    block = src[start:end].replace("Path(__file__).resolve().parent.parent", f'Path("{REPO_ROOT}")')
    ns: dict[str, Any] = {"Path": Path, "importlib": importlib, "sys": sys, "torch": torch, "np": np}
    exec(block, ns)  # noqa: S102 - executing our own repo source under test
    return ns


def _frames_u8() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(T, C, H, W), dtype=np.uint8)


# --- shared: the path-based import must work outside the videollm env -------


@pytest.mark.parametrize("path", [VSALM_PATH, QWEN_PATH])
def test_degradation_module_loads_by_path(path: Path) -> None:
    """Regression: without registering the module in sys.modules first, @dataclass
    raises AttributeError — which would crash every external run."""
    ns = _exec_vd_block(path.read_text())
    assert ns["VISUAL_MODES"] == ("clean", "motion_blur", "occlusion", "frame_drop", "downscale")
    assert ns["_VD"].DegradationConfig(mode="occlusion", severity=0.5).severity == 0.5


# --- video-SALMONN 2+: process_video_frames override ------------------------


def _vsalm_override_factory() -> tuple[Any, Any]:
    src = VSALM_PATH.read_text()
    ns = _exec_vd_block(src)
    start = src.index("def _make_process_video_frames_override")
    end = src.index("def _make_process_audio_override")
    exec(src[start:end], ns)  # noqa: S102 - our own source
    return ns["_make_process_video_frames_override"], ns["_VD"]


@pytest.mark.parametrize("cfg_kind", ["none", "clean", "zero_severity"])
def test_vsalm_no_op_returns_original_method(cfg_kind: str) -> None:
    """A no-op config must return the repo's own bound method untouched, so stock
    inference stays byte-identical."""
    make, vd = _vsalm_override_factory()
    cfg = {
        "none": None,
        "clean": vd.DegradationConfig(mode="clean"),
        "zero_severity": vd.DegradationConfig(mode="occlusion", severity=0.0),
    }[cfg_kind]

    def original(video, frame_idx, video_length):  # noqa: ANN001, ANN202
        return ("tensor", "grid", "spg")

    assert make(original, cfg) is original


def test_vsalm_override_preserves_upstream_contract() -> None:
    make, vd = _vsalm_override_factory()
    captured: dict[str, Any] = {}

    def original(video, frame_idx, video_length):  # noqa: ANN001, ANN202
        captured["video"] = video
        captured["frame_idx"] = frame_idx
        captured["video_length"] = video_length
        return ("tensor", "grid", "spg")

    frames = _frames_u8()
    override = make(original, vd.DegradationConfig(mode="occlusion", severity=1.0, seed=7))
    result = override(frames, list(range(T)), 4.0)

    assert result == ("tensor", "grid", "spg")  # the repo's own return value flows through
    assert captured["frame_idx"] == list(range(T))
    assert captured["video_length"] == 4.0
    got = captured["video"]
    assert got.shape == frames.shape, "upstream expects (T, C, H, W)"
    assert got.dtype == np.uint8, "upstream expects uint8"
    assert not np.array_equal(got, frames), "the intervention must actually change pixels"
    assert got[:, :, H // 2, W // 2].max() == 0  # occlusion blanks the centre
    assert got[:, :, 0, 0].min() > 0  # ... and leaves the corners


def test_vsalm_override_is_deterministic() -> None:
    make, vd = _vsalm_override_factory()
    seen: list[np.ndarray] = []

    def original(video, frame_idx, video_length):  # noqa: ANN001, ANN202
        seen.append(video.copy())
        return None

    frames = _frames_u8()
    override = make(original, vd.DegradationConfig(mode="frame_drop", severity=0.66, seed=11))
    override(frames, list(range(T)), 4.0)
    override(frames, list(range(T)), 4.0)
    assert np.array_equal(seen[0], seen[1])


# --- Qwen2.5-Omni: degrade_videos on the decoded tensors --------------------


def _qwen_degrade() -> tuple[Any, Any]:
    src = QWEN_PATH.read_text()
    ns = _exec_vd_block(src)
    start = src.index("def degrade_videos")
    end = src.index("def ", start + 1)
    exec(src[start:end], ns)  # noqa: S102 - our own source
    return ns["degrade_videos"], ns["_VD"]


def test_qwen_no_op_passthrough() -> None:
    degrade, vd = _qwen_degrade()
    vids = [torch.randint(0, 256, (T, C, H, W), dtype=torch.uint8)]
    assert degrade(None, vd.DegradationConfig(mode="occlusion")) is None
    assert degrade(vids, None) is vids
    assert degrade(vids, vd.DegradationConfig(mode="clean")) is vids
    assert degrade(vids, vd.DegradationConfig(mode="occlusion", severity=0.0)) is vids


def test_qwen_preserves_dtype_and_shape() -> None:
    degrade, vd = _qwen_degrade()
    vids = [torch.randint(0, 256, (T, C, H, W), dtype=torch.uint8)]
    out = degrade(vids, vd.DegradationConfig(mode="occlusion", severity=1.0, seed=3))
    assert len(out) == 1
    assert out[0].shape == vids[0].shape
    assert out[0].dtype == torch.uint8
    assert not torch.equal(out[0], vids[0])
    assert int(out[0][:, :, H // 2, W // 2].max()) == 0


def test_qwen_accepts_float_zero_one_tensors() -> None:
    degrade, vd = _qwen_degrade()
    vids = [torch.rand(T, C, H, W)]
    out = degrade(vids, vd.DegradationConfig(mode="downscale", severity=1.0))
    assert out[0].dtype == vids[0].dtype
    assert out[0].shape == vids[0].shape


def test_qwen_rejects_normalised_float_tensors() -> None:
    """Silently clamping an already-normalised tensor to [0, 1] would corrupt the input
    instead of degrading it, so the boundary is validated."""
    degrade, vd = _qwen_degrade()
    vids = [torch.randn(T, C, H, W) * 3.0]  # clearly outside [0, 1]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        degrade(vids, vd.DegradationConfig(mode="occlusion", severity=1.0))


# --- audio null length (external adapters) -----------------------------------


def _vsalm_mode_waveform() -> Any:
    """Extract video-SALMONN's get_mode_waveform without importing the upstream repo."""
    src = VSALM_PATH.read_text()
    ns: dict[str, Any] = {"np": np, "SR": 16000, "CLIP_SECONDS": 30}
    start = src.index("def _safe_load_wav")
    end = src.index("def _make_process_video_frames_override")
    exec(src[start:end], ns)  # noqa: S102 - our own source
    return ns["get_mode_waveform"], ns


def test_silent_null_matches_the_real_recording_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the null used to be a fixed 30 s window.

    video-SALMONN chunks audio into 30 s segments and emits one ``<|audio_pad|>`` block
    per chunk, so a fixed-length null on a longer clip supplies fewer blocks than the
    prompt declares and the model returns an empty string. This was invisible on
    MUSIC-AVQA (every clip < 30 s) and silently voided 223 of 300 AVUT items.
    """
    get_mode_waveform, ns = _vsalm_mode_waveform()
    long_wav = np.zeros(90 * 16000, dtype=np.float32)  # a 90 s recording
    ns["_safe_load_wav"] = lambda _path: long_wav

    silent = get_mode_waveform("silent", "clip.mp4", None, 3.0)
    noise = get_mode_waveform("noise", "clip.mp4", None, 3.0)

    assert len(silent) == len(long_wav), "silent null must span the real recording"
    assert len(noise) == len(long_wav), "noise null must span the real recording"
    assert not silent.any(), "silent must actually be silent"
    assert noise.any(), "noise must not be all-zero"


def test_null_falls_back_to_a_fixed_window_when_audio_is_unreadable() -> None:
    """An unreadable track must still yield a usable null rather than a zero-length one."""
    get_mode_waveform, ns = _vsalm_mode_waveform()
    ns["_safe_load_wav"] = lambda _path: np.zeros(0, dtype=np.float32)
    silent = get_mode_waveform("silent", "broken.mp4", None, 3.0)
    assert len(silent) == 30 * 16000


def test_shuffled_donor_is_length_matched_to_the_item() -> None:
    """Regression: a donor clip of a different length breaks the forward pass.

    video-SALMONN emits one ``<|audio_pad|>`` block per 30 s chunk, so substituting a donor
    waveform of a different duration changes the block count and the model fails to
    broadcast. Seventeen AVUT items failed this way before the fix. The silent and noise
    nulls were already length-matched; shuffled was not, which is the same defect recurring
    in the one audio mode that had escaped the earlier audit.
    """
    get_mode_waveform, ns = _vsalm_mode_waveform()
    own = np.zeros(70 * 16000, dtype=np.float32)   # this item: 70 s
    donor = np.ones(20 * 16000, dtype=np.float32)  # donor: 20 s, shorter
    ns["_safe_load_wav"] = lambda path: donor if path == "donor.mp4" else own

    out = get_mode_waveform("shuffled", "clip.mp4", "donor.mp4", 3.0)
    assert len(out) == len(own), "donor must be resized to the item's own length"
    assert out.any(), "the donor's content must survive; only its duration is adjusted"

    ns["_safe_load_wav"] = lambda path: (
        np.ones(200 * 16000, dtype=np.float32) if path == "donor.mp4" else own
    )
    longer = get_mode_waveform("shuffled", "clip.mp4", "donor.mp4", 3.0)
    assert len(longer) == len(own), "a longer donor must be truncated, not left as-is"
