"""Tests for the timeout-enforceable video decoder.

The regression these guard against: an earlier ``fork``-based implementation worked on
its first call and then failed on every subsequent one, because the parent had already
initialised decord's and torch's native threads. A single-call test passed; only
repeated calls exposed it. Every good video would have been silently recorded as a
failed inference, quietly pulling the measured audio effect toward zero.

Tests needing real media are marked ``slow`` (skipped by default, like the rest of the
suite); the timeout and error paths are exercised without any video file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from videollm.data.video_decode import DecodeRequest, GuardedDecoder, decode_frames
from videollm.data.video_degradation import DegradationConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO_ROOT / "data" / "videos"
NUM_FRAMES, IMAGE_SIZE = 8, 384


def _a_real_video() -> Path | None:
    """Return some decodable clip, or None when the media set is absent."""
    if not VIDEO_DIR.is_dir():
        return None
    return next(iter(sorted(VIDEO_DIR.glob("*.mp4"))), None)


needs_video = pytest.mark.skipif(_a_real_video() is None, reason="data/videos not available")


# --- construction ------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_timeout_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        GuardedDecoder(timeout_s=bad)


# --- failure paths (no media required) ---------------------------------------


def test_missing_file_returns_none_not_raise() -> None:
    """An undecodable item must be reported, not crash the sweep."""
    with GuardedDecoder(timeout_s=30.0) as decoder:
        assert decoder.decode("/nonexistent/clip.mp4", NUM_FRAMES, IMAGE_SIZE, None) is None


def test_timeout_returns_none_and_worker_recovers() -> None:
    """A timeout must kill the worker and leave the decoder usable afterwards —
    otherwise one pathological clip poisons every later item."""
    with GuardedDecoder(timeout_s=0.001) as decoder:  # nothing can finish this fast
        assert decoder.decode("/nonexistent/clip.mp4", NUM_FRAMES, IMAGE_SIZE, None) is None
        # the replacement worker must still serve requests
        assert decoder.decode("/nonexistent/clip.mp4", NUM_FRAMES, IMAGE_SIZE, None) is None


def test_close_is_idempotent() -> None:
    decoder = GuardedDecoder(timeout_s=10.0)
    decoder.close()
    decoder.close()


# --- real media --------------------------------------------------------------


@needs_video
@pytest.mark.slow
def test_decode_frames_shape_and_range() -> None:
    video = _a_real_video()
    assert video is not None
    frames = decode_frames(DecodeRequest(str(video), NUM_FRAMES, IMAGE_SIZE, None))
    assert frames.shape[0] == NUM_FRAMES
    assert frames.shape[1] == 3


@needs_video
@pytest.mark.slow
def test_repeated_decodes_match_the_in_process_result() -> None:
    """Regression for the fork-based implementation: call it repeatedly, not once.

    Every call must return exactly what the in-process path returns; the old code
    returned nothing from call two onward.
    """
    video = _a_real_video()
    assert video is not None
    request = DecodeRequest(str(video), NUM_FRAMES, IMAGE_SIZE, None)
    reference = torch.from_numpy(decode_frames(request))

    with GuardedDecoder(timeout_s=60.0) as decoder:
        for _ in range(5):
            frames = decoder.decode(str(video), NUM_FRAMES, IMAGE_SIZE, None)
            assert frames is not None, "guarded decode returned nothing for a good video"
            assert torch.equal(frames, reference)


@needs_video
@pytest.mark.slow
def test_degradation_is_applied_inside_the_worker() -> None:
    video = _a_real_video()
    assert video is not None
    with GuardedDecoder(timeout_s=60.0) as decoder:
        clean = decoder.decode(str(video), NUM_FRAMES, IMAGE_SIZE, None)
        degraded = decoder.decode(
            str(video), NUM_FRAMES, IMAGE_SIZE, DegradationConfig(mode="occlusion", severity=1.0, seed=1)
        )
    assert clean is not None and degraded is not None
    assert clean.shape == degraded.shape
    assert not torch.equal(clean, degraded)
