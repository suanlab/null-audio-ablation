"""Timeout-enforceable video decoding.

decord's frame extraction hangs indefinitely on some files — intermittently, on files
that decoded fine moments earlier. The eval harness's SIGALRM per-sample timeout cannot
preempt a blocking native call, so one such video stalls an entire sweep.

Isolating the decode in a child process makes the timeout enforceable, but ``fork`` is
not usable here: the parent has already initialised decord's and torch's native threads
by the time the first video is decoded, and a forked child inherits that state and dies
immediately (every call after the first returns nothing). This module therefore uses a
**spawned** worker, started once and reused across calls, and replaces it when it hangs.

The worker is a package module rather than a script function so ``spawn`` can import it
by module path instead of re-executing a ``__main__`` script.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from videollm.data.transforms import VideoTransform, uniform_frame_sample
from videollm.data.video_degradation import DegradationConfig, apply_visual_degradation

if TYPE_CHECKING:
    from multiprocessing.context import SpawnProcess
    from multiprocessing.queues import Queue

    import numpy as np

logger = logging.getLogger(__name__)

_STOP = "__stop__"


@dataclass(frozen=True)
class DecodeRequest:
    """One decode job handed to the worker."""

    video_path: str
    num_frames: int
    image_size: int
    degradation: DegradationConfig | None


def decode_frames(request: DecodeRequest) -> np.ndarray:
    """Decode and preprocess one clip, returning ``(T, C, H, W)`` float32 in ``[0, 1]``.

    Kept identical in behaviour to the in-process path in ``scripts/eval_avqa.py`` so
    guarded and unguarded runs produce the same tensors.

    Args:
        request: The decode job.

    Returns:
        Preprocessed frames as a numpy array of shape ``(T, C, H, W)``.
    """
    import decord

    decord.bridge.set_bridge("torch")
    transform = VideoTransform(image_size=request.image_size, is_train=False)
    reader = decord.VideoReader(request.video_path, num_threads=1)
    indices = uniform_frame_sample(len(reader), request.num_frames)
    frames = reader.get_batch(indices)  # (T, H, W, C)
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
    if request.degradation is not None:
        frames = apply_visual_degradation(frames, request.degradation)  # (T, C, H, W)
    return transform(frames).numpy()  # (T, C, H', W')


def _worker_loop(requests: Queue, responses: Queue) -> None:
    """Serve decode requests until told to stop.

    Results cross the queue as numpy arrays: a torch tensor would be reduced to a
    shared-memory file descriptor that the parent cannot detach once this process exits.
    """
    while True:
        item = requests.get()
        if item == _STOP:
            return
        try:
            responses.put(decode_frames(item))
        except Exception as exc:  # noqa: BLE001 - any failure is an undecodable item
            responses.put(("error", repr(exc)))


class GuardedDecoder:
    """Decode videos in a reusable spawned worker, with a hard per-video timeout.

    A hung worker is killed and replaced, so a single pathological file costs one
    timeout instead of stalling the run. ``decode`` returns ``None`` on timeout or
    failure; callers should record that item as a failed inference rather than dropping
    it, keeping the evaluated item set matched across models.
    """

    def __init__(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self.timeout_s = timeout_s
        self._ctx = mp.get_context("spawn")
        self._requests: Queue | None = None
        self._responses: Queue | None = None
        self._proc: SpawnProcess | None = None

    def _start(self) -> None:
        self._requests = self._ctx.Queue()
        self._responses = self._ctx.Queue()
        self._proc = self._ctx.Process(target=_worker_loop, args=(self._requests, self._responses), daemon=True)
        self._proc.start()

    def _restart(self) -> None:
        self.close()
        self._start()

    def decode(
        self,
        video_path: str,
        num_frames: int,
        image_size: int,
        degradation: DegradationConfig | None,
    ) -> torch.Tensor | None:
        """Decode one clip, or return ``None`` if it times out or fails.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to sample.
            image_size: Output spatial size for the transform.
            degradation: Optional visual-degradation intervention.

        Returns:
            Frames of shape ``(T, C, H, W)``, or ``None``.
        """
        if self._proc is None or not self._proc.is_alive():
            self._restart()
        assert self._requests is not None and self._responses is not None

        self._requests.put(DecodeRequest(video_path, num_frames, image_size, degradation))
        try:
            payload = self._responses.get(timeout=self.timeout_s)
        except queue.Empty:
            logger.warning("Decode timed out after %.0fs, replacing worker: %s", self.timeout_s, video_path)
            self._restart()
            return None
        if isinstance(payload, tuple) and payload and payload[0] == "error":
            logger.warning("Decode failed for %s: %s", video_path, payload[1])
            return None
        return torch.from_numpy(payload)  # (T, C, H, W)

    def close(self) -> None:
        """Stop the worker, killing it if it does not exit promptly."""
        if self._proc is None:
            return
        try:
            if self._proc.is_alive() and self._requests is not None:
                self._requests.put(_STOP)
                self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        finally:
            self._proc = None
            self._requests = None
            self._responses = None

    def __enter__(self) -> GuardedDecoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
