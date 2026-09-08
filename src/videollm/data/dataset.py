"""Video-Audio-Text dataset loaders for VideoLLM."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUntypedBaseClass=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import ctypes
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from videollm.data.constants import (
    DEFAULT_AUDIO_DURATION,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_NUM_FRAMES,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    MODAL_INDEX_MAP,
)
from videollm.data.transforms import AudioTransform, VideoTransform, uniform_frame_sample
from videollm.utils import load_audio_from_video

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)



class VideoAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Video-audio-text dataset for instruction tuning.

    Annotation format (JSON/JSONL):
    ``{"video": "path/to/video.mp4", "conversations": [{"from": "human", "value": "..."}, ...]}``

    Args:
        data_path: JSON or JSONL annotations path.
        tokenizer: Text tokenizer.
        image_processor: Optional image processor-like object with ``size`` attribute.
        num_frames: Number of sampled frames per sample.
        max_length: Maximum text sequence length before batching.
        video_transform: Optional video transform.
        audio_transform: Optional audio transform.
        video_dir: Optional base directory for relative video paths.
        audio_dropout_prob: Probability of replacing audio with silence (0.0 to 1.0).
        sample_weights_path: Optional path to JSON with per-sample loss weights.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: object | None = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        max_length: int = 2048,
        video_transform: VideoTransform | None = None,
        audio_transform: AudioTransform | None = None,
        video_dir: str | None = None,
        audio_dropout_prob: float = 0.0,
        sample_weights_path: str | None = None,
    ) -> None:
        super().__init__()
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_length = max_length

        inferred_image_size = 384
        if image_processor is not None:
            maybe_size = getattr(image_processor, "size", None)
            if isinstance(maybe_size, dict) and "height" in maybe_size and isinstance(maybe_size["height"], int):
                inferred_image_size = int(maybe_size["height"])

        self.video_transform = (
            video_transform if video_transform is not None else VideoTransform(image_size=inferred_image_size)
        )
        self.audio_transform = audio_transform if audio_transform is not None else AudioTransform()
        self.video_dir = Path(video_dir) if video_dir is not None else Path(".")
        self.audio_dropout_prob = audio_dropout_prob

        if not (0.0 <= audio_dropout_prob <= 1.0):
            raise ValueError(f"audio_dropout_prob must be in [0.0, 1.0], got {audio_dropout_prob}")
        self._annotations = self._load_annotations(Path(data_path))
        self._sample_weights = self._load_sample_weights(sample_weights_path)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load one sample and return model-ready tensors.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with ``input_ids``, ``labels``, ``attention_mask``, ``pixel_values``,
            ``waveforms``, and optionally ``sample_weight``.
        """
        if idx < 0 or idx >= len(self._annotations):
            raise IndexError(f"Index out of range: {idx}")

        annotation = self._annotations[idx]
        video_relative = annotation.get("video")
        if not isinstance(video_relative, str):
            raise ValueError("Annotation field 'video' must be a string")
        video_path = self.video_dir / video_relative
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        conversations_obj = annotation.get("conversations")
        if not isinstance(conversations_obj, list):
            raise ValueError("Annotation field 'conversations' must be a list")

        pixel_values = self._load_video_safe(video_path)  # (T, C, H, W)

        waveforms = self._load_audio(video_path)  # (samples,)
        # Audio dropout: replace audio with silence during training
        if self.audio_dropout_prob > 0.0 and torch.rand(1).item() < self.audio_dropout_prob:
            waveforms = torch.zeros_like(waveforms)
        input_ids, labels = self._tokenize_conversations(conversations_obj)  # (L,), (L,)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)  # (L,)

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "waveforms": waveforms,
        }
        if self._sample_weights is not None:
            result["sample_weight"] = torch.tensor(self._sample_weights[idx], dtype=torch.float32)  # scalar
        return result

    def _load_sample_weights(self, weights_path: str | None) -> list[float] | None:
        """Load per-sample loss weights from a weights JSON file.

        The JSON file should have a ``weights`` list where each entry has ``index`` and ``weight``.
        If no weights file is provided, returns None (uniform weighting).

        Args:
            weights_path: Path to sample weights JSON, or None.

        Returns:
            List of per-sample weights aligned with annotations, or None.
        """
        if weights_path is None:
            return None

        path = Path(weights_path)
        if not path.exists():
            logger.warning("Sample weights file not found: %s — using uniform weights", weights_path)
            return None

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        weights_list = data.get("weights", [])
        if not isinstance(weights_list, list):
            logger.warning("Invalid weights format in %s — using uniform weights", weights_path)
            return None

        # Build index-to-weight mapping
        weight_by_index: dict[int, float] = {}
        for entry in weights_list:
            if isinstance(entry, dict):
                idx = entry.get("index", -1)
                w = entry.get("weight", 1.0)
                if isinstance(idx, int) and isinstance(w, (int, float)):
                    weight_by_index[int(idx)] = float(w)

        # Align weights with annotations (default=1.0 for missing entries)
        num_annotations = len(self._annotations)
        aligned_weights = [weight_by_index.get(i, 1.0) for i in range(num_annotations)]

        loaded_count = sum(1 for w in aligned_weights if w != 1.0)
        logger.info(
            "Loaded %d sample weights from %s (%d non-default out of %d)",
            len(weight_by_index),
            weights_path,
            loaded_count,
            num_annotations,
        )
        return aligned_weights

    def _load_video_safe(self, video_path: Path, timeout_seconds: int = 30) -> torch.Tensor:
        """Load video frames with thread-based timeout that can interrupt C extensions.

        Uses ``ctypes.pythonapi.PyThreadState_SetAsyncExc`` to raise an exception in a
        stuck loader thread, which works even when decord is blocked in C code.

        Args:
            video_path: Video path.
            timeout_seconds: Max seconds before aborting.

        Returns:
            Float tensor ``(T, C, H, W)`` in transformed space, or black frames on failure.
        """
        result_holder: list[torch.Tensor] = []
        error_holder: list[str] = []

        def _worker() -> None:
            try:
                frames = self._load_video(video_path)
                result_holder.append(frames)
            except Exception as exc:
                error_holder.append(str(exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Thread is stuck in C code — force-raise SystemExit via ctypes
            tid = thread.ident
            if tid is not None:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid), ctypes.py_object(SystemExit)
                )
            logger.warning("Video load TIMED OUT for %s after %ds — using black frames", video_path.name, timeout_seconds)
            return self._black_frames()

        if error_holder:
            logger.warning("Video load failed for %s: %s — using black frames", video_path.name, error_holder[0])
            return self._black_frames()

        if result_holder:
            return result_holder[0]

        logger.warning("Video load returned no data for %s — using black frames", video_path.name)
        return self._black_frames()

    def _black_frames(self) -> torch.Tensor:
        """Return zero-filled frames as fallback."""
        image_size = self.video_transform.image_size if hasattr(self.video_transform, "image_size") else 384
        black = torch.zeros(self.num_frames, 3, image_size, image_size, dtype=torch.float32)
        return self.video_transform(black)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Load and sample video frames with decord.

        Args:
            video_path: Video path.

        Returns:
            Float tensor ``(T, C, H, W)`` in transformed space.
        """
        import decord

        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(str(video_path), num_threads=1)

        total_frames = len(reader)
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        frame_indices = uniform_frame_sample(total_frames=total_frames, num_frames=self.num_frames)
        frames = reader.get_batch(frame_indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0  # (T, C, H, W)
        frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
        return self.video_transform(frames)  # (T, C, H', W')

    def _load_audio(self, video_path: Path) -> torch.Tensor:
        """Load audio waveform from media path.

        Args:
            video_path: Media path.

        Returns:
            Mono waveform ``(samples,)`` after transform.
        """
        try:
            waveform, sample_rate = load_audio_from_video(video_path)  # (C, samples)
        except (RuntimeError, OSError) as exc:
            logger.warning("Audio load failed for %s: %s", str(video_path), str(exc))
            sample_count = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
            return torch.zeros(sample_count, dtype=torch.float32)

        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform with shape (C, samples), got {tuple(waveform.shape)}")

        mono = waveform.mean(dim=0)  # (samples,)
        return self.audio_transform(mono, sample_rate)  # (samples,)

    def _tokenize_conversations(self, conversations: list[object]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize dialogue while preserving modal placeholder indices.

        Args:
            conversations: List of turns ``{"from": ..., "value": ...}``.

        Returns:
            Tuple ``(input_ids, labels)`` with shape ``(L,)``.
        """
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token is not None else ""

        sequence_ids: list[int] = []
        label_ids: list[int] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                raise ValueError(f"Each conversation turn must be a dict, got {type(turn)!r}")

            role_obj = turn.get("from")
            text_obj = turn.get("value")
            if not isinstance(role_obj, str) or not isinstance(text_obj, str):
                raise ValueError("Each conversation turn must contain string 'from' and 'value' fields")

            role = role_obj.lower().strip()
            turn_text = f"{text_obj}{eos_token}"
            turn_ids = self._tokenize_with_modal_placeholders(turn_text)
            sequence_ids.extend(turn_ids)

            if role in {"human", "user"}:
                label_ids.extend([IGNORE_INDEX] * len(turn_ids))
            elif role in {"gpt", "assistant"}:
                for token_id in turn_ids:
                    if token_id < 0:
                        label_ids.append(IGNORE_INDEX)
                    else:
                        label_ids.append(token_id)
            else:
                raise ValueError(f"Unsupported conversation role: {role_obj}")

        if len(sequence_ids) == 0:
            empty = torch.zeros(1, dtype=torch.long)
            return empty, torch.full((1,), IGNORE_INDEX, dtype=torch.long)

        if len(sequence_ids) > self.max_length:
            sequence_ids = sequence_ids[-self.max_length :]
            label_ids = label_ids[-self.max_length :]

        input_ids = torch.tensor(sequence_ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        return input_ids, labels

    def _tokenize_with_modal_placeholders(self, text: str) -> list[int]:
        """Tokenize text while replacing special modal strings with modal indices.

        Args:
            text: Raw text potentially containing ``<video>`` and ``<audio>``.

        Returns:
            Token id sequence with placeholder IDs (negative values) inserted.
        """
        if len(text) == 0:
            return []

        modal_tokens = sorted(MODAL_INDEX_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
        ids: list[int] = []
        cursor = 0

        while cursor < len(text):
            matched = False
            for modal_token, modal_index in modal_tokens:
                if text.startswith(modal_token, cursor):
                    ids.append(modal_index)
                    cursor += len(modal_token)
                    matched = True
                    break
            if matched:
                continue

            next_modal_position = len(text)
            for modal_token, _ in modal_tokens:
                pos = text.find(modal_token, cursor)
                if pos != -1 and pos < next_modal_position:
                    next_modal_position = pos

            segment = text[cursor:next_modal_position]
            if len(segment) > 0:
                tokenized = self.tokenizer(segment, add_special_tokens=False)
                token_list_obj = tokenized.get("input_ids")
                if not isinstance(token_list_obj, list):
                    raise ValueError("Tokenizer must return list[int] for input_ids")
                ids.extend(int(token_id) for token_id in token_list_obj)
            cursor = next_modal_position

        if DEFAULT_VIDEO_TOKEN in text and MODAL_INDEX_MAP[DEFAULT_VIDEO_TOKEN] not in ids:
            raise ValueError("Video token appears in text but was not mapped to modal placeholder index")
        return ids

    @staticmethod
    def _load_annotations(path: Path) -> list[dict[str, object]]:
        """Load JSON or JSONL annotations.

        Args:
            path: Input annotation path.

        Returns:
            Annotation list.
        """
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        if path.suffix == ".jsonl":
            annotations: list[dict[str, object]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if len(stripped) == 0:
                        continue
                    try:
                        item = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
                    if not isinstance(item, dict):
                        raise ValueError(f"JSONL item at line {line_no} must be an object")
                    annotations.append(item)
            return annotations

        if path.suffix != ".json":
            raise ValueError(f"Unsupported annotation extension: {path.suffix}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"JSON annotations must be a list, got {type(data)!r}")

        validated: list[dict[str, object]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"JSON item at index {index} must be an object")
            validated.append(item)
        return validated


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dataset samples into a batch.

    This pads variable-length token sequences and stacks fixed-shape video/audio tensors.

    Args:
        batch: Sample list from ``VideoAudioDataset``.

    Returns:
        Batched dictionary with keys ``input_ids``, ``labels``, ``attention_mask``,
        ``pixel_values``, ``waveforms``, and optionally ``sample_weight``.
    """
    if len(batch) == 0:
        raise ValueError("batch must not be empty")

    batch_size = len(batch)
    max_text_len = max(int(item["input_ids"].shape[0]) for item in batch)

    input_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)  # (B, S_max)
    labels = torch.full((batch_size, max_text_len), IGNORE_INDEX, dtype=torch.long)  # (B, S_max)
    attention_mask = torch.zeros(batch_size, max_text_len, dtype=torch.long)  # (B, S_max)

    for sample_idx, sample in enumerate(batch):
        sample_len = int(sample["input_ids"].shape[0])
        input_ids[sample_idx, :sample_len] = sample["input_ids"]  # (S,)
        labels[sample_idx, :sample_len] = sample["labels"]  # (S,)
        attention_mask[sample_idx, :sample_len] = sample["attention_mask"]  # (S,)

    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)  # (B, T, C, H, W)
    waveforms = torch.stack([item["waveforms"] for item in batch], dim=0)  # (B, samples)

    result: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "waveforms": waveforms,
    }
    # Include sample weights if any sample has them
    if "sample_weight" in batch[0]:
        result["sample_weight"] = torch.stack([item["sample_weight"] for item in batch], dim=0)  # (B,)
    return result
