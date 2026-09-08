"""Preference pair dataset for DPO training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from videollm.data.constants import (
    DEFAULT_AUDIO_DURATION,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_FRAMES,
    IGNORE_INDEX,
    MODAL_INDEX_MAP,
)
from videollm.data.transforms import AudioTransform, VideoTransform, uniform_frame_sample
from videollm.utils import load_audio_from_video

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for DPO preference pairs with video and audio.

    Annotation format (JSONL)::

        {
            "video": "path/to/video.mp4",
            "prompt": "<video>\\n<audio>\\nDescribe this video.",
            "chosen": "A detailed description of the video...",
            "rejected": "A short or incorrect description..."
        }

    Args:
        data_path: Path to JSONL file with preference pairs.
        tokenizer: HuggingFace tokenizer.
        video_transform: Video preprocessing transform.
        audio_transform: Audio preprocessing transform.
        num_frames: Number of video frames to sample.
        max_length: Maximum token sequence length.
        video_dir: Base directory for resolving relative video paths.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        video_transform: VideoTransform | None = None,
        audio_transform: AudioTransform | None = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        max_length: int = 2048,
        video_dir: str = "",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.video_transform = video_transform or VideoTransform()
        self.audio_transform = audio_transform or AudioTransform()
        self.num_frames = num_frames
        self.max_length = max_length
        self.video_dir = Path(video_dir)

        self.annotations = self._load_annotations(data_path)
        logger.info("Loaded %d preference pairs from %s", len(self.annotations), data_path)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and preprocess a preference pair.

        Returns:
            Dictionary with keys: ``chosen_input_ids``, ``chosen_labels``,
            ``chosen_attention_mask``, ``rejected_input_ids``, ``rejected_labels``,
            ``rejected_attention_mask``, ``pixel_values``, ``waveforms``.
        """
        ann = self.annotations[idx]
        video_value = ann.get("video")
        video_path = self.video_dir / str(video_value) if video_value is not None else None

        # --- Load video frames ---
        pixel_values = torch.zeros(self.num_frames, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)  # (T, C, H, W)
        if video_path is not None and video_path.exists():
            try:
                pixel_values = self._load_video(str(video_path))  # (T, C, H, W)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Failed to load video from %s, using black frames: %s", video_path, exc)

        # --- Load audio ---
        duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
        waveform = torch.zeros(duration_samples)  # (samples,)
        if video_path is not None and video_path.exists():
            waveform = self._load_audio(str(video_path))  # (samples,)

        # --- Tokenize chosen and rejected ---
        prompt = str(ann.get("prompt", ""))
        chosen = str(ann.get("chosen", ""))
        rejected = str(ann.get("rejected", ""))

        if chosen == "" or rejected == "":
            raise ValueError("Preference sample must include non-empty 'chosen' and 'rejected' fields")

        chosen_ids, chosen_labels = self._tokenize_pair(prompt, chosen)
        rejected_ids, rejected_labels = self._tokenize_pair(prompt, rejected)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": (chosen_ids != pad_id).long(),
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": (rejected_ids != pad_id).long(),
            "pixel_values": pixel_values,
            "waveforms": waveform,
        }

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize_pair(
        self,
        prompt: str,
        response: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a prompt-response pair for DPO.

        The prompt portion is masked with IGNORE_INDEX in labels so that
        only the response contributes to the log-probability.

        Args:
            prompt: Human prompt text (may contain modal tokens).
            response: Model response text.

        Returns:
            Tuple of ``(input_ids, labels)`` with shape ``(max_length,)``.
        """
        full_text = prompt + response + self.tokenizer.eos_token

        # Strip modal tokens and record which ones appear
        modal_tokens_present: list[int] = []
        processed = full_text
        for token_str, token_idx in MODAL_INDEX_MAP.items():
            if token_str in processed:
                modal_tokens_present.append(token_idx)
                processed = processed.replace(token_str, "")

        # Also strip from prompt to measure its token length
        prompt_stripped = prompt
        for token_str in MODAL_INDEX_MAP:
            prompt_stripped = prompt_stripped.replace(token_str, "")

        n_modal = len(modal_tokens_present)

        encoded = self.tokenizer(
            processed,
            max_length=self.max_length - n_modal,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        text_ids = encoded["input_ids"].squeeze(0)  # (L,)

        # Prepend modal token indices, then pad to max_length
        prefix = torch.tensor(modal_tokens_present, dtype=text_ids.dtype)
        input_ids = torch.cat([prefix, text_ids])  # (n_modal + L,)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        pad_len = self.max_length - input_ids.shape[0]
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])
        else:
            input_ids = input_ids[: self.max_length]

        # Measure prompt token length (to know where response starts)
        prompt_encoded = self.tokenizer(prompt_stripped, return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1] + n_modal

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        labels[input_ids == pad_id] = IGNORE_INDEX

        return input_ids, labels

    # ------------------------------------------------------------------
    # Video / Audio loading (same as VideoAudioDataset)
    # ------------------------------------------------------------------

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video frames using decord.

        Args:
            video_path: Path to video file.

        Returns:
            Tensor of shape ``(T, C, H, W)``.
        """
        import decord

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        indices = uniform_frame_sample(total_frames, self.num_frames)
        frames = vr.get_batch(indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        frames = self.video_transform(frames)  # (T, C, H', W')
        return frames

    def _load_audio(self, video_path: str) -> torch.Tensor:
        """Extract and preprocess audio from a video file.

        Args:
            video_path: Path to video file.

        Returns:
            Waveform tensor of shape ``(samples,)``.
        """
        try:
            waveform, sr = load_audio_from_video(video_path)  # (channels, samples)
        except (RuntimeError, OSError, ImportError) as exc:
            logger.warning("Failed to load audio from %s, using silence: %s", video_path, exc)
            duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
            return torch.zeros(duration_samples)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)
        waveform = waveform.squeeze(0)  # (samples,)
        waveform = self.audio_transform(waveform, sr)  # (samples,)
        return waveform

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_annotations(data_path: str) -> list[dict[str, object]]:
        """Load annotations from JSONL file."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Preference data not found: {data_path}")

        annotations: list[dict[str, object]] = []
        with open(path) as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {line_idx} in {data_path}") from exc

                    if not isinstance(raw, dict):
                        raise ValueError(f"Expected JSON object on line {line_idx} in {data_path}, got {type(raw)}")

                    annotations.append(raw)
        return annotations


def preference_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for preference pair batches.

    Args:
        batch: List of sample dicts from ``PreferenceDataset``.

    Returns:
        Batched dictionary with stacked tensors for chosen/rejected pairs.
    """
    chosen_ids = torch.stack([s["chosen_input_ids"] for s in batch])
    chosen_labels = torch.stack([s["chosen_labels"] for s in batch])
    chosen_mask = torch.stack([s["chosen_attention_mask"] for s in batch])
    rejected_ids = torch.stack([s["rejected_input_ids"] for s in batch])
    rejected_labels = torch.stack([s["rejected_labels"] for s in batch])
    rejected_mask = torch.stack([s["rejected_attention_mask"] for s in batch])
    pixel_values = torch.stack([s["pixel_values"] for s in batch])

    # Pad waveforms to max length in batch
    max_audio_len = max(s["waveforms"].shape[0] for s in batch)
    waveforms = torch.zeros(len(batch), max_audio_len)
    for i, s in enumerate(batch):
        wlen = s["waveforms"].shape[0]
        waveforms[i, :wlen] = s["waveforms"]

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_mask,
        "pixel_values": pixel_values,
        "waveforms": waveforms,
    }
