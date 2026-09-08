"""Video and audio transforms/augmentations for VideoLLM."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
import torchaudio
import torchvision.transforms as T  # noqa: N812

from videollm.data.constants import DEFAULT_AUDIO_DURATION, DEFAULT_AUDIO_SAMPLE_RATE, DEFAULT_IMAGE_SIZE


def uniform_frame_sample(total_frames: int, num_frames: int) -> list[int]:
    """Uniformly sample frame indices from a video timeline.

    Args:
        total_frames: Total number of frames in the source video.
        num_frames: Number of desired samples.

    Returns:
        List of sampled frame indices with length ``num_frames``.

    Raises:
        ValueError: If frame counts are invalid.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    if num_frames == 1:
        return [0]

    positions = torch.linspace(0, total_frames - 1, steps=num_frames, dtype=torch.float32)  # (T,)
    indices = positions.round().to(dtype=torch.long).tolist()
    return [min(max(int(index), 0), total_frames - 1) for index in indices]


class VideoTransform:
    """Video frame transform pipeline.

    Train mode applies mild augmentation (flip + color jitter); eval mode is deterministic.

    Args:
        image_size: Output size for height and width.
        is_train: Whether train-time augmentation is enabled.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
    """

    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        is_train: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.image_size = image_size
        self.is_train = is_train

        self.resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        # NOTE: hue=0.0 to avoid RGB→HSV conversion that produces NaN on near-black frames
        self.color_jitter = T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.0)
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Transform a clip tensor.

        Args:
            frames: Video frames, shape ``(T, C, H, W)``, expected in range ``[0, 1]``.

        Returns:
            Transformed frames, shape ``(T, C, H_out, W_out)``.
        """
        if frames.ndim != 4:
            raise ValueError(f"frames must have shape (T, C, H, W), got {tuple(frames.shape)}")
        if frames.shape[1] != 3:
            raise ValueError(f"frames channel dimension must be 3, got {frames.shape[1]}")

        transformed_frames: list[torch.Tensor] = []
        flip_flag = bool(torch.rand(1).item() < 0.5) if self.is_train else False

        for frame_idx in range(frames.shape[0]):
            frame = frames[frame_idx]  # (C, H, W)
            frame = self.resize(frame)  # (C, H', W')
            if self.is_train and flip_flag:
                frame = torch.flip(frame, dims=[2])  # (C, H', W')
            if self.is_train:
                frame = self.color_jitter(frame)  # (C, H', W')
            frame = self.normalize(frame)  # (C, H', W')
            frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            transformed_frames.append(frame)

        return torch.stack(transformed_frames, dim=0)  # (T, C, H', W')


class AudioTransform:
    """Audio waveform transform pipeline.

    The transform performs resampling, amplitude normalization, fixed-length
    pad/truncate, and optional SpecAugment-style masking.

    Args:
        target_sample_rate: Target sample rate for CLAP.
        duration_seconds: Target waveform duration.
        is_train: Whether to enable masking augmentations.
    """

    def __init__(
        self,
        target_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        duration_seconds: float = DEFAULT_AUDIO_DURATION,
        is_train: bool = True,
    ) -> None:
        if target_sample_rate <= 0:
            raise ValueError(f"target_sample_rate must be positive, got {target_sample_rate}")
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")

        self.target_sample_rate = target_sample_rate
        self.target_num_samples = int(target_sample_rate * duration_seconds)
        self.is_train = is_train

    def __call__(self, waveform: torch.Tensor, source_sample_rate: int) -> torch.Tensor:
        """Transform mono waveform.

        Args:
            waveform: Mono waveform tensor of shape ``(samples,)``.
            source_sample_rate: Original sample rate.

        Returns:
            Processed waveform of shape ``(target_samples,)``.
        """
        if waveform.ndim != 1:
            raise ValueError(f"waveform must have shape (samples,), got {tuple(waveform.shape)}")
        if source_sample_rate <= 0:
            raise ValueError(f"source_sample_rate must be positive, got {source_sample_rate}")

        processed = waveform.to(dtype=torch.float32)
        if source_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
            processed = resampler(processed.unsqueeze(0)).squeeze(0)  # (samples')

        peak = torch.max(torch.abs(processed))
        if peak > 0:
            processed = processed / peak

        if processed.shape[0] >= self.target_num_samples:
            processed = processed[: self.target_num_samples]  # (target_samples,)
        else:
            padding = self.target_num_samples - processed.shape[0]
            processed = F.pad(processed, (0, padding))  # (target_samples,)

        if self.is_train:
            processed = self._apply_time_mask(processed, max_ratio=0.08)
            processed = self._apply_frequency_mask(processed, max_ratio=0.08)

        return processed

    @staticmethod
    def _apply_time_mask(waveform: torch.Tensor, max_ratio: float) -> torch.Tensor:
        """Apply random temporal masking on waveform.

        Args:
            waveform: Waveform ``(samples,)``.
            max_ratio: Maximum masked ratio.

        Returns:
            Masked waveform ``(samples,)``.
        """
        if max_ratio <= 0:
            return waveform

        num_samples = waveform.shape[0]
        max_mask = int(num_samples * max_ratio)
        if max_mask <= 1:
            return waveform

        mask_width = int(torch.randint(1, max_mask + 1, (1,)).item())
        start = int(torch.randint(0, num_samples - mask_width + 1, (1,)).item())
        output = waveform.clone()
        output[start : start + mask_width] = 0.0
        return output

    @staticmethod
    def _apply_frequency_mask(waveform: torch.Tensor, max_ratio: float) -> torch.Tensor:
        """Apply random frequency-band masking in the Fourier domain.

        Args:
            waveform: Waveform ``(samples,)``.
            max_ratio: Maximum masked frequency-band ratio.

        Returns:
            Waveform with masked spectral band, shape ``(samples,)``.
        """
        if max_ratio <= 0:
            return waveform

        spectrum = torch.fft.rfft(waveform)  # (F,)
        num_bins = spectrum.shape[0]
        max_band = int(num_bins * max_ratio)
        if max_band <= 1:
            return waveform

        band_width = int(torch.randint(1, max_band + 1, (1,)).item())
        start = int(torch.randint(0, num_bins - band_width + 1, (1,)).item())

        masked = spectrum.clone()
        masked[start : start + band_width] = 0.0
        reconstructed = torch.fft.irfft(masked, n=waveform.shape[0])  # (samples,)
        return reconstructed.to(dtype=waveform.dtype)
