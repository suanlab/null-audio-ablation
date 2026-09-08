"""Unit tests for video and audio transforms."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import pytest
import torch

from videollm.data.transforms import AudioTransform, VideoTransform, uniform_frame_sample

T_FRAMES = 4
C_CHANNELS = 3
H_SIZE = 64
W_SIZE = 64
IMAGE_SIZE = 32


def test_uniform_frame_sample_total_greater_than_num() -> None:
    """uniform_frame_sample returns in-range indices when total exceeds samples."""
    indices = uniform_frame_sample(total_frames=12, num_frames=4)

    assert len(indices) == 4
    assert indices == sorted(indices)
    assert all(0 <= idx < 12 for idx in indices)


def test_uniform_frame_sample_total_less_or_equal_num() -> None:
    """uniform_frame_sample handles total <= num with bounded outputs."""
    less_indices = uniform_frame_sample(total_frames=3, num_frames=5)
    equal_indices = uniform_frame_sample(total_frames=5, num_frames=5)

    assert len(less_indices) == 5
    assert all(0 <= idx < 3 for idx in less_indices)
    assert equal_indices == [0, 1, 2, 3, 4]


def test_uniform_frame_sample_raises_for_non_positive_inputs() -> None:
    """uniform_frame_sample validates positive frame counts."""
    with pytest.raises(ValueError, match="total_frames must be > 0"):
        uniform_frame_sample(total_frames=0, num_frames=4)

    with pytest.raises(ValueError, match="num_frames must be > 0"):
        uniform_frame_sample(total_frames=4, num_frames=0)


def test_video_transform_resizes_and_normalizes_clip() -> None:
    """VideoTransform returns normalized clip with configured output size."""
    transform = VideoTransform(image_size=IMAGE_SIZE, is_train=False)
    frames = torch.full((T_FRAMES, C_CHANNELS, H_SIZE, W_SIZE), 0.5)  # (T, C, H, W)

    output = transform(frames)  # (T, C, H', W')

    assert output.shape == (T_FRAMES, C_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    expected_channel0 = (0.5 - 0.485) / 0.229
    assert torch.isclose(output[0, 0, 0, 0], torch.tensor(expected_channel0), atol=1e-5)


def test_video_transform_train_enables_augmentation_eval_is_deterministic() -> None:
    """Train mode applies augmentation while eval mode stays deterministic."""
    frames = torch.linspace(0.0, 1.0, steps=T_FRAMES * C_CHANNELS * H_SIZE * W_SIZE)
    frames = frames.view(T_FRAMES, C_CHANNELS, H_SIZE, W_SIZE)  # (T, C, H, W)
    train_transform = VideoTransform(image_size=IMAGE_SIZE, is_train=True)
    eval_transform = VideoTransform(image_size=IMAGE_SIZE, is_train=False)

    torch.manual_seed(7)
    train_output = train_transform(frames.clone())  # (T, C, H', W')
    eval_output_1 = eval_transform(frames.clone())  # (T, C, H', W')
    eval_output_2 = eval_transform(frames.clone())  # (T, C, H', W')

    assert train_output.shape == eval_output_1.shape == (T_FRAMES, C_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(eval_output_1, eval_output_2)
    assert not torch.allclose(train_output, eval_output_1)


def test_audio_transform_resampling_normalization_and_padding() -> None:
    """AudioTransform resamples, normalizes peak amplitude, and pads to target length."""
    transform = AudioTransform(target_sample_rate=16000, duration_seconds=1.0, is_train=False)
    waveform = torch.linspace(-2.0, 2.0, steps=6000)  # (samples,), 0.75s at 8kHz

    output = transform(waveform, source_sample_rate=8000)  # (target_samples,)

    assert output.shape == (16000,)
    assert torch.max(torch.abs(output)) <= 1.0 + 1e-6
    assert torch.allclose(output[12000:], torch.zeros(4000))


def test_audio_transform_truncates_long_waveform() -> None:
    """AudioTransform truncates waveform longer than target duration."""
    transform = AudioTransform(target_sample_rate=16000, duration_seconds=1.0, is_train=False)
    waveform = torch.linspace(-1.0, 1.0, steps=48000)  # (samples,)

    output = transform(waveform, source_sample_rate=16000)  # (target_samples,)

    assert output.shape == (16000,)


def test_audio_transform_train_mode_differs_from_eval() -> None:
    """Train-time masking changes waveform compared with eval mode."""
    train_transform = AudioTransform(target_sample_rate=16000, duration_seconds=1.0, is_train=True)
    eval_transform = AudioTransform(target_sample_rate=16000, duration_seconds=1.0, is_train=False)
    waveform = torch.ones(16000)  # (samples,)

    torch.manual_seed(42)
    train_output = train_transform(waveform.clone(), source_sample_rate=16000)  # (target_samples,)
    eval_output = eval_transform(waveform.clone(), source_sample_rate=16000)  # (target_samples,)

    assert train_output.shape == eval_output.shape == (16000,)
    assert not torch.allclose(train_output, eval_output)


def test_apply_time_mask_zeroes_region() -> None:
    """AudioTransform._apply_time_mask zeros at least one sample when ratio is positive."""
    waveform = torch.ones(1000)  # (samples,)

    torch.manual_seed(1)
    masked = AudioTransform._apply_time_mask(waveform, max_ratio=0.1)  # (samples,)

    assert masked.shape == waveform.shape
    assert torch.count_nonzero(masked == 0.0) > 0
