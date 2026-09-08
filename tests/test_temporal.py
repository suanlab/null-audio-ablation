"""Unit tests for the AGTA temporal bridge (CHECK.md B.3 fix: 2026-05-20).

Old tests imported `GatedFusion` and `TemporalImportanceEstimator`, which were
removed in the AGTA v2 simplification. We now test only the surviving public
surface and the load-bearing zero-init claims documented in `temporal.py`.
"""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import torch

from videollm.model.temporal import AudioVisualTemporalBridge, CrossModalAttention

D_MODEL = 64
N_HEADS = 4
BATCH = 2
T_VISUAL = 8
T_AUDIO = 4


def test_cross_modal_attention_output_and_gradients() -> None:
    """CrossModalAttention maps audio queries to visual key/value context."""
    module = CrossModalAttention(d_model=D_MODEL, n_heads=N_HEADS)
    audio_q = torch.randn(BATCH, T_AUDIO, D_MODEL, requires_grad=True)  # (B, T_a, D)
    visual_kv = torch.randn(BATCH, T_VISUAL, D_MODEL, requires_grad=True)  # (B, T_v, D)

    output = module(audio_q=audio_q, visual_kv=visual_kv)  # (B, T_a, D)

    assert output.shape == (BATCH, T_AUDIO, D_MODEL)

    output.sum().backward()
    assert audio_q.grad is not None
    assert visual_kv.grad is not None


def test_cross_modal_attention_out_proj_is_zero_initialized() -> None:
    """The cross-attention out_proj weight and bias must be initialized to zero,
    so the bridge delta is exactly zero at init. This is the load-bearing
    architectural claim of AGTA v2 (paper §3.3 "zero-init residual")."""
    module = CrossModalAttention(d_model=D_MODEL, n_heads=N_HEADS)
    assert torch.all(module.attn.out_proj.weight == 0.0)
    assert torch.all(module.attn.out_proj.bias == 0.0)


def test_cross_modal_attention_delta_is_zero_at_init() -> None:
    """Because out_proj is zero-initialized, the attention output (the delta)
    must be exactly zero for any input at initialization."""
    module = CrossModalAttention(d_model=D_MODEL, n_heads=N_HEADS)
    audio_q = torch.randn(BATCH, T_AUDIO, D_MODEL)
    visual_kv = torch.randn(BATCH, T_VISUAL, D_MODEL)
    with torch.no_grad():
        delta = module(audio_q=audio_q, visual_kv=visual_kv)
    assert torch.all(delta == 0.0)


def test_audio_visual_temporal_bridge_output_shape_and_gradients() -> None:
    """AudioVisualTemporalBridge returns concatenated visual and fused audio tokens."""
    bridge = AudioVisualTemporalBridge(d_model=D_MODEL, n_heads=N_HEADS)
    visual = torch.randn(BATCH, T_VISUAL, D_MODEL, requires_grad=True)  # (B, T_v, D)
    audio = torch.randn(BATCH, T_AUDIO, D_MODEL, requires_grad=True)  # (B, T_a, D)

    output = bridge(visual_tokens=visual, audio_tokens=audio)  # (B, T_v + T_a, D)

    assert output.shape == (BATCH, T_VISUAL + T_AUDIO, D_MODEL)

    output.sum().backward()
    assert visual.grad is not None
    assert audio.grad is not None


def test_bridge_alpha_starts_at_zero() -> None:
    """Learnable residual scale α must be initialized to 0.0 so the bridge
    starts as a no-op (paper §3.3 Eq. 2)."""
    bridge = AudioVisualTemporalBridge(d_model=D_MODEL, n_heads=N_HEADS)
    assert bridge.alpha.item() == 0.0
    assert bridge.alpha.requires_grad


def test_bridge_is_noop_at_init() -> None:
    """At initialization (α=0, out_proj=0), the bridge must pass audio tokens
    through unchanged and concatenate them after visual tokens. This is the
    contract the paper's §3.3 description rests on."""
    bridge = AudioVisualTemporalBridge(d_model=D_MODEL, n_heads=N_HEADS)
    visual = torch.randn(BATCH, T_VISUAL, D_MODEL)
    audio = torch.randn(BATCH, T_AUDIO, D_MODEL)
    with torch.no_grad():
        output = bridge(visual_tokens=visual, audio_tokens=audio)
    assert torch.allclose(output[:, :T_VISUAL], visual)
    assert torch.allclose(output[:, T_VISUAL:], audio)


def test_interpolate_temporal_identity_for_same_length() -> None:
    """Temporal interpolation is identity when source and target lengths match."""
    source = torch.randn(BATCH, T_AUDIO, D_MODEL)  # (B, T_src, C)

    interpolated = AudioVisualTemporalBridge.interpolate_temporal(source, target_len=T_AUDIO)  # (B, T_tar, C)

    assert interpolated is source


def test_interpolate_temporal_changes_length() -> None:
    """Temporal interpolation returns correctly resized sequence."""
    source = torch.randn(BATCH, T_AUDIO, D_MODEL)  # (B, T_src, C)

    upsampled = AudioVisualTemporalBridge.interpolate_temporal(source, target_len=T_VISUAL)  # (B, T_tar, C)

    assert upsampled.shape == (BATCH, T_VISUAL, D_MODEL)
