"""Audio-Visual Temporal Bridge (AGTA v2): zero-init residual design.

Key design principles (from Oracle diagnosis of v1 failure):
    1. Bridge starts as **no-op**: output = input + alpha * delta, alpha=0 at init.
    2. Cross-attention out_proj is **zero-initialized** to prevent output explosion.
    3. No TemporalImportance (was constant 0.5, actively harmful).
    4. No GatedFusion (gate was saturated at random 0/1).
    5. No output LayerNorm (was masking 70x scale mismatch).
    6. Learnable ``alpha`` scalar controls bridge contribution, starts at 0.

The old bridge had: cross_attn_std=38, fused/text_ratio=70x, COS(real,noise)=1.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class CrossModalAttention(nn.Module):
    """Audio-query, visual-key/value cross-modal multi-head attention.

    The output projection is **zero-initialized** so the initial cross-attention
    contribution is exactly zero, making the bridge a no-op at init.

    Args:
        d_model: Token hidden dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout.
    """

    def __init__(self, d_model: int = 4096, n_heads: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Zero-init out_proj so bridge_delta = 0 at init
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

        # Scale in_proj_weight to prevent attention logit explosion
        # Xavier uniform for Q,K; small init for V (since out_proj is zero anyway)
        with torch.no_grad():
            fan_in = d_model
            bound = 1.0 / math.sqrt(fan_in)
            self.attn.in_proj_weight.uniform_(-bound, bound)
            if self.attn.in_proj_bias is not None:
                self.attn.in_proj_bias.zero_()

    def forward(self, audio_q: torch.Tensor, visual_kv: torch.Tensor) -> torch.Tensor:
        """Run cross-modal attention.

        Args:
            audio_q: Audio query tokens with shape ``(B, T_a, D)``.
            visual_kv: Visual key/value tokens with shape ``(B, T_v, D)``.

        Returns:
            Cross-attended tokens with shape ``(B, T_a, D)``.
        """
        if not isinstance(audio_q, torch.Tensor):
            raise TypeError(f"audio_q must be torch.Tensor, got {type(audio_q)!r}")
        if not isinstance(visual_kv, torch.Tensor):
            raise TypeError(f"visual_kv must be torch.Tensor, got {type(visual_kv)!r}")
        if audio_q.ndim != 3:
            raise ValueError(f"audio_q must have shape (B, T_a, D), got {tuple(audio_q.shape)}")
        if visual_kv.ndim != 3:
            raise ValueError(f"visual_kv must have shape (B, T_v, D), got {tuple(visual_kv.shape)}")
        if audio_q.shape[0] != visual_kv.shape[0]:
            raise ValueError("audio_q and visual_kv must have the same batch size")
        if audio_q.shape[2] != visual_kv.shape[2]:
            raise ValueError("audio_q and visual_kv must have the same hidden dimension")

        q = self.ln_q(audio_q)  # (B, T_a, D)
        kv = self.ln_kv(visual_kv)  # (B, T_v, D)
        out, _attn_weights = self.attn(q, kv, kv)  # (B, T_a, D)
        return out


class AudioVisualTemporalBridge(nn.Module):
    """AGTA v2: zero-init residual bridge for audio-visual fusion.

    Design: ``fused_audio = audio_tokens + alpha * cross_attn(audio, visual)``

    At initialization, ``alpha = 0`` and ``cross_attn.out_proj = 0``, so the
    bridge is a perfect no-op. During training, the bridge gradually learns to
    inject audio-guided visual information into the audio token stream.

    Args:
        d_model: Token hidden dimension.
        n_heads: Number of heads for cross-modal attention.
        dropout: Dropout probability for attention.
    """

    def __init__(self, d_model: int = 4096, n_heads: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.cross_attn = CrossModalAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # Learnable residual scale — starts at 0 (no-op)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Post-attention layer norm (on the delta, not the output)
        self.delta_norm = nn.LayerNorm(d_model)

    def forward(self, visual_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse visual and audio streams with zero-init residual AGTA.

        Steps:
            1. Cross-attend from audio queries to visual keys/values.
            2. Normalize the attention delta.
            3. Scale by learnable ``alpha`` (starts at 0).
            4. Add residual to audio tokens.
            5. Concatenate ``[visual_tokens, fused_audio_tokens]``.

        Args:
            visual_tokens: Visual tokens with shape ``(B, T_v, D)``.
            audio_tokens: Audio tokens with shape ``(B, T_a, D)``.

        Returns:
            Combined sequence with shape ``(B, T_v + T_a, D)``.
        """
        if not isinstance(visual_tokens, torch.Tensor):
            raise TypeError(f"visual_tokens must be torch.Tensor, got {type(visual_tokens)!r}")
        if not isinstance(audio_tokens, torch.Tensor):
            raise TypeError(f"audio_tokens must be torch.Tensor, got {type(audio_tokens)!r}")
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must have shape (B, T_v, D), got {tuple(visual_tokens.shape)}")
        if audio_tokens.ndim != 3:
            raise ValueError(f"audio_tokens must have shape (B, T_a, D), got {tuple(audio_tokens.shape)}")
        if visual_tokens.shape[0] != audio_tokens.shape[0]:
            raise ValueError("visual_tokens and audio_tokens must have the same batch size")
        if visual_tokens.shape[2] != audio_tokens.shape[2]:
            raise ValueError("visual_tokens and audio_tokens must have the same hidden dimension")

        # Cross-attend: audio queries attend to visual keys/values
        delta = self.cross_attn(audio_q=audio_tokens, visual_kv=visual_tokens)  # (B, T_a, D)
        delta = self.delta_norm(delta)  # (B, T_a, D) — normalize delta magnitude

        # Zero-init residual: alpha starts at 0, bridge is no-op initially
        fused_audio = audio_tokens + self.alpha * delta  # (B, T_a, D)

        return torch.cat([visual_tokens, fused_audio], dim=1)  # (B, T_v + T_a, D)

    @staticmethod
    def interpolate_temporal(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate a temporal sequence to a target length.

        Args:
            x: Input tensor with shape ``(B, T_src, C)``.
            target_len: Target temporal length.

        Returns:
            Interpolated tensor with shape ``(B, target_len, C)``.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.Tensor, got {type(x)!r}")
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T_src, C), got {tuple(x.shape)}")
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")

        if x.shape[1] == target_len:
            return x

        x_channels_first = x.transpose(1, 2)  # (B, C, T_src)
        x_interp = F.interpolate(x_channels_first, size=target_len, mode="linear", align_corners=False)  # (B, C, T_tar)
        return x_interp.transpose(1, 2)  # (B, T_tar, C)
