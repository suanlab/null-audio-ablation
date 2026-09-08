"""DPO (Direct Preference Optimization) trainer for VideoLLM.

Implements the DPO loss from Rafailov et al. (2023):
    L_DPO = -log sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x) - log pi(y_l|x) + log pi_ref(y_l|x)))

where y_w = chosen response, y_l = rejected response, and pi_ref is a frozen
reference model.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import Trainer

from videollm.data.constants import IGNORE_INDEX
from videollm.utils import freeze_module

if TYPE_CHECKING:
    from videollm.model.videollm import VideoLLM

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training.

    Attributes:
        beta: KL divergence penalty coefficient. Higher = more conservative.
        reference_free: Skip reference model (use implicit KL baseline).
        label_smoothing: Label smoothing for the DPO loss.
        loss_type: DPO loss variant (``sigmoid`` or ``hinge``).
    """

    beta: float = 0.1
    reference_free: bool = False
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"


class DPOTrainer(Trainer):
    """HuggingFace Trainer subclass for DPO preference optimization.

    Accepts a ``VideoLLM`` policy model and an optional frozen reference model.
    Overrides ``compute_loss`` to compute the DPO objective.

    Args:
        model: The policy ``VideoLLM`` model to train.
        ref_model: Frozen reference model. If ``None`` and not ``reference_free``,
            a deep copy of *model* is created and frozen.
        dpo_config: DPO hyperparameters.
        **kwargs: Forwarded to ``transformers.Trainer.__init__``.
    """

    def __init__(
        self,
        model: VideoLLM,
        ref_model: VideoLLM | None = None,
        dpo_config: DPOConfig | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.dpo_config = dpo_config or DPOConfig()

        # Build reference model
        if self.dpo_config.reference_free:
            self.ref_model: VideoLLM | None = None
            logger.info("DPO reference-free mode: no reference model")
        elif ref_model is not None:
            self.ref_model = ref_model
            _freeze_model(self.ref_model)
            logger.info("DPO using provided reference model (frozen)")
        else:
            logger.info("DPO creating reference model via deep copy...")
            self.ref_model = copy.deepcopy(model)
            _freeze_model(self.ref_model)
            logger.info("DPO reference model created and frozen")

        self._reference_moved_to_device = False

    # ------------------------------------------------------------------
    # DPO loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DPO loss for a batch of preference pairs.

        Encodes vision/audio once and reuses cached features for both chosen
        and rejected forward passes to reduce peak memory by ~40%.

        Args:
            model: The policy model (``VideoLLM``).
            inputs: Batch dict from ``preference_collate_fn`` containing
                ``chosen_input_ids``, ``rejected_input_ids``, ``pixel_values``,
                ``waveforms``, etc.
            return_outputs: If ``True``, return ``(loss, metrics_dict)``.
            **kwargs: Unused additional keyword arguments.

        Returns:
            DPO loss scalar, or ``(loss, metrics)`` if ``return_outputs`` is set.
        """
        chosen_input_ids = inputs["chosen_input_ids"]  # (B, S)
        chosen_labels = inputs["chosen_labels"]  # (B, S)
        chosen_mask = inputs["chosen_attention_mask"]  # (B, S)
        rejected_input_ids = inputs["rejected_input_ids"]  # (B, S)
        rejected_labels = inputs["rejected_labels"]  # (B, S)
        rejected_mask = inputs["rejected_attention_mask"]  # (B, S)
        pixel_values = inputs.get("pixel_values")  # (B, T, C, H, W) or None
        waveforms = inputs.get("waveforms")  # (B, samples) or None

        if self.ref_model is not None and not self._reference_moved_to_device:
            self.ref_model.to(chosen_input_ids.device)
            self._reference_moved_to_device = True

        cached_embeds = _encode_modalities_once(model, pixel_values, waveforms)

        policy_chosen_logps = _compute_log_probs_cached(
            model, chosen_input_ids, chosen_mask, chosen_labels, cached_embeds
        )  # (B,)
        policy_rejected_logps = _compute_log_probs_cached(
            model, rejected_input_ids, rejected_mask, rejected_labels, cached_embeds
        )  # (B,)

        if self.ref_model is not None:
            with torch.no_grad():
                ref_cached = _encode_modalities_once(self.ref_model, pixel_values, waveforms)
                ref_chosen_logps = _compute_log_probs_cached(
                    self.ref_model, chosen_input_ids, chosen_mask, chosen_labels, ref_cached
                )  # (B,)
                ref_rejected_logps = _compute_log_probs_cached(
                    self.ref_model, rejected_input_ids, rejected_mask, rejected_labels, ref_cached
                )  # (B,)
        else:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        loss, metrics = _dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=self.dpo_config.beta,
            label_smoothing=self.dpo_config.label_smoothing,
            loss_type=self.dpo_config.loss_type,
        )

        if return_outputs:
            return loss, metrics
        return loss


# ======================================================================
# Helper functions (module-level)
# ======================================================================


def _freeze_model(model: nn.Module) -> None:
    """Freeze all parameters of a model and set to eval mode."""
    freeze_module(model)
    model.eval()


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DistributedDataParallel to get the raw module."""
    return model.module if hasattr(model, "module") else model


def _encode_modalities_once(
    model: nn.Module,
    pixel_values: torch.Tensor | None,
    waveforms: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    """Encode vision/audio once and return cached embeddings.

    Args:
        model: ``VideoLLM`` model (may be wrapped in DataParallel).
        pixel_values: ``(B, T, C, H, W)`` video frames or ``None``.
        waveforms: ``(B, samples)`` audio waveforms or ``None``.

    Returns:
        Dict with ``visual_tokens`` and ``audio_tokens``.
    """
    raw = _unwrap(model)
    visual_tokens: torch.Tensor | None = None
    audio_tokens: torch.Tensor | None = None

    with torch.no_grad():
        if pixel_values is not None:
            visual_tokens = raw.encode_video(pixel_values)  # (B, L_v, D)
        if waveforms is not None and raw.audio_tower is not None:
            audio_tokens = raw.encode_audio(waveforms)  # (B, L_a, D)

    if visual_tokens is not None and audio_tokens is not None and raw.temporal_bridge is not None:
        fused = raw.temporal_bridge(visual_tokens, audio_tokens)  # (B, L_v+L_a, D)
        vlen = visual_tokens.shape[1]
        visual_tokens = fused[:, :vlen, :]  # (B, L_v, D)
        audio_tokens = fused[:, vlen:, :]  # (B, L_a, D)

    return {"visual_tokens": visual_tokens, "audio_tokens": audio_tokens}


def _build_embeds_from_cache(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cached: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build input embeddings using pre-cached modal tokens.

    Replaces VIDEO_TOKEN_INDEX / AUDIO_TOKEN_INDEX placeholders with cached
    embeddings and returns padded (embeds, mask, labels).

    Args:
        model: ``VideoLLM`` model.
        input_ids: ``(B, S)`` token ids.
        attention_mask: ``(B, S)`` attention mask.
        labels: ``(B, S)`` labels.
        cached: Pre-computed visual/audio tokens from ``_encode_modalities_once``.

    Returns:
        Tuple of ``(inputs_embeds, attention_mask, labels)`` with modal tokens spliced in.
    """
    from videollm.data.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, MODAL_INDEX_MAP, VIDEO_TOKEN_INDEX

    raw = _unwrap(model)
    visual_tokens = cached["visual_tokens"]
    audio_tokens = cached["audio_tokens"]
    modal_indices = set(MODAL_INDEX_MAP.values())

    embedding_layer = raw.llm.get_input_embeddings()
    embedding_dtype = embedding_layer.weight.dtype
    embedding_device = embedding_layer.weight.device

    batch_embeds: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []

    bsz, seq_len = input_ids.shape
    for b in range(bsz):
        safe_ids = input_ids[b].clone()
        safe_ids[safe_ids < 0] = 0
        text_emb = embedding_layer(safe_ids.to(device=embedding_device)).to(dtype=embedding_dtype)  # (S, D)

        e_segs: list[torch.Tensor] = []
        m_segs: list[torch.Tensor] = []
        l_segs: list[torch.Tensor] = []

        for pos in range(seq_len):
            tid = int(input_ids[b, pos].item())
            if tid == VIDEO_TOKEN_INDEX and visual_tokens is not None:
                vt = visual_tokens[b].to(device=embedding_device, dtype=embedding_dtype)
                e_segs.append(vt)
                m_segs.append(torch.ones(vt.shape[0], dtype=attention_mask.dtype, device=attention_mask.device))
                l_segs.append(torch.full((vt.shape[0],), IGNORE_INDEX, dtype=labels.dtype, device=labels.device))
            elif tid == AUDIO_TOKEN_INDEX and audio_tokens is not None:
                at = audio_tokens[b].to(device=embedding_device, dtype=embedding_dtype)
                e_segs.append(at)
                m_segs.append(torch.ones(at.shape[0], dtype=attention_mask.dtype, device=attention_mask.device))
                l_segs.append(torch.full((at.shape[0],), IGNORE_INDEX, dtype=labels.dtype, device=labels.device))
            elif tid in modal_indices:
                l_segs.append(torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device))
            else:
                e_segs.append(text_emb[pos : pos + 1])
                m_segs.append(attention_mask[b, pos : pos + 1])
                l_segs.append(labels[b, pos : pos + 1])

        batch_embeds.append(torch.cat(e_segs, dim=0))
        batch_masks.append(torch.cat(m_segs, dim=0))
        batch_labels.append(torch.cat(l_segs, dim=0))

    max_len = max(e.shape[0] for e in batch_embeds)
    hid = batch_embeds[0].shape[1]
    pad_e = torch.zeros(bsz, max_len, hid, dtype=batch_embeds[0].dtype, device=batch_embeds[0].device)
    pad_m = torch.zeros(bsz, max_len, dtype=batch_masks[0].dtype, device=batch_masks[0].device)
    pad_l = torch.full((bsz, max_len), IGNORE_INDEX, dtype=batch_labels[0].dtype, device=batch_labels[0].device)

    for b in range(bsz):
        clen = batch_embeds[b].shape[0]
        pad_e[b, :clen] = batch_embeds[b]
        pad_m[b, :clen] = batch_masks[b]
        pad_l[b, :clen] = batch_labels[b]

    return pad_e, pad_m, pad_l


def _compute_log_probs_cached(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cached: dict[str, torch.Tensor | None],
) -> torch.Tensor:
    """Compute per-sequence log-probs using pre-cached modal embeddings.

    Args:
        model: ``VideoLLM`` model (policy or reference).
        input_ids: ``(B, S)`` token ids.
        attention_mask: ``(B, S)`` attention mask.
        labels: ``(B, S)`` labels with ``IGNORE_INDEX`` for prompt tokens.
        cached: Pre-computed visual/audio tokens.

    Returns:
        Per-sequence log-probability sums ``(B,)``.
    """
    inputs_embeds, new_mask, new_labels = _build_embeds_from_cache(model, input_ids, attention_mask, labels, cached)

    raw = _unwrap(model)
    outputs = raw.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=new_mask,
        labels=None,
        return_dict=True,
    )
    logits = outputs.logits  # (B, S', V)

    aligned_labels = new_labels  # (B, S')
    if aligned_labels.shape[1] != logits.shape[1]:
        pad_len = logits.shape[1] - aligned_labels.shape[1]
        if pad_len > 0:
            aligned_labels = nn.functional.pad(aligned_labels, (0, pad_len), value=IGNORE_INDEX)
        else:
            aligned_labels = aligned_labels[:, : logits.shape[1]]

    shift_logits = logits[:, :-1, :].contiguous()  # (B, S'-1, V)
    shift_labels = aligned_labels[:, 1:].contiguous()  # (B, S'-1)

    log_probs = functional.log_softmax(shift_logits, dim=-1)  # (B, S'-1, V)
    gather_labels = shift_labels.clamp(min=0).unsqueeze(-1)  # (B, S'-1, 1)
    per_token_logps = log_probs.gather(dim=-1, index=gather_labels).squeeze(-1)  # (B, S'-1)

    response_mask = (shift_labels != IGNORE_INDEX).float()  # (B, S'-1)
    per_token_logps = per_token_logps * response_mask  # (B, S'-1)

    seq_logps = per_token_logps.sum(dim=-1)  # (B,)
    return seq_logps


def _dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: Policy log-probs for chosen ``(B,)``.
        policy_rejected_logps: Policy log-probs for rejected ``(B,)``.
        ref_chosen_logps: Reference log-probs for chosen ``(B,)``.
        ref_rejected_logps: Reference log-probs for rejected ``(B,)``.
        beta: KL penalty coefficient.
        label_smoothing: Label smoothing factor.
        loss_type: ``sigmoid`` (default) or ``hinge``.

    Returns:
        Tuple of ``(loss, metrics)`` where *metrics* contains reward margins
        and accuracy.
    """
    # Log-ratio margins
    chosen_logratios = policy_chosen_logps - ref_chosen_logps  # (B,)
    rejected_logratios = policy_rejected_logps - ref_rejected_logps  # (B,)
    logits = beta * (chosen_logratios - rejected_logratios)  # (B,)

    if loss_type == "sigmoid":
        # Standard DPO: -log sigmoid(beta * delta)
        if label_smoothing > 0:
            loss = (
                -functional.logsigmoid(logits) * (1 - label_smoothing)
                - functional.logsigmoid(-logits) * label_smoothing
            )
        else:
            loss = -functional.logsigmoid(logits)
    elif loss_type == "hinge":
        loss = torch.relu(1 - logits)
    else:
        raise ValueError(f"Unknown DPO loss_type: {loss_type!r}. Expected 'sigmoid' or 'hinge'.")

    loss = loss.mean()

    # Metrics for logging
    with torch.no_grad():
        chosen_rewards = beta * chosen_logratios.detach()
        rejected_rewards = beta * rejected_logratios.detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (chosen_logratios > rejected_logratios).float().mean()

    metrics = {
        "dpo_loss": loss.detach(),
        "reward_margin": reward_margin,
        "accuracy": accuracy,
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
    }

    return loss, metrics
