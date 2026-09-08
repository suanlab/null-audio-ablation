"""Main VideoLLM model: Audio-Visual Video Understanding with LLMs."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUntypedBaseClass=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from videollm.data.constants import (
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_ENCODER,
    DEFAULT_LLM,
    DEFAULT_NUM_FRAMES,
    DEFAULT_VISION_ENCODER,
    IGNORE_INDEX,
    MODAL_INDEX_MAP,
    VIDEO_TOKEN_INDEX,
)
from videollm.model.audio_encoder import AudioTower, build_audio_tower
from videollm.model.encoder import VisionTower, build_vision_tower
from videollm.model.projector import (
    AudioProjector,
    ProjectorConfig,
    STCConnector,
    build_audio_projector,
    build_vision_projector,
)
from videollm.model.temporal import AudioVisualTemporalBridge
from videollm.utils import count_parameters, format_param_count, freeze_module

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the Audio-Visual VideoLLM.

    Attributes:
        llm_path: HuggingFace identifier/path for LLM backbone.
        vision_encoder: HuggingFace identifier/path for vision encoder.
        audio_encoder: HuggingFace identifier/path for audio encoder.
        mm_projector_type: Vision projector type. One of ``{"stc", "mlp", "linear"}``.
        num_frames: Expected number of sampled frames per video clip.
        freeze_vision: Freeze vision encoder parameters.
        freeze_audio: Freeze audio encoder parameters.
        freeze_llm: Freeze LLM parameters.
        use_lora: Enable LoRA adapters on LLM.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        use_audio: Enable audio branch.
        use_temporal_bridge: Enable AGTA temporal bridge when both modalities exist.
    """

    llm_path: str = DEFAULT_LLM
    vision_encoder: str = DEFAULT_VISION_ENCODER
    audio_encoder: str = DEFAULT_AUDIO_ENCODER
    mm_projector_type: str = "stc"
    num_frames: int = DEFAULT_NUM_FRAMES
    freeze_vision: bool = True
    freeze_audio: bool = True
    freeze_llm: bool = False
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    use_audio: bool = True
    use_temporal_bridge: bool = True
    use_cache: bool = False
    load_in_4bit: bool = False
    # HuggingFace Trainer compatibility fields (set dynamically after tokenizer init)
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None

class VideoLLM(nn.Module):
    """Audio-Visual VideoLLM model.

    This module integrates SigLIP visual features, CLAP audio features, STC/MLP
    projection, AGTA fusion, and a causal LLM backbone.

    Args:
        config: Model configuration.

    Raises:
        TypeError: If config type is invalid.
        ValueError: If config values are invalid.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.mm_projector_type not in {"stc", "mlp", "linear"}:
            raise ValueError(f"mm_projector_type must be one of stc/mlp/linear, got {config.mm_projector_type}")
        if config.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {config.num_frames}")
        if config.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {config.lora_r}")
        if config.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {config.lora_alpha}")

        self.config = config
        llm_kwargs: dict[str, object] = {}
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Loading LLM in 4-bit (QLoRA mode)")
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_path, **llm_kwargs)  # type: ignore[arg-type]
        llm_hidden_size = int(self.llm.config.hidden_size)

        if config.freeze_llm:
            freeze_module(self.llm)

        self.vision_tower: VisionTower = build_vision_tower(
            model_name=config.vision_encoder, freeze=config.freeze_vision
        )
        vision_proj_cfg = ProjectorConfig(
            mm_hidden_size=self.vision_tower.hidden_size,
            mm_hidden_size_a=self.vision_tower.hidden_size,
            hidden_size=llm_hidden_size,
            mm_projector_type=config.mm_projector_type,
        )
        self.vision_projector: nn.Module = build_vision_projector(vision_proj_cfg)

        self.audio_tower: AudioTower | None = None
        self.audio_projector: AudioProjector | nn.Linear | None = None
        self.temporal_bridge: AudioVisualTemporalBridge | None = None

        if config.use_audio:
            self.audio_tower = build_audio_tower(model_name=config.audio_encoder, freeze=config.freeze_audio)
            audio_proj_cfg = ProjectorConfig(
                mm_hidden_size=self.audio_tower.hidden_size,
                mm_hidden_size_a=self.audio_tower.hidden_size,
                hidden_size=llm_hidden_size,
            )
            self.audio_projector = build_audio_projector(audio_proj_cfg)
            if config.use_temporal_bridge:
                n_heads = max(1, llm_hidden_size // 128)
                self.temporal_bridge = AudioVisualTemporalBridge(d_model=llm_hidden_size, n_heads=n_heads)

        if config.use_lora and not config.freeze_llm:
            self._apply_lora(config)

        llm_dtype = self.llm.get_input_embeddings().weight.dtype
        self.vision_projector = self.vision_projector.to(dtype=llm_dtype)
        if self.audio_projector is not None:
            self.audio_projector = self.audio_projector.to(dtype=llm_dtype)
        if self.temporal_bridge is not None:
            self.temporal_bridge = self.temporal_bridge.to(dtype=llm_dtype)

        total_params = count_parameters(self, trainable_only=False)
        trainable_params = count_parameters(self, trainable_only=True)
        logger.info(
            "VideoLLM initialized: total=%s trainable=%s",
            format_param_count(total_params),
            format_param_count(trainable_params),
        )

    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video frames to LLM-space visual tokens.

        Args:
            pixel_values: Video tensor of shape ``(B, T, C, H, W)``.

        Returns:
            Visual tokens of shape ``(B, L_v, D_llm)``.

        Raises:
            TypeError: If input type is invalid.
            ValueError: If input shape is invalid.
        """
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be torch.Tensor, got {type(pixel_values)!r}")
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must have shape (B, T, C, H, W), got {tuple(pixel_values.shape)}")
        if pixel_values.shape[1] <= 0:
            raise ValueError("pixel_values has zero frames")

        bsz, num_frames, channels, height, width = pixel_values.shape  # (B, T, C, H, W)
        if channels != 3:
            raise ValueError(f"pixel_values channel dimension must be 3, got {channels}")

        flat_frames = pixel_values.reshape(bsz * num_frames, channels, height, width)  # (B*T, C, H, W)
        with torch.no_grad():
            frame_features = self.vision_tower(flat_frames)  # (B*T, N, D_v)

        num_patches = frame_features.shape[1]
        vision_dim = frame_features.shape[2]
        grid_features = frame_features.reshape(bsz, num_frames, num_patches, vision_dim)  # (B, T, N, D_v)

        projector_dtype = next(self.vision_projector.parameters()).dtype
        grid_features = grid_features.to(dtype=projector_dtype)

        if self.config.mm_projector_type == "stc":
            if not isinstance(self.vision_projector, STCConnector):
                raise TypeError("vision_projector is expected to be STCConnector when mm_projector_type='stc'")
            grid_side = int(num_patches**0.5)
            if grid_side * grid_side != num_patches:
                raise ValueError(f"num_patches {num_patches} is not a perfect square for STC grid reshape")
            grid_5d = grid_features.reshape(bsz, num_frames, grid_side, grid_side, vision_dim)  # (B, T, H, W, D_v)
            visual_tokens = self.vision_projector(grid_5d)  # (B, L_v, D_llm)
        else:
            token_features = grid_features.reshape(bsz, num_frames * num_patches, vision_dim)  # (B, T*N, D_v)
            visual_tokens = self.vision_projector(token_features)  # (B, L_v, D_llm)
        return visual_tokens

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode raw waveforms to LLM-space audio tokens.

        Args:
            waveforms: Audio tensor of shape ``(B, samples)``.

        Returns:
            Audio tokens of shape ``(B, T_a, D_llm)``.

        Raises:
            RuntimeError: If audio branch is disabled.
            ValueError: If waveform shape is invalid.
        """
        if self.audio_tower is None or self.audio_projector is None:
            raise RuntimeError("Audio branch is disabled. Set use_audio=True in ModelConfig.")
        if waveforms.ndim != 2:
            raise ValueError(f"waveforms must have shape (B, samples), got {tuple(waveforms.shape)}")

        with torch.no_grad():
            audio_features = self.audio_tower(waveforms)  # (B, T_a, D_a)

        projector_dtype = next(self.audio_projector.parameters()).dtype
        audio_features = audio_features.to(dtype=projector_dtype)
        audio_tokens = self.audio_projector(audio_features)  # (B, T_a, D_llm)
        return audio_tokens

    def prepare_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Replace placeholder modal indices with encoded modal embeddings.

        Args:
            input_ids: Token ids with modal placeholders, shape ``(B, S)``.
            attention_mask: Mask tensor, shape ``(B, S)``.
            labels: Optional labels tensor, shape ``(B, S)``.
            pixel_values: Optional video tensor, shape ``(B, T, C, H, W)``.
            waveforms: Optional audio tensor, shape ``(B, samples)``.

        Returns:
            Tuple ``(inputs_embeds, new_attention_mask, new_labels)`` with variable-length
            replacement handled and padded to the batch max sequence length.
        """
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2D tensors")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids and attention_mask shape mismatch: {tuple(input_ids.shape)} vs {tuple(attention_mask.shape)}"
            )
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(
                f"labels shape must match input_ids shape, got {tuple(labels.shape)} vs {tuple(input_ids.shape)}"
            )

        modal_indices = set(MODAL_INDEX_MAP.values())
        visual_tokens: torch.Tensor | None = None
        audio_tokens: torch.Tensor | None = None
        if pixel_values is not None:
            visual_tokens = self.encode_video(pixel_values)  # (B, L_v, D)
        if waveforms is not None and self.audio_tower is not None:
            audio_tokens = self.encode_audio(waveforms)  # (B, L_a, D)

        if visual_tokens is not None and audio_tokens is not None and self.temporal_bridge is not None:
            fused_tokens = self.temporal_bridge(visual_tokens, audio_tokens)  # (B, L_v+L_a, D)
            visual_length = visual_tokens.shape[1]
            visual_tokens = fused_tokens[:, :visual_length, :]  # (B, L_v, D)
            audio_tokens = fused_tokens[:, visual_length:, :]  # (B, L_a, D)

        embedding_layer = self.llm.get_input_embeddings()
        embedding_dtype = embedding_layer.weight.dtype
        embedding_device = embedding_layer.weight.device

        batch_embeds: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []

        batch_size, seq_len = input_ids.shape
        for batch_idx in range(batch_size):
            sample_ids = input_ids[batch_idx]  # (S,)
            sample_mask = attention_mask[batch_idx]  # (S,)
            sample_labels = labels[batch_idx] if labels is not None else None

            safe_ids = sample_ids.clone()
            safe_ids[safe_ids < 0] = 0
            text_embeds = embedding_layer(safe_ids.to(device=embedding_device)).to(dtype=embedding_dtype)  # (S, D)

            embed_segments: list[torch.Tensor] = []
            mask_segments: list[torch.Tensor] = []
            label_segments: list[torch.Tensor] = []

            for position in range(seq_len):
                token_id = int(sample_ids[position].item())
                if token_id in modal_indices:
                    if token_id == VIDEO_TOKEN_INDEX and visual_tokens is not None:
                        insert_visual = visual_tokens[batch_idx].to(
                            device=embedding_device, dtype=embedding_dtype
                        )  # (L_v, D)
                        embed_segments.append(insert_visual)
                        mask_segments.append(
                            torch.ones(insert_visual.shape[0], dtype=sample_mask.dtype, device=sample_mask.device)
                        )
                        if sample_labels is not None:
                            label_segments.append(
                                torch.full(
                                    (insert_visual.shape[0],),
                                    IGNORE_INDEX,
                                    dtype=sample_labels.dtype,
                                    device=sample_labels.device,
                                )
                            )
                        continue

                    if token_id == AUDIO_TOKEN_INDEX and audio_tokens is not None:
                        insert_audio = audio_tokens[batch_idx].to(
                            device=embedding_device, dtype=embedding_dtype
                        )  # (L_a, D)
                        embed_segments.append(insert_audio)
                        mask_segments.append(
                            torch.ones(insert_audio.shape[0], dtype=sample_mask.dtype, device=sample_mask.device)
                        )
                        if sample_labels is not None:
                            label_segments.append(
                                torch.full(
                                    (insert_audio.shape[0],),
                                    IGNORE_INDEX,
                                    dtype=sample_labels.dtype,
                                    device=sample_labels.device,
                                )
                            )
                        continue

                    # Modal token with no active encoder — skip entirely (no embed, no label)
                    continue

                embed_segments.append(text_embeds[position : position + 1, :])  # (1, D)
                mask_segments.append(sample_mask[position : position + 1])  # (1,)
                if sample_labels is not None:
                    label_segments.append(sample_labels[position : position + 1])  # (1,)

            if not embed_segments:
                raise ValueError("All tokens were filtered out while preparing multimodal inputs")

            sample_embeds = torch.cat(embed_segments, dim=0)  # (S_new, D)
            sample_new_mask = torch.cat(mask_segments, dim=0)  # (S_new,)
            batch_embeds.append(sample_embeds)
            batch_masks.append(sample_new_mask)
            if sample_labels is not None:
                batch_labels.append(torch.cat(label_segments, dim=0))  # (S_new,)

        max_length = max(sample_embed.shape[0] for sample_embed in batch_embeds)
        hidden_size = batch_embeds[0].shape[1]

        padded_embeds = torch.zeros(
            batch_size,
            max_length,
            hidden_size,
            dtype=batch_embeds[0].dtype,
            device=batch_embeds[0].device,
        )
        padded_mask = torch.zeros(batch_size, max_length, dtype=batch_masks[0].dtype, device=batch_masks[0].device)
        padded_labels: torch.Tensor | None = None

        if labels is not None:
            padded_labels = torch.full(
                (batch_size, max_length),
                IGNORE_INDEX,
                dtype=batch_labels[0].dtype,
                device=batch_labels[0].device,
            )

        for batch_idx in range(batch_size):
            current_length = batch_embeds[batch_idx].shape[0]
            padded_embeds[batch_idx, :current_length, :] = batch_embeds[batch_idx]
            padded_mask[batch_idx, :current_length] = batch_masks[batch_idx]
            if padded_labels is not None:
                padded_labels[batch_idx, :current_length] = batch_labels[batch_idx]

        return padded_embeds, padded_mask, padded_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass for language modeling with optional video/audio inputs.

        Args:
            input_ids: Token ids, shape ``(B, S)``.
            attention_mask: Attention mask, shape ``(B, S)``.
            pixel_values: Optional video frames, shape ``(B, T, C, H, W)``.
            waveforms: Optional waveforms, shape ``(B, samples)``.
            labels: Optional labels, shape ``(B, S)``.

        Returns:
            Causal language modeling output.
        """
        if pixel_values is None and waveforms is None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        inputs_embeds, new_attention_mask, new_labels = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            waveforms=waveforms,
        )

        if not torch.isfinite(inputs_embeds).all():
            logger.warning("Non-finite values in inputs_embeds — replacing with zeros")
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            return_dict=True,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text from text-only or audio-visual prompts.

        Args:
            input_ids: Prompt token ids, shape ``(B, S)``.
            attention_mask: Prompt mask, shape ``(B, S)``.
            pixel_values: Optional video tensor, shape ``(B, T, C, H, W)``.
            waveforms: Optional audio tensor, shape ``(B, samples)``.
            max_new_tokens: Maximum number of new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Generated token ids, shape ``(B, S_out)``.
        """
        do_sample = temperature > 0.0
        with torch.no_grad():
            if pixel_values is None and waveforms is None:
                return self.llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )

            inputs_embeds, new_attention_mask, _ = self.prepare_multimodal_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                pixel_values=pixel_values,
                waveforms=waveforms,
            )
            return self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object] | None = None) -> None:
        """Enable gradient checkpointing on the LLM backbone."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the LLM backbone."""
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()

    def _apply_lora(self, config: ModelConfig) -> None:
        """Apply LoRA adapters to the LLM backbone.

        Args:
            config: Model config with LoRA hyperparameters.
        """
        from peft import LoraConfig, get_peft_model

        if config.load_in_4bit:
            from peft import prepare_model_for_kbit_training
            self.llm = prepare_model_for_kbit_training(self.llm)
            logger.info("Prepared LLM for k-bit training (gradient checkpointing enabled)")

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(_LORA_TARGET_MODULES),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        logger.info("Applied LoRA adapters (r=%d, alpha=%d)", config.lora_r, config.lora_alpha)
