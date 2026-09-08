"""Vision encoder towers for VideoLLM."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from videollm.data.constants import DEFAULT_VISION_ENCODER

logger = logging.getLogger(__name__)


class VisionTower(nn.Module):
    """Wrapper around a HuggingFace vision encoder (SigLIP or CLIP).

    Extracts patch-level hidden states suitable for downstream projection.

    Args:
        model_name: HuggingFace model identifier.
        freeze: Whether to freeze encoder weights.
        select_layer: Which hidden layer to extract (-1 = last, -2 = second-to-last).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VISION_ENCODER,
        freeze: bool = True,
        select_layer: int = -2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.select_layer = select_layer

        self._model: nn.Module
        self._processor: object
        self._config: CLIPVisionConfig | SiglipVisionConfig
        self._has_cls_token: bool = False

        self._load_model(model_name)

        if freeze:
            self.requires_grad_(False)
            logger.info("Vision tower frozen: %s", model_name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        """Dimension of the encoder hidden states."""
        return int(self._config.hidden_size)

    @property
    def image_size(self) -> int:
        """Expected input image resolution."""
        return int(self._config.image_size)

    @property
    def patch_size(self) -> int:
        """Patch size used by the vision transformer."""
        return int(self._config.patch_size)

    @property
    def num_patches(self) -> int:
        """Number of spatial patches per image (excluding CLS)."""
        side = self.image_size // self.patch_size
        return side * side

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype of the vision model."""
        return next(self._model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Device of the vision model."""
        return next(self._model.parameters()).device

    @property
    def processor(self) -> object:
        """The image processor associated with this tower."""
        return self._processor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch-level features.

        Args:
            images: Preprocessed pixel values of shape ``(B, C, H, W)``.

        Returns:
            Patch features of shape ``(B, N, D)`` where *N* = ``num_patches``.
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be torch.Tensor, got {type(images)!r}")
        if images.ndim != 4:
            raise ValueError(f"images must have shape (B, C, H, W), got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"images channel dimension must be 3, got {images.shape[1]}")

        outputs = self._model(
            pixel_values=images.to(dtype=self.dtype, device=self.device),  # (B, C, H, W)
            output_hidden_states=True,
        )
        all_hidden_states = outputs.hidden_states
        if all_hidden_states is None:
            raise ValueError("Vision model did not return hidden states")
        if self.select_layer >= len(all_hidden_states) or self.select_layer < -len(all_hidden_states):
            raise ValueError(
                f"select_layer {self.select_layer} is out of range for {len(all_hidden_states)} hidden states"
            )
        hidden_states = all_hidden_states[self.select_layer]  # (B, N+1, D) or (B, N, D)

        # SigLIP has no CLS token; CLIP has a CLS token at position 0
        if self._has_cls_token:
            hidden_states = hidden_states[:, 1:, :]  # (B, N, D)

        return hidden_states  # (B, N, D)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load_model(self, model_name: str) -> None:
        """Load the vision model and processor from HuggingFace."""
        is_siglip = "siglip" in model_name.lower()
        is_clip = "clip" in model_name.lower()

        if not is_siglip and not is_clip:
            raise ValueError(f"Unsupported vision model '{model_name}'. Expected a SigLIP or CLIP checkpoint.")

        if is_siglip:
            self._model = SiglipVisionModel.from_pretrained(model_name)
            self._processor = SiglipImageProcessor.from_pretrained(model_name)
            self._config = self._model.config
            self._has_cls_token = False
        else:
            self._model = CLIPVisionModel.from_pretrained(model_name)
            self._processor = CLIPImageProcessor.from_pretrained(model_name)
            self._config = self._model.config
            self._has_cls_token = True

        logger.info(
            "Loaded vision tower: %s  hidden=%d  patches=%d  image=%d",
            model_name,
            self.hidden_size,
            self.num_patches,
            self.image_size,
        )


def build_vision_tower(
    model_name: str = DEFAULT_VISION_ENCODER,
    freeze: bool = True,
    select_layer: int = -2,
) -> VisionTower:
    """Factory for building a vision tower.

    Args:
        model_name: HuggingFace model identifier (SigLIP or CLIP).
        freeze: Freeze encoder parameters.
        select_layer: Hidden layer index to extract features from.

    Returns:
        Initialised ``VisionTower`` instance.
    """
    return VisionTower(model_name=model_name, freeze=freeze, select_layer=select_layer)
