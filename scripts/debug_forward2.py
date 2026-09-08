"""Debug forward pass - focus on vision pipeline step by step."""

import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from videollm.data.dataset import VideoAudioDataset  # noqa: E402
from videollm.model.videollm import ModelConfig, VideoLLM  # noqa: E402


def main() -> None:
    """Debug the vision encoding pipeline."""
    device = torch.device("cuda:0")

    logger.info("=== Building model ===")
    config = ModelConfig(
        mm_projector_type="stc",
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=True,
        use_lora=False,
        use_audio=True,
        use_temporal_bridge=False,
    )
    model = VideoLLM(config)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = transformers.AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    dataset = VideoAudioDataset(
        data_path="data/alignment/audiocaps_train.jsonl",
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_frames=16,
        max_length=2048,
        video_dir="data/videos",
    )

    sample = dataset[0]
    pixel_values = sample["pixel_values"]  # (T, C, H, W)
    logger.info("pixel_values: shape=%s, dtype=%s", pixel_values.shape, pixel_values.dtype)
    logger.info(
        "  mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        pixel_values.mean().item(),
        pixel_values.std().item(),
        pixel_values.min().item(),
        pixel_values.max().item(),
    )
    logger.info(
        "  has_nan=%s, has_inf=%s", torch.isnan(pixel_values).any().item(), torch.isinf(pixel_values).any().item()
    )

    # Step 1: Through SigLIP
    pixel_values = pixel_values.unsqueeze(0).to(device)  # (1, T, C, H, W)
    bsz, num_frames, channels, height, width = pixel_values.shape
    flat_frames = pixel_values.reshape(bsz * num_frames, channels, height, width)
    logger.info("\nflat_frames: shape=%s", flat_frames.shape)
    logger.info(
        "  has_nan=%s, has_inf=%s", torch.isnan(flat_frames).any().item(), torch.isinf(flat_frames).any().item()
    )

    with torch.no_grad():
        frame_features = model.vision_tower(flat_frames)  # (B*T, N, D_v)
    logger.info("\nSigLIP output: shape=%s", frame_features.shape)
    logger.info(
        "  mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        frame_features.mean().item(),
        frame_features.std().item(),
        frame_features.min().item(),
        frame_features.max().item(),
    )
    logger.info(
        "  has_nan=%s, has_inf=%s", torch.isnan(frame_features).any().item(), torch.isinf(frame_features).any().item()
    )

    # Check per-frame
    for i in range(min(num_frames, 4)):
        ff = frame_features[i]
        logger.info(
            "  frame %d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, nan=%s",
            i,
            ff.mean().item(),
            ff.std().item(),
            ff.min().item(),
            ff.max().item(),
            torch.isnan(ff).any().item(),
        )

    if torch.isnan(frame_features).any():
        logger.info("\n*** NaN detected in SigLIP output! ***")
        # Find which frame(s) have NaN
        for i in range(num_frames):
            if torch.isnan(frame_features[i]).any():
                logger.info("  Frame %d has NaN", i)
                # Check input frame
                inp_frame = flat_frames[i]
                logger.info(
                    "    Input frame %d: mean=%.4f, std=%.4f, nan=%s, inf=%s",
                    i,
                    inp_frame.mean().item(),
                    inp_frame.std().item(),
                    torch.isnan(inp_frame).any().item(),
                    torch.isinf(inp_frame).any().item(),
                )
        return

    # Step 2: Reshape for STC
    num_patches = frame_features.shape[1]
    vision_dim = frame_features.shape[2]
    grid_features = frame_features.reshape(bsz, num_frames, num_patches, vision_dim)
    grid_side = int(num_patches**0.5)
    grid_5d = grid_features.reshape(bsz, num_frames, grid_side, grid_side, vision_dim)
    logger.info("\ngrid_5d for STC: shape=%s", grid_5d.shape)
    logger.info(
        "  mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        grid_5d.mean().item(),
        grid_5d.std().item(),
        grid_5d.min().item(),
        grid_5d.max().item(),
    )

    # Step 3: Through STC
    proj_dtype = next(model.vision_projector.parameters()).dtype
    grid_5d = grid_5d.to(dtype=proj_dtype)

    with torch.no_grad():
        visual_tokens = model.vision_projector(grid_5d)
    logger.info("\nSTC output: shape=%s", visual_tokens.shape)
    logger.info(
        "  mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        visual_tokens.mean().item(),
        visual_tokens.std().item(),
        visual_tokens.min().item(),
        visual_tokens.max().item(),
    )
    logger.info(
        "  has_nan=%s, has_inf=%s", torch.isnan(visual_tokens).any().item(), torch.isinf(visual_tokens).any().item()
    )

    # Audio
    waveforms = sample["waveforms"].unsqueeze(0).to(device)
    logger.info(
        "\nwaveforms: shape=%s, mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        waveforms.shape,
        waveforms.mean().item(),
        waveforms.std().item(),
        waveforms.min().item(),
        waveforms.max().item(),
    )

    with torch.no_grad():
        audio_tokens = model.encode_audio(waveforms)
    logger.info(
        "audio_tokens: shape=%s, mean=%.6f, std=%.6f, nan=%s",
        audio_tokens.shape,
        audio_tokens.mean().item(),
        audio_tokens.std().item(),
        torch.isnan(audio_tokens).any().item(),
    )


if __name__ == "__main__":
    main()
