"""Debug STC connector layer by layer."""

import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from videollm.model.projector import ProjectorConfig, STCConnector  # noqa: E402


def main() -> None:
    """Debug STC forward pass step by step."""
    device = torch.device("cuda:0")

    config = ProjectorConfig(mm_hidden_size=1152, hidden_size=3584, mm_hidden_size_a=1024)
    stc = STCConnector(config).to(device=device, dtype=torch.float32)

    logger.info("=== STCConnector Layer Weights ===")
    for name, param in stc.named_parameters():
        logger.info(
            "  %s: shape=%s, mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
            name,
            tuple(param.shape),
            param.data.mean().item(),
            param.data.std().item(),
            param.data.min().item(),
            param.data.max().item(),
        )

    # Simulate SigLIP output
    bsz, t_steps, h, w, d_v = 1, 16, 27, 27, 1152
    # Use realistic SigLIP output stats
    x = torch.randn(bsz, t_steps, h, w, d_v, device=device, dtype=torch.float32) * 2.67
    logger.info(
        "\n=== Input: mean=%.4f, std=%.4f, min=%.4f, max=%.4f ===",
        x.mean().item(),
        x.std().item(),
        x.min().item(),
        x.max().item(),
    )

    # Step through manually
    channels = d_v

    # Step 1: reshape + s1
    x2d = x.permute(0, 1, 4, 2, 3).reshape(bsz * t_steps, channels, h, w)
    logger.info(
        "\nBefore s1: shape=%s, mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        x2d.shape,
        x2d.mean().item(),
        x2d.std().item(),
        x2d.min().item(),
        x2d.max().item(),
    )

    with torch.no_grad():
        x2d_s1 = stc.s1(x2d)
    logger.info(
        "After s1: shape=%s, mean=%.4f, std=%.4f, min=%.4f, max=%.4f, has_nan=%s, has_inf=%s",
        x2d_s1.shape,
        x2d_s1.mean().item(),
        x2d_s1.std().item(),
        x2d_s1.min().item(),
        x2d_s1.max().item(),
        torch.isnan(x2d_s1).any().item(),
        torch.isinf(x2d_s1).any().item(),
    )

    # Step 2: reshape + sampler (Conv3d)
    out_chs = x2d_s1.shape[1]
    x3d = x2d_s1.reshape(bsz, t_steps, out_chs, h, w).permute(0, 2, 1, 3, 4)
    logger.info("\nBefore sampler: shape=%s, mean=%.4f, std=%.4f", x3d.shape, x3d.mean().item(), x3d.std().item())

    with torch.no_grad():
        x3d_ds = stc.sampler(x3d)
    logger.info(
        "After sampler: shape=%s, mean=%.4f, std=%.4f, has_nan=%s, has_inf=%s",
        x3d_ds.shape,
        x3d_ds.mean().item(),
        x3d_ds.std().item(),
        torch.isnan(x3d_ds).any().item(),
        torch.isinf(x3d_ds).any().item(),
    )

    # Step 3: s2
    _, channels_ds, t_ds, h_ds, w_ds = x3d_ds.shape
    x2d_ds = x3d_ds.permute(0, 2, 1, 3, 4).reshape(bsz * t_ds, channels_ds, h_ds, w_ds)
    logger.info("\nBefore s2: shape=%s, mean=%.4f, std=%.4f", x2d_ds.shape, x2d_ds.mean().item(), x2d_ds.std().item())

    with torch.no_grad():
        x2d_s2 = stc.s2(x2d_ds)
    logger.info(
        "After s2: shape=%s, mean=%.4f, std=%.4f, has_nan=%s, has_inf=%s",
        x2d_s2.shape,
        x2d_s2.mean().item(),
        x2d_s2.std().item(),
        torch.isnan(x2d_s2).any().item(),
        torch.isinf(x2d_s2).any().item(),
    )

    # Step 4: readout
    x_tokens = x2d_s2.reshape(bsz, t_ds, channels_ds, h_ds, w_ds)
    x_tokens = x_tokens.permute(0, 1, 3, 4, 2).reshape(bsz, t_ds * h_ds * w_ds, channels_ds)
    logger.info(
        "\nBefore readout: shape=%s, mean=%.4f, std=%.4f", x_tokens.shape, x_tokens.mean().item(), x_tokens.std().item()
    )

    with torch.no_grad():
        x_readout = stc.readout(x_tokens)
    logger.info(
        "After readout: shape=%s, mean=%.4f, std=%.4f, has_nan=%s, has_inf=%s",
        x_readout.shape,
        x_readout.mean().item(),
        x_readout.std().item(),
        torch.isnan(x_readout).any().item(),
        torch.isinf(x_readout).any().item(),
    )

    logger.info("output_scale = %.6f", stc.output_scale.item())
    final = x_readout * stc.output_scale
    logger.info(
        "Final output: mean=%.4f, std=%.4f, has_nan=%s, has_inf=%s",
        final.mean().item(),
        final.std().item(),
        torch.isnan(final).any().item(),
        torch.isinf(final).any().item(),
    )

    # Also test with REAL SigLIP output
    logger.info("\n\n=== Testing with real SigLIP output ===")
    from videollm.model.encoder import build_vision_tower

    vt = build_vision_tower("google/siglip-so400m-patch14-384", freeze=True)
    vt = vt.to(device=device)

    dummy_frames = torch.randn(1, 3, 384, 384, device=device) * 0.5 + 0.5
    with torch.no_grad():
        real_features = vt(dummy_frames)  # (1, 729, 1152)
    logger.info(
        "Real SigLIP output: shape=%s, mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        real_features.shape,
        real_features.mean().item(),
        real_features.std().item(),
        real_features.min().item(),
        real_features.max().item(),
    )

    # Reshape for STC
    real_grid = real_features.reshape(1, 1, 27, 27, 1152)  # single frame
    real_grid_16 = real_grid.expand(1, 16, 27, 27, 1152).contiguous()

    with torch.no_grad():
        real_output = stc(real_grid_16.to(dtype=torch.float32))
    logger.info(
        "STC output from real SigLIP: shape=%s, mean=%.6f, std=%.6f, has_nan=%s, has_inf=%s",
        real_output.shape,
        real_output.mean().item(),
        real_output.std().item(),
        torch.isnan(real_output).any().item(),
        torch.isinf(real_output).any().item(),
    )


if __name__ == "__main__":
    main()
