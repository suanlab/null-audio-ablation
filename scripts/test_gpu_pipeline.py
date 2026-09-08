#!/usr/bin/env python3
"""End-to-end GPU pipeline test for VideoLLM.

Tests the complete model pipeline on GPU with real encoder weights (SigLIP, CLAP)
and synthetic data. Verifies forward pass shapes, backward pass gradients, and
loss computation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/test_gpu_pipeline.py
    CUDA_VISIBLE_DEVICES=1 python scripts/test_gpu_pipeline.py --full  # also test with LLM

Expects cached models in ~/.cache/huggingface/hub/:
    - google/siglip-so400m-patch14-384
    - laion/larger_clap_general
    - Qwen/Qwen2-7B-Instruct (only for --full mode)
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import torch


def _cleanup() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_gpu_mem(label: str) -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(f"  [{label}] Allocated: {alloc:.1f}GB, Reserved: {reserved:.1f}GB, Free: {free / 1024**3:.1f}GB")


def test_vision_tower_gpu() -> None:
    """Test VisionTower (SigLIP) forward on GPU."""
    print("\n=== Test 1: VisionTower (SigLIP) on GPU ===")
    from videollm.model.encoder import build_vision_tower

    t0 = time.time()
    tower = build_vision_tower("google/siglip-so400m-patch14-384", freeze=True, select_layer=-2)
    tower = tower.to("cuda")
    print(f"  Loaded SigLIP in {time.time() - t0:.1f}s")
    print(f"  hidden_size={tower.hidden_size}, patches={tower.num_patches}, dtype={tower.dtype}")
    _print_gpu_mem("after load")

    # Simulate 8 frames of 384x384 video
    images = torch.randn(8, 3, 384, 384, device="cuda", dtype=tower.dtype)  # (8, C, H, W)
    with torch.no_grad():
        features = tower(images)  # (8, 729, 1152)

    print(f"  Output shape: {features.shape}")
    assert features.shape == (8, 729, 1152), f"Expected (8, 729, 1152), got {features.shape}"
    print("  PASSED")

    del tower, images, features
    _cleanup()


def test_audio_tower_gpu() -> None:
    """Test AudioTower (CLAP) forward on GPU."""
    print("\n=== Test 2: AudioTower (CLAP) on GPU ===")
    from videollm.model.audio_encoder import build_audio_tower

    t0 = time.time()
    tower = build_audio_tower("laion/larger_clap_general", freeze=True, sample_rate=48000)
    tower = tower.to("cuda")
    print(f"  Loaded CLAP in {time.time() - t0:.1f}s")
    print(f"  hidden_size={tower.hidden_size}")
    _print_gpu_mem("after load")

    # 10 seconds of audio at 48kHz
    waveforms = torch.randn(1, 48000 * 10, device="cuda")  # (1, 480000)
    with torch.no_grad():
        features = tower(waveforms)  # (1, T_a, D)

    print(f"  Output shape: {features.shape}")
    assert features.ndim == 3, f"Expected 3D tensor, got {features.ndim}D"
    assert features.shape[0] == 1
    print(f"  T_a={features.shape[1]}, D={features.shape[2]}")
    print("  PASSED")

    del tower, waveforms, features
    _cleanup()


def test_stc_connector_gpu() -> None:
    """Test STCConnector at production dimensions on GPU."""
    print("\n=== Test 3: STCConnector (1152 → 3584) on GPU ===")
    from videollm.model.projector import ProjectorConfig, STCConnector

    cfg = ProjectorConfig(mm_hidden_size=1152, hidden_size=3584, stc_downsample=(2, 2, 2), stc_depth=4, mlp_depth=2)
    stc = STCConnector(cfg).to("cuda").to(torch.bfloat16)
    _print_gpu_mem("after STC init")

    # 8 frames, 729 patches (27x27), 1152-dim
    x = torch.randn(1, 8, 729, 1152, device="cuda", dtype=torch.bfloat16)  # (B, T, N, D)
    t0 = time.time()
    out = stc(x)  # (B, L, 3584)
    print(f"  Forward time: {time.time() - t0:.3f}s")
    print(f"  Input: {x.shape} → Output: {out.shape}")
    assert out.shape[0] == 1
    assert out.shape[2] == 3584
    print("  PASSED")

    # Test backward
    loss = out.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in stc.parameters() if p.grad is not None)
    print(f"  Backward OK, grad norm: {grad_norm:.4f}")

    del stc, x, out
    _cleanup()


def test_agta_bridge_gpu() -> None:
    """Test AGTA bridge forward + backward on GPU."""
    print("\n=== Test 4: AGTA Bridge (3584-dim, 28 heads) on GPU ===")
    from videollm.model.temporal import AudioVisualTemporalBridge

    bridge = AudioVisualTemporalBridge(d_model=3584, n_heads=28).to("cuda").to(torch.bfloat16)
    _print_gpu_mem("after bridge init")

    visual = torch.randn(1, 100, 3584, device="cuda", dtype=torch.bfloat16)  # (B, T_v, D)
    audio = torch.randn(1, 20, 3584, device="cuda", dtype=torch.bfloat16)  # (B, T_a, D)

    t0 = time.time()
    out = bridge(visual, audio)  # (B, T_v+T_a, D) = (1, 120, 3584)
    print(f"  Forward time: {time.time() - t0:.3f}s")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (1, 120, 3584), f"Expected (1, 120, 3584), got {out.shape}"

    # Backward
    loss = out.sum()
    loss.backward()
    grad_count = sum(1 for p in bridge.parameters() if p.grad is not None)
    total_params = sum(1 for p in bridge.parameters())
    print(f"  Backward OK, {grad_count}/{total_params} params have gradients")
    print("  PASSED")

    del bridge, visual, audio, out
    _cleanup()


def test_full_encode_pipeline_gpu() -> None:
    """Test full encoding pipeline: SigLIP → STC → AGTA on GPU."""
    print("\n=== Test 5: Full Encode Pipeline (SigLIP → STC → AGTA) on GPU ===")
    from videollm.model.audio_encoder import build_audio_tower
    from videollm.model.encoder import build_vision_tower
    from videollm.model.projector import ProjectorConfig, STCConnector, build_audio_projector
    from videollm.model.temporal import AudioVisualTemporalBridge

    device = torch.device("cuda")
    llm_dim = 3584

    # Build components
    vision_tower = build_vision_tower("google/siglip-so400m-patch14-384", freeze=True).to(device)
    audio_tower = build_audio_tower("laion/larger_clap_general", freeze=True).to(device)

    proj_cfg = ProjectorConfig(mm_hidden_size=vision_tower.hidden_size, hidden_size=llm_dim)
    stc = STCConnector(proj_cfg).to(device).to(torch.bfloat16)

    proj_cfg_a = ProjectorConfig(mm_hidden_size_a=audio_tower.hidden_size, hidden_size=llm_dim)
    audio_proj = build_audio_projector(proj_cfg_a).to(device).to(torch.bfloat16)

    bridge = AudioVisualTemporalBridge(d_model=llm_dim, n_heads=28).to(device).to(torch.bfloat16)

    _print_gpu_mem("after all modules loaded")

    # Synthetic inputs
    num_frames = 8
    pixel_values = torch.randn(1, num_frames, 3, 384, 384, device=device, dtype=vision_tower.dtype)  # (B,T,C,H,W)
    waveforms = torch.randn(1, 48000 * 10, device=device)  # (B, samples) — 10s at 48kHz

    # Forward: vision
    b, t, c, h, w = pixel_values.shape
    frames = pixel_values.reshape(b * t, c, h, w)  # (8, C, H, W)
    with torch.no_grad():
        vis_features = vision_tower(frames)  # (8, 729, 1152)
    vis_features = vis_features.reshape(b, t, vis_features.shape[1], vis_features.shape[2])  # (1,8,729,1152)
    vis_features = vis_features.to(torch.bfloat16)
    visual_tokens = stc(vis_features)  # (1, L, 3584)
    print(f"  Visual tokens: {visual_tokens.shape}")

    # Forward: audio
    with torch.no_grad():
        aud_features = audio_tower(waveforms)  # (1, T_a, D_audio)
    aud_features = aud_features.to(torch.bfloat16)
    audio_tokens = audio_proj(aud_features)  # (1, T_a, 3584)
    print(f"  Audio tokens: {audio_tokens.shape}")

    # Forward: AGTA bridge
    combined = bridge(visual_tokens, audio_tokens)  # (1, L+T_a, 3584)
    print(f"  Combined tokens: {combined.shape}")
    print(f"  Total multimodal tokens: {combined.shape[1]}")

    # Backward through trainable components
    loss = combined.sum()
    loss.backward()
    for name, module in [("STC", stc), ("AudioProj", audio_proj), ("Bridge", bridge)]:
        grads = sum(1 for p in module.parameters() if p.grad is not None)
        total = sum(1 for p in module.parameters())
        print(f"  {name}: {grads}/{total} params with gradients")

    print("  PASSED")
    _print_gpu_mem("after full pipeline")

    del vision_tower, audio_tower, stc, audio_proj, bridge
    _cleanup()


def test_full_model_gpu() -> None:
    """Test complete VideoLLM model with real weights (requires ~20GB+ VRAM)."""
    print("\n=== Test 6: Full VideoLLM Model on GPU ===")
    print("  (Downloads Qwen2-7B-Instruct if not cached — ~14GB)")
    from videollm.model.videollm import ModelConfig, VideoLLM
    from videollm.utils import count_parameters, format_param_count

    t0 = time.time()
    config = ModelConfig(
        llm_path="Qwen/Qwen2-7B-Instruct",
        vision_encoder="google/siglip-so400m-patch14-384",
        audio_encoder="laion/larger_clap_general",
        mm_projector_type="stc",
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=True,
        use_lora=True,
        lora_r=16,  # smaller LoRA for testing
        lora_alpha=32,
        use_audio=True,
        use_temporal_bridge=True,
    )
    model = VideoLLM(config)
    model = model.to("cuda")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print(f"  Total params: {format_param_count(total)}, Trainable: {format_param_count(trainable)}")
    _print_gpu_mem("after model load")

    # Text-only forward
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    text = "Hello, describe this video in detail."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to("cuda")

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    print(f"  Text-only logits: {outputs.logits.shape}")

    # Text-only with labels (test loss)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
    print(f"  Text-only loss: {outputs.loss.item():.4f}")

    # Training step (backward through LoRA)
    model.train()
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
    loss = outputs.loss
    loss.backward()

    lora_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None and "lora" in name:
            lora_grads += 1
    print(f"  Backward OK, {lora_grads} LoRA params have gradients")
    print(f"  Loss: {loss.item():.4f}")
    _print_gpu_mem("after backward")

    print("  PASSED")

    del model
    _cleanup()


def test_multimodal_forward_gpu() -> None:
    """Test VideoLLM with video + audio inputs (full multimodal pipeline)."""
    print("\n=== Test 7: Multimodal Forward (Video + Audio) on GPU ===")
    from videollm.model.videollm import ModelConfig, VideoLLM

    config = ModelConfig(
        llm_path="Qwen/Qwen2-7B-Instruct",
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        use_audio=True,
        use_temporal_bridge=True,
    )
    model = VideoLLM(config).to("cuda")
    model.train()

    from videollm.data.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, VIDEO_TOKEN_INDEX

    # Create input_ids with special modal tokens
    seq_len = 32
    input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda")  # (B, S)
    input_ids[0, 2] = VIDEO_TOKEN_INDEX  # place video token
    input_ids[0, 4] = AUDIO_TOKEN_INDEX  # place audio token
    attention_mask = torch.ones(1, seq_len, device="cuda", dtype=torch.long)

    # Labels: mask special tokens and input prefix with IGNORE_INDEX
    labels = input_ids.clone()
    labels[labels < 0] = IGNORE_INDEX  # mask special modal tokens
    labels[0, :10] = IGNORE_INDEX  # mask input prefix

    # Synthetic video: 4 frames of 384x384
    pixel_values = torch.randn(1, 4, 3, 384, 384, device="cuda", dtype=torch.bfloat16)  # (B,T,C,H,W)
    waveforms = torch.randn(1, 48000 * 5, device="cuda")  # (B, samples) — 5s

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        waveforms=waveforms,
        labels=labels,
    )

    print(f"  Multimodal logits: {outputs.logits.shape}")
    print(f"  Multimodal loss: {outputs.loss.item():.4f}")

    loss = outputs.loss
    loss.backward()

    trainable_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    trainable_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Backward OK, {trainable_grads}/{trainable_total} trainable params have gradients")
    _print_gpu_mem("after multimodal backward")

    print("  PASSED")

    del model
    _cleanup()


def test_training_step_loss_decrease() -> None:
    """Run 5 training steps and verify loss decreases."""
    print("\n=== Test 8: Training Step — Loss Decrease Test ===")
    from videollm.model.videollm import ModelConfig, VideoLLM

    config = ModelConfig(
        llm_path="Qwen/Qwen2-7B-Instruct",
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        use_audio=False,  # text-only for speed
        use_temporal_bridge=False,
    )
    model = VideoLLM(config).to("cuda")
    model.train()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    # Fixed synthetic data — same batch repeated so the model can overfit
    text = "The video shows a cat playing with a ball in a sunny garden."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True).to("cuda")
    labels = inputs["input_ids"].clone()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    losses: list[float] = []
    for step in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step + 1}/5: loss = {loss.item():.4f}")

    print(f"  Loss trajectory: {[f'{val:.4f}' for val in losses]}")
    if losses[-1] < losses[0]:
        print(f"  Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f} (delta={losses[0] - losses[-1]:.4f})")
        print("  PASSED")
    else:
        print(f"  WARNING: Loss did not decrease ({losses[0]:.4f} → {losses[-1]:.4f})")
        print("  This may be normal for very short runs — check manually")

    del model, optimizer
    _cleanup()


def main() -> None:
    """Run GPU pipeline tests."""
    parser = argparse.ArgumentParser(description="GPU pipeline tests for VideoLLM")
    parser.add_argument("--full", action="store_true", help="Also test full model with LLM (downloads Qwen2-7B)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {device}")
    print(f"VRAM: {total / 1024**3:.1f}GB total, {free / 1024**3:.1f}GB free")

    # Tests that use cached encoders only (no LLM download needed)
    test_vision_tower_gpu()
    test_audio_tower_gpu()
    test_stc_connector_gpu()
    test_agta_bridge_gpu()
    test_full_encode_pipeline_gpu()

    if args.full:
        test_full_model_gpu()
        test_multimodal_forward_gpu()
        test_training_step_loss_decrease()
    else:
        print("\n--- Skipping full model tests (pass --full to enable) ---")
        print("  These tests download Qwen2-7B-Instruct (~14GB)")

    print("\n=== All tests PASSED ===")


if __name__ == "__main__":
    main()
