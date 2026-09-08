"""Debug forward pass to diagnose gradient explosion."""

import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from videollm.data.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, VIDEO_TOKEN_INDEX  # noqa: E402
from videollm.data.dataset import VideoAudioDataset  # noqa: E402
from videollm.model.videollm import ModelConfig, VideoLLM  # noqa: E402


def main() -> None:
    """Run diagnostic forward pass on a single sample."""
    device = torch.device("cuda:0")

    logger.info("=== Building model (fp32, no LoRA, freeze LLM) ===")
    config = ModelConfig(
        mm_projector_type="stc",
        freeze_vision=True,
        freeze_audio=True,
        freeze_llm=True,
        use_lora=False,
        use_audio=True,
        use_temporal_bridge=False,  # Start simple, no bridge
    )
    model = VideoLLM(config)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    # Check output_scale values
    if hasattr(model.vision_projector, "output_scale"):
        logger.info("vision_projector.output_scale = %s", model.vision_projector.output_scale.item())
    if model.audio_projector is not None and hasattr(model.audio_projector, "output_scale"):
        logger.info("audio_projector.output_scale = %s", model.audio_projector.output_scale.item())

    # Load one sample
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
    logger.info("\n=== Sample 0 ===")
    logger.info("input_ids shape: %s", sample["input_ids"].shape)
    logger.info("labels shape: %s", sample["labels"].shape)
    logger.info("pixel_values shape: %s", sample["pixel_values"].shape)
    logger.info("waveforms shape: %s", sample["waveforms"].shape)

    # Check modal tokens
    ids = sample["input_ids"]
    labels = sample["labels"]
    n_video_tokens = (ids == VIDEO_TOKEN_INDEX).sum().item()
    n_audio_tokens = (ids == AUDIO_TOKEN_INDEX).sum().item()
    n_ignore_labels = (labels == IGNORE_INDEX).sum().item()
    n_real_labels = (labels != IGNORE_INDEX).sum().item()
    logger.info("video placeholders: %d, audio placeholders: %d", n_video_tokens, n_audio_tokens)
    logger.info("IGNORE labels: %d, real labels: %d", n_ignore_labels, n_real_labels)
    logger.info("real label values: %s", labels[labels != IGNORE_INDEX].tolist())

    # Batch it
    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    labels_batch = labels.unsqueeze(0).to(device)
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    waveforms = sample["waveforms"].unsqueeze(0).to(device)

    logger.info("\n=== Encoding video ===")
    with torch.no_grad():
        visual_tokens = model.encode_video(pixel_values)
    logger.info(
        "visual_tokens: shape=%s, mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        visual_tokens.shape,
        visual_tokens.mean().item(),
        visual_tokens.std().item(),
        visual_tokens.min().item(),
        visual_tokens.max().item(),
    )

    logger.info("\n=== Encoding audio ===")
    with torch.no_grad():
        audio_tokens = model.encode_audio(waveforms)
    logger.info(
        "audio_tokens: shape=%s, mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        audio_tokens.shape,
        audio_tokens.mean().item(),
        audio_tokens.std().item(),
        audio_tokens.min().item(),
        audio_tokens.max().item(),
    )

    logger.info("\n=== LLM embedding reference ===")
    emb_layer = model.llm.get_input_embeddings()
    emb_weight = emb_layer.weight
    logger.info("LLM embedding weight: mean=%.6f, std=%.6f", emb_weight.mean().item(), emb_weight.std().item())

    # Check a few text token embeddings
    text_token_ids = ids[ids >= 0][:10]
    text_embeds = emb_layer(text_token_ids.to(device))
    logger.info(
        "text_embeds sample: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        text_embeds.mean().item(),
        text_embeds.std().item(),
        text_embeds.min().item(),
        text_embeds.max().item(),
    )

    logger.info("\n=== prepare_multimodal_inputs ===")
    with torch.no_grad():
        inputs_embeds, new_mask, new_labels = model.prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch,
            pixel_values=pixel_values,
            waveforms=waveforms,
        )
    logger.info("inputs_embeds: shape=%s", inputs_embeds.shape)
    logger.info(
        "  mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        inputs_embeds.mean().item(),
        inputs_embeds.std().item(),
        inputs_embeds.min().item(),
        inputs_embeds.max().item(),
    )
    logger.info("new_mask sum: %d / %d", new_mask.sum().item(), new_mask.numel())
    if new_labels is not None:
        n_real = (new_labels != IGNORE_INDEX).sum().item()
        n_total = new_labels.numel()
        logger.info("new_labels: %d real / %d total", n_real, n_total)
        real_labels = new_labels[new_labels != IGNORE_INDEX]
        logger.info("  real label range: min=%d, max=%d", real_labels.min().item(), real_labels.max().item())
        logger.info("  real labels[:20]: %s", real_labels[:20].tolist())

    logger.info("\n=== Forward through LLM (frozen, no grad) ===")
    with torch.no_grad():
        outputs = model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_mask,
            labels=new_labels,
            return_dict=True,
        )
    logger.info("Loss: %.6e", outputs.loss.item())
    logits = outputs.logits
    logger.info(
        "Logits: shape=%s, mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        logits.shape,
        logits.mean().item(),
        logits.std().item(),
        logits.min().item(),
        logits.max().item(),
    )

    # Check logit distribution at positions where labels are real
    if new_labels is not None:
        real_positions = (new_labels[0] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if len(real_positions) > 0:
            # Logits are shifted by 1 for loss computation (predict next token)
            # Loss compares logits[:-1] with labels[1:]
            shifted_positions = real_positions - 1
            shifted_positions = shifted_positions[shifted_positions >= 0]
            if len(shifted_positions) > 0:
                pos_logits = logits[0, shifted_positions, :]
                logger.info(
                    "Logits at label positions: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                    pos_logits.mean().item(),
                    pos_logits.std().item(),
                    pos_logits.min().item(),
                    pos_logits.max().item(),
                )

    # Now test with ONLY text (no video/audio) to see what the baseline loss looks like
    logger.info("\n=== Baseline: text-only forward (no modal tokens) ===")
    # Create a simple text-only input
    text_input = "Describe what you hear and see in this video.\nA woman talks nearby as water pours"
    enc = tokenizer(text_input, return_tensors="pt", max_length=512, truncation=True)
    text_ids = enc["input_ids"].to(device)
    text_mask = enc["attention_mask"].to(device)
    # Labels = same as input_ids (teacher forcing)
    text_labels = text_ids.clone()
    with torch.no_grad():
        text_out = model.llm(input_ids=text_ids, attention_mask=text_mask, labels=text_labels, return_dict=True)
    logger.info("Text-only loss: %.6e", text_out.loss.item())
    logger.info(
        "Text-only logits: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        text_out.logits.mean().item(),
        text_out.logits.std().item(),
        text_out.logits.min().item(),
        text_out.logits.max().item(),
    )

    # Test: what if we feed random embeddings matching LLM embedding stats?
    logger.info("\n=== Test: random embeddings at LLM scale ===")
    seq_len = inputs_embeds.shape[1]
    random_embeds = torch.randn(1, seq_len, 3584, device=device, dtype=torch.float32) * emb_weight.std().item()
    with torch.no_grad():
        rand_out = model.llm(
            inputs_embeds=random_embeds,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long, device=device),
            labels=new_labels,
            return_dict=True,
        )
    logger.info("Random embeddings loss: %.6e", rand_out.loss.item())
    logger.info(
        "Random logits: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        rand_out.logits.mean().item(),
        rand_out.logits.std().item(),
        rand_out.logits.min().item(),
        rand_out.logits.max().item(),
    )


if __name__ == "__main__":
    main()
