"""Audio-Visual GRPO Trainer for AGTA.

Self-contained Group Relative Policy Optimization implementation for multimodal
video-LLMs. Unlike trl's GRPOTrainer which is designed for standard HuggingFace
models, this handles custom multimodal inputs (pixel_values + waveforms) natively.

Key features:
  - Audio dropout during rollouts: half generations with real audio, half with silence
  - Custom reward functions for audio utilization
  - GRPO advantage computation with group normalization
  - Compatible with VideoLLM's generate() interface

Architecture inspired by Video-R1's temporal augmentation GRPO approach.

Usage::

    from videollm.grpo_trainer import AudioVisualGRPOTrainer

    trainer = AudioVisualGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_fn=combined_audio_reward,
        config=GRPOTrainingConfig(...),
    )
    trainer.train()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable

from videollm.data.constants import (
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_DURATION,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_NUM_FRAMES,
    VIDEO_TOKEN_INDEX,
)
from videollm.data.transforms import AudioTransform, VideoTransform
from videollm.grpo_reward import extract_answer_letter
from videollm.utils import load_audio_from_video

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainingConfig:
    """Configuration for Audio-Visual GRPO training.

    Args:
        output_dir: Directory to save checkpoints.
        num_generations: Number of completions per prompt (G in GRPO).
        audio_dropout_ratio: Fraction of generations with silent audio.
        max_completion_length: Maximum tokens per generated completion.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling probability.
        learning_rate: Optimizer learning rate.
        num_epochs: Number of training epochs.
        batch_size: Prompts per training step.
        gradient_accumulation_steps: Accumulation steps before optimizer update.
        beta: KL divergence penalty coefficient (0 = no reference model).
        epsilon: GRPO clipping parameter.
        max_grad_norm: Gradient clipping norm.
        num_frames: Video frames to sample per clip.
        logging_steps: Log every N optimizer steps.
        save_strategy: When to save checkpoints ("epoch" or "steps").
        bf16: Use bfloat16 precision.
    """

    output_dir: str = "checkpoints/grpo"
    num_generations: int = 8
    audio_dropout_ratio: float = 0.5
    max_completion_length: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    beta: float = 0.0
    epsilon: float = 0.2
    max_grad_norm: float = 1.0
    num_frames: int = DEFAULT_NUM_FRAMES
    logging_steps: int = 1
    save_strategy: str = "epoch"
    bf16: bool = True
    video_dir: str = "data/videos"
    reward_weights: list[float] = field(default_factory=lambda: [1.0])


class AudioVisualGRPOTrainer:
    """GRPO trainer for multimodal video-LLMs with audio dropout during rollouts.

    Implements the core GRPO algorithm:
      1. Generate G completions per prompt (mixed audio conditions)
      2. Score each completion with reward function
      3. Compute group-normalized advantages
      4. Update policy with clipped advantage-weighted loss

    Args:
        model: VideoLLM model instance.
        tokenizer: Text tokenizer.
        train_dataset: List of training samples with video/question/answer fields.
        reward_fn: Callable reward function matching trl interface.
        config: GRPO training configuration.
        ref_model: Optional reference model for KL penalty (None = no KL).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        train_dataset: list[dict[str, object]],
        reward_fn: Callable[..., list[float]],
        config: GRPOTrainingConfig,
        ref_model: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reward_fn = reward_fn
        self.config = config
        self.ref_model = ref_model

        self.device = next(model.parameters()).device
        self.video_transform = VideoTransform(is_train=False)
        self.audio_transform = AudioTransform(is_train=False)
        self.video_dir = Path(config.video_dir)

        # Determine number of real vs silent generations
        self.num_real = max(1, config.num_generations - int(config.num_generations * config.audio_dropout_ratio))
        self.num_silent = config.num_generations - self.num_real

        logger.info(
            "AudioVisualGRPOTrainer: G=%d (real=%d, silent=%d), epsilon=%.2f, beta=%.2f",
            config.num_generations,
            self.num_real,
            self.num_silent,
            config.epsilon,
            config.beta,
        )

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames.

        Args:
            video_path: Path to MP4 file.

        Returns:
            Tensor of shape ``(T, C, H, W)``.
        """
        import decord

        decord.bridge.set_bridge("torch")
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            total = len(vr)
            step = max(1, total // self.config.num_frames)
            indices = list(range(0, total, step))[: self.config.num_frames]
            while len(indices) < self.config.num_frames:
                indices.append(indices[-1] if indices else 0)
            frames = vr.get_batch(indices)  # (T, H, W, C)
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
            return self.video_transform(frames)  # (T, C, H', W')
        except Exception as e:
            logger.warning("Failed to load video %s: %s", video_path, e)
            return self.video_transform(torch.zeros(self.config.num_frames, 3, 384, 384))

    def _load_audio_waveform(self, video_path: str) -> torch.Tensor:
        """Load audio waveform from video.

        Args:
            video_path: Path to MP4 file.

        Returns:
            Waveform tensor of shape ``(samples,)``.
        """
        try:
            waveform, sr = load_audio_from_video(Path(video_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            mono = waveform.squeeze(0)  # (samples,)
            return self.audio_transform(mono, sr)
        except Exception:
            duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
            return torch.zeros(duration_samples)

    @torch.no_grad()
    def _generate_completions(
        self,
        sample: dict[str, object],
    ) -> tuple[list[str], list[str], list[torch.Tensor], list[torch.Tensor | None]]:
        """Generate G completions for a single prompt with mixed audio conditions.

        Half of the completions use real audio, half use silence (audio dropout).
        This teaches the model to be robust without audio while still using it when helpful.

        Args:
            sample: Training sample with video, question, ground_truth fields.

        Returns:
            Tuple of (completions, audio_modes, completion_token_ids, old_log_probs).
            completions: List of G decoded text strings.
            audio_modes: List of G strings ("real" or "silent").
            completion_token_ids: List of G tensors with generated token IDs.
            old_log_probs: List of G per-token log prob tensors (for KL), or None if beta=0.
        """
        video_file = str(sample.get("video", ""))
        video_path = str(self.video_dir / video_file)
        question = str(sample.get("prompt", ""))

        # Load video (same for all generations)
        pixel_values = self._load_video_frames(video_path).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # Load real audio
        real_waveform = self._load_audio_waveform(video_path).to(self.device)  # (samples,)
        silent_waveform = torch.zeros_like(real_waveform)  # (samples,)

        # Tokenize prompt
        encoding = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoding["input_ids"].to(self.device)  # (1, S)
        attention_mask = encoding["attention_mask"].to(self.device)  # (1, S)

        # Prepend modal tokens
        modal_ids = torch.tensor([[VIDEO_TOKEN_INDEX, AUDIO_TOKEN_INDEX]], dtype=input_ids.dtype, device=self.device)
        input_ids = torch.cat([modal_ids, input_ids], dim=1)  # (1, S+2)
        modal_mask = torch.ones(1, 2, dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat([modal_mask, attention_mask], dim=1)  # (1, S+2)

        completions: list[str] = []
        audio_modes: list[str] = []
        completion_ids_list: list[torch.Tensor] = []
        old_log_probs_list: list[torch.Tensor | None] = []

        self.model.eval()

        for gen_idx in range(self.config.num_generations):
            # Audio condition: first num_real with real, rest with silent
            if gen_idx < self.num_real:
                waveforms = real_waveform.unsqueeze(0)  # (1, samples)
                audio_mode = "real"
            else:
                waveforms = silent_waveform.unsqueeze(0)  # (1, samples)
                audio_mode = "silent"

            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                waveforms=waveforms,
                max_new_tokens=self.config.max_completion_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )  # (1, S+2+completion_len)

            # When generate() uses inputs_embeds (multimodal), HuggingFace
            # returns ONLY the generated tokens without the prompt prefix.
            # So output_ids IS the completion already.
            comp_ids = output_ids[0]  # (completion_len,)

            # Compute old (reference) log probs under current policy before any updates.
            # These serve as the "old policy" for KL regularization in GRPO.
            if self.config.beta > 0:
                old_lp = self._compute_log_probs(sample, comp_ids, audio_mode)  # (L,)
                old_log_probs_list.append(old_lp.detach())
            else:
                old_log_probs_list.append(None)

            decoded = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
            completions.append(decoded)
            audio_modes.append(audio_mode)
            completion_ids_list.append(comp_ids)

        return completions, audio_modes, completion_ids_list, old_log_probs_list

    def _compute_log_probs(
        self,
        sample: dict[str, object],
        completion_ids: torch.Tensor,
        audio_mode: str,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for a completion.

        Args:
            sample: Training sample.
            completion_ids: Token IDs of the completion, shape ``(L,)``.
            audio_mode: Audio condition ("real" or "silent").

        Returns:
            Per-token log probabilities tensor of shape ``(L,)``.
        """
        video_file = str(sample.get("video", ""))
        video_path = str(self.video_dir / video_file)
        question = str(sample.get("prompt", ""))

        # Load inputs
        pixel_values = self._load_video_frames(video_path).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        if audio_mode == "real":
            waveforms = self._load_audio_waveform(video_path).unsqueeze(0).to(self.device)  # (1, samples)
        else:
            duration_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION)
            waveforms = torch.zeros(1, duration_samples, device=self.device)  # (1, samples)

        # Build full input: prompt + completion
        encoding = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
        prompt_ids = encoding["input_ids"].to(self.device)  # (1, S)

        modal_ids = torch.tensor([[VIDEO_TOKEN_INDEX, AUDIO_TOKEN_INDEX]], dtype=prompt_ids.dtype, device=self.device)
        prompt_ids = torch.cat([modal_ids, prompt_ids], dim=1)  # (1, S+2)

        # Concatenate prompt and completion
        full_ids = torch.cat([prompt_ids, completion_ids.unsqueeze(0)], dim=1)  # (1, S+2+L)
        full_mask = torch.ones_like(full_ids)

        # Forward pass
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=pixel_values,
            waveforms=waveforms,
        )
        logits = outputs.logits  # (1, expanded_seq_len, V) — expanded due to multimodal token replacement

        # Extract log probs for completion tokens.
        # IMPORTANT: prepare_multimodal_inputs replaces VIDEO_TOKEN_INDEX and AUDIO_TOKEN_INDEX
        # with L_v and L_a embedding tokens respectively, so logits.shape[1] != full_ids.shape[1].
        # Instead of tracking the expanded prompt length, we use the fact that completion tokens
        # are always at the END of the sequence. Logits at position t predict token at t+1,
        # so for L completion tokens at the tail, we slice logits[-(L+1):-1].
        comp_len = completion_ids.shape[0]  # L
        completion_logits = logits[0, -(comp_len + 1) : -1, :]  # (L, V)
        log_probs = F.log_softmax(completion_logits, dim=-1)  # (L, V)

        # Gather log probs for actual completion tokens
        token_log_probs = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)  # (L,)

        return token_log_probs

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-normalized advantages from rewards.

        GRPO normalizes rewards within each group (all completions for one prompt):
            A = (R - mean(R_group)) / (std(R_group) + eps)

        Args:
            rewards: Reward tensor of shape ``(G,)`` for one prompt group.

        Returns:
            Advantage tensor of shape ``(G,)``.
        """
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages  # (G,)

    def _grpo_loss_for_prompt(
        self,
        sample: dict[str, object],
        completions: list[str],
        audio_modes: list[str],
        completion_ids_list: list[torch.Tensor],
        rewards: torch.Tensor,
        old_log_probs_list: list[torch.Tensor | None] | None = None,
    ) -> float:
        """Compute GRPO loss for one prompt group of completions.

        Processes each completion individually with immediate backward to avoid OOM.
        Each forward pass creates a large computation graph, so we free it after each.

        Args:
            sample: Training sample.
            completions: List of G completion texts.
            audio_modes: List of G audio conditions.
            completion_ids_list: List of G completion token tensors.
            rewards: Reward tensor of shape ``(G,)``.
            old_log_probs_list: Optional old policy log probs for KL penalty.

        Returns:
            Scalar loss value (detached float for logging).
        """
        advantages = self._compute_advantages(rewards)  # (G,)

        total_loss_val = 0.0
        num_valid = 0

        for g in range(len(completions)):
            comp_ids = completion_ids_list[g]
            if comp_ids.numel() == 0:
                continue

            # Compute log probs under current policy (with grad)
            self.model.train()
            log_probs = self._compute_log_probs(sample, comp_ids, audio_modes[g])  # (L,)

            # Mean log prob for this completion
            mean_log_prob = log_probs.mean()  # scalar

            # GRPO loss: -advantage * log_prob (simplified without clipping for first iteration)
            gen_loss = -advantages[g] * mean_log_prob

            # KL penalty using old log probs from generation step
            if self.config.beta > 0 and old_log_probs_list is not None and old_log_probs_list[g] is not None:
                old_lp = old_log_probs_list[g].to(log_probs.device)  # (L,)
                kl = (log_probs - old_lp).mean()
                gen_loss = gen_loss + self.config.beta * kl

            # Scale and backward immediately to free the computation graph
            scaled = gen_loss / (len(completions) * self.config.gradient_accumulation_steps)
            scaled.backward()
            total_loss_val += gen_loss.item()
            num_valid += 1

            # Free memory between completions
            del log_probs, mean_log_prob, gen_loss, scaled
            torch.cuda.empty_cache()

        if num_valid > 0:
            total_loss_val /= num_valid

        return total_loss_val

    def train(self) -> dict[str, float]:
        """Run GRPO training loop.

        Returns:
            Dictionary of training metrics.
        """
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.0,
        )

        total_steps = 0
        metrics: dict[str, list[float]] = {
            "loss": [],
            "reward_mean": [],
            "reward_std": [],
            "real_accuracy": [],
            "silent_accuracy": [],
        }

        logger.info("Starting GRPO training: %d samples, %d epochs", len(self.train_dataset), config.num_epochs)

        for epoch in range(config.num_epochs):
            epoch_losses: list[float] = []

            for step_idx, sample in enumerate(self.train_dataset):
                # Step 1: Generate completions with mixed audio (no grad for generation)
                # Old log probs for KL are also computed here under no_grad
                with torch.no_grad():
                    completions, audio_modes, completion_ids_list, old_log_probs_list = self._generate_completions(sample)

                # Debug: log completions and rewards for first 5 steps
                if step_idx < 5:
                    gt_text = str(sample.get('ground_truth', ''))
                    logger.info('[DEBUG step %d] GT: %s', step_idx, gt_text[:80])
                    for gi, (comp, mode) in enumerate(zip(completions, audio_modes, strict=False)):
                        pred = extract_answer_letter(comp)
                        logger.info('  [gen %d] mode=%s, pred=%s, text=%r', gi, mode, pred, comp[:100])

                # Step 2: Compute rewards
                ground_truth_text = str(sample.get("ground_truth", ""))
                reward_values = self.reward_fn(
                    prompts=[str(sample.get("prompt", ""))] * config.num_generations,
                    completions=completions,
                    ground_truth=[ground_truth_text] * config.num_generations,
                    audio_mode=audio_modes,
                )
                rewards = torch.tensor(reward_values, dtype=torch.float32, device=self.device)  # (G,)

                if step_idx < 5:
                    logger.info('[DEBUG step %d] rewards=%s, advantages would be: mean=%.3f, std=%.3f',
                                step_idx, [f'{r:.2f}' for r in reward_values], rewards.mean().item(), rewards.std().item())

                # Step 3: Compute GRPO loss
                loss = self._grpo_loss_for_prompt(
                    sample=sample,
                    completions=completions,
                    audio_modes=audio_modes,
                    completion_ids_list=completion_ids_list,
                    rewards=rewards,
                    old_log_probs_list=old_log_probs_list,
                )

                # Step 4: Gradient accumulation
                # Note: backward() is already called inside _grpo_loss_for_prompt
                # to avoid OOM from holding multiple computation graphs.
                # We just need to handle the optimizer step.
                if (step_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1

                # Track metrics
                epoch_losses.append(loss)
                metrics["loss"].append(loss)
                metrics["reward_mean"].append(rewards.mean().item())
                metrics["reward_std"].append(rewards.std().item())

                # Per-mode accuracy
                real_correct = sum(
                    1
                    for c, m, gt in zip(completions, audio_modes, [ground_truth_text] * len(completions), strict=False)
                    if m == "real" and _is_correct(c, gt)
                )
                silent_correct = sum(
                    1
                    for c, m, gt in zip(completions, audio_modes, [ground_truth_text] * len(completions), strict=False)
                    if m == "silent" and _is_correct(c, gt)
                )
                if self.num_real > 0:
                    metrics["real_accuracy"].append(real_correct / self.num_real)
                if self.num_silent > 0:
                    metrics["silent_accuracy"].append(silent_correct / max(self.num_silent, 1))

                # Logging
                if (step_idx + 1) % config.logging_steps == 0:
                    avg_loss = sum(epoch_losses[-config.logging_steps :]) / min(config.logging_steps, len(epoch_losses))
                    avg_reward = rewards.mean().item()
                    logger.info(
                        "[Epoch %d/%d, Step %d/%d] loss=%.4f, reward=%.3f, real_acc=%.1f%%, silent_acc=%.1f%%",
                        epoch + 1,
                        config.num_epochs,
                        step_idx + 1,
                        len(self.train_dataset),
                        avg_loss,
                        avg_reward,
                        (real_correct / max(self.num_real, 1)) * 100,
                        (silent_correct / max(self.num_silent, 1)) * 100,
                    )

            # End of epoch
            logger.info(
                "Epoch %d complete: avg_loss=%.4f, avg_reward=%.3f",
                epoch + 1,
                sum(epoch_losses) / max(len(epoch_losses), 1),
                sum(metrics["reward_mean"][-len(epoch_losses) :]) / max(len(epoch_losses), 1),
            )

            if config.save_strategy == "epoch":
                self._save_checkpoint(output_dir / f"epoch_{epoch + 1}")

        # Save final
        self._save_checkpoint(output_dir / "final")

        return {
            "avg_loss": sum(metrics["loss"]) / max(len(metrics["loss"]), 1),
            "avg_reward": sum(metrics["reward_mean"]) / max(len(metrics["reward_mean"]), 1),
            "total_steps": total_steps,
        }

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Directory to save the checkpoint.
        """
        path.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_model

            save_model(self.model, str(path / "model.safetensors"))
        except ImportError:
            torch.save(self.model.state_dict(), str(path / "model.pt"))
        self.tokenizer.save_pretrained(str(path))
        logger.info("Saved checkpoint to %s", path)


def _is_correct(completion: str, ground_truth: str) -> bool:
    """Check if completion matches ground truth answer.

    Args:
        completion: Model completion text.
        ground_truth: Ground truth answer text.

    Returns:
        True if predicted letter matches ground truth letter.
    """
    pred = extract_answer_letter(completion)
    gt_match = re.search(r"answer is ([A-D])", ground_truth, re.IGNORECASE)
    gt = gt_match.group(1).upper() if gt_match else ""
    return pred == gt and pred != ""
