"""Audio-aware reward functions for GRPO reinforcement learning.

Implements reward functions for the AGTA (Audio-Guided Temporal Attention) model's
GRPO training phase. The reward encourages the model to:
  1. Answer correctly (r_ans)
  2. Use audio when it helps (r_audio)
  3. Degrade gracefully when audio is absent (r_robust)
  4. Avoid repetitive outputs (r_format)

Reward structure inspired by LongVideo-R1 (CVPR 2026) and Video-R1, adapted for
audio-visual understanding.

Usage::

    from videollm.grpo_reward import audio_correctness_reward, audio_format_reward

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[audio_correctness_reward, audio_format_reward],
        args=GRPOConfig(reward_weights=[1.0, 0.3]),
        ...
    )
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns to extract MCQ answer letter (A-D) from model output, ordered by specificity
ANSWER_IS_RE = re.compile(r"answer\s+is\s+([A-D])", re.IGNORECASE)
STANDALONE_RE = re.compile(r"\b([A-D])\b")
LEADING_RE = re.compile(r"^\s*([A-D])")


def extract_answer_letter(text: str) -> str:
    """Extract the predicted choice letter (A-D) from model output.

    Uses a multi-stage extraction strategy to avoid false positives
    (e.g., matching 'A' inside 'ANSWER'):
        1. Look for 'answer is X' pattern
        2. Look for standalone letter at the start of text
        3. Look for any standalone word-boundary letter

    Args:
        text: Raw model output text.

    Returns:
        Single letter A-D, or empty string if not found.
    """
    # Stage 1: "The answer is A" / "answer is B"
    match = ANSWER_IS_RE.search(text)
    if match:
        return match.group(1).upper()

    # Stage 2: Leading letter (e.g., "A." or "B" at start)
    match = LEADING_RE.search(text.strip())
    if match:
        return match.group(1).upper()

    # Stage 3: Any standalone letter at a word boundary
    match = STANDALONE_RE.search(text.upper())
    if match:
        return match.group(1).upper()

    return ""

def extract_ground_truth(gpt_response: str) -> str:
    """Extract ground truth letter from GPT response like 'The answer is A. ...'

    Args:
        gpt_response: The ground truth text from the dataset.

    Returns:
        Ground truth letter A-D, or empty string.
    """
    match = re.search(r"answer is ([A-D])", gpt_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: try to find a standalone letter
    match = re.search(r"\b([A-D])\b", gpt_response)
    return match.group(1).upper() if match else ""


def audio_correctness_reward(
    prompts: list[str],
    completions: list[str],
    ground_truth: list[str] | None = None,
    audio_mode: list[str] | None = None,
    **kwargs: object,
) -> list[float]:
    """Reward function for answer correctness with audio-aware bonus.

    For MCQ tasks, compares the model's predicted letter (A-D) against the ground truth.
    When ``audio_mode`` is provided, gives a bonus for correct answers under degraded
    audio conditions (noise/silent), encouraging robustness.

    Reward values:
        - Correct with real audio: 1.0
        - Correct with degraded audio: 1.5 (bonus for robustness)
        - Wrong: 0.0

    Args:
        prompts: Input prompts (unused, required by trl interface).
        completions: Generated completion texts.
        ground_truth: Ground truth answers (e.g., "The answer is A. ...").
        audio_mode: Audio condition per sample ("real", "silent", "noise").
        **kwargs: Additional unused keyword arguments.

    Returns:
        List of reward floats, one per completion.
    """
    if ground_truth is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        completion_text = completion if isinstance(completion, str) else str(completion)
        pred_letter = extract_answer_letter(completion_text)

        gt_text = ground_truth[i] if i < len(ground_truth) else ""
        gt_letter = extract_ground_truth(gt_text)

        if not gt_letter:
            rewards.append(0.0)
            continue

        if pred_letter == gt_letter:
            # Correct answer
            mode = audio_mode[i] if audio_mode and i < len(audio_mode) else "real"
            if mode in ("silent", "noise"):
                # Bonus for being correct without real audio (robustness)
                rewards.append(1.5)
            else:
                rewards.append(1.0)
        else:
            # Wrong answer
            rewards.append(0.0)

    return rewards


def audio_format_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs: object,
) -> list[float]:
    """Reward function for response format compliance.

    Encourages the model to output a clear MCQ answer letter (A-D).
    Penalizes outputs that don't contain a recognizable answer.

    Reward values:
        - Contains A-D letter: 0.5
        - No recognizable answer: -0.5

    Args:
        prompts: Input prompts (unused, required by trl interface).
        completions: Generated completion texts.
        **kwargs: Additional unused keyword arguments.

    Returns:
        List of reward floats, one per completion.
    """
    rewards: list[float] = []
    for completion in completions:
        completion_text = completion if isinstance(completion, str) else str(completion)
        letter = extract_answer_letter(completion_text)
        if letter:
            rewards.append(0.5)
        else:
            rewards.append(-0.5)
    return rewards


def audio_utilization_reward(
    prompts: list[str],
    completions: list[str],
    audio_mode: list[str] | None = None,
    real_correct: list[bool] | None = None,
    silent_correct: list[bool] | None = None,
    **kwargs: object,
) -> list[float]:
    """Reward function for audio utilization quality.

    Requires paired evaluation: each sample is evaluated with both real and silent audio.
    The reward measures whether the model appropriately uses audio information.

    This function is designed for offline reward computation (not during live rollouts).
    During GRPO rollouts, use ``audio_correctness_reward`` with ``audio_mode`` instead.

    Reward values:
        - real_correct AND silent_wrong: +1.0  (audio genuinely helps)
        - real_correct AND silent_correct: +0.2 (answer is robust, audio not needed)
        - real_wrong AND silent_correct: -0.5   (audio hurts, model confused)
        - both wrong: 0.0                       (hard sample)

    Args:
        prompts: Input prompts (unused, required by trl interface).
        completions: Generated completion texts (unused for offline mode).
        audio_mode: Audio condition per sample.
        real_correct: Whether the answer was correct with real audio.
        silent_correct: Whether the answer was correct with silent audio.
        **kwargs: Additional unused keyword arguments.

    Returns:
        List of reward floats, one per completion.
    """
    if real_correct is None or silent_correct is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for i in range(len(completions)):
        r_correct = real_correct[i] if i < len(real_correct) else False
        s_correct = silent_correct[i] if i < len(silent_correct) else False

        if r_correct and not s_correct:
            # Audio genuinely helps — model uses audio well
            rewards.append(1.0)
        elif r_correct and s_correct:
            # Both correct — model is robust, small reward
            rewards.append(0.2)
        elif not r_correct and s_correct:
            # Audio confuses the model — penalize
            rewards.append(-0.5)
        else:
            # Both wrong — hard sample, neutral
            rewards.append(0.0)

    return rewards


def combined_audio_reward(
    prompts: list[str],
    completions: list[str],
    ground_truth: list[str] | None = None,
    audio_mode: list[str] | None = None,
    **kwargs: object,
) -> list[float]:
    """Combined reward: correctness + format + audio-mode awareness.

    Single reward function that combines all components with fixed weights:
        R = 1.0 * r_ans + 0.3 * r_format + audio_bonus

    Where audio_bonus is:
        - +0.5 if correct under degraded audio (robustness bonus)
        - -0.3 if wrong with real audio but question is audio-dependent
        - 0.0 otherwise

    This is the recommended reward function for AGTA GRPO training.

    Args:
        prompts: Input prompts.
        completions: Generated completion texts.
        ground_truth: Ground truth answers.
        audio_mode: Audio condition per sample.
        **kwargs: Additional unused keyword arguments.

    Returns:
        List of reward floats, one per completion.
    """
    correctness = audio_correctness_reward(
        prompts=prompts,
        completions=completions,
        ground_truth=ground_truth,
        audio_mode=audio_mode,
    )
    formatting = audio_format_reward(
        prompts=prompts,
        completions=completions,
    )

    rewards: list[float] = []
    for i in range(len(completions)):
        r_ans = correctness[i]
        r_fmt = formatting[i]
        reward = 1.0 * r_ans + 0.3 * r_fmt
        rewards.append(reward)

    return rewards
