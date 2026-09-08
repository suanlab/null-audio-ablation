"""Compute per-sample audio utility weights for Reward-Weighted SFT.

Compares model predictions with real audio vs silent audio to determine
how much each training sample benefits from audio information.

Weight assignment:
    - real_correct AND silent_wrong  → audio helps    → weight = 2.0
    - real_correct AND silent_correct → audio neutral → weight = 1.0
    - real_wrong AND silent_correct  → audio hurts   → weight = 0.5
    - real_wrong AND silent_wrong    → both fail      → weight = 1.0

Usage::

    python scripts/compute_sample_weights.py \
        --real_results eval_results/train_utility_real.json \
        --silent_results eval_results/train_utility_silent.json \
        --output data/instruct/avqa_train_quick5k_weights.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Weight constants for audio utility categories
WEIGHT_AUDIO_HELPS = 2.0  # real correct, silent wrong
WEIGHT_AUDIO_NEUTRAL = 1.0  # both correct or both wrong
WEIGHT_AUDIO_HURTS = 0.5  # real wrong, silent correct


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Compute per-sample audio utility weights")
    parser.add_argument("--real_results", type=str, required=True, help="Path to real-audio eval results JSON")
    parser.add_argument("--silent_results", type=str, required=True, help="Path to silent-audio eval results JSON")
    parser.add_argument("--output", type=str, required=True, help="Output weights JSON path")
    parser.add_argument(
        "--weight_audio_helps",
        type=float,
        default=WEIGHT_AUDIO_HELPS,
        help=f"Weight for samples where audio helps (default: {WEIGHT_AUDIO_HELPS})",
    )
    parser.add_argument(
        "--weight_audio_neutral",
        type=float,
        default=WEIGHT_AUDIO_NEUTRAL,
        help=f"Weight for neutral samples (default: {WEIGHT_AUDIO_NEUTRAL})",
    )
    parser.add_argument(
        "--weight_audio_hurts",
        type=float,
        default=WEIGHT_AUDIO_HURTS,
        help=f"Weight for samples where audio hurts (default: {WEIGHT_AUDIO_HURTS})",
    )
    return parser.parse_args()


def load_predictions(results_path: str) -> dict[str, dict[str, str]]:
    """Load per-sample predictions from eval results JSON.

    Args:
        results_path: Path to eval results JSON with 'predictions' list.

    Returns:
        Dict mapping video filename → prediction info dict.
    """
    with open(results_path) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError(f"Expected 'predictions' list in {results_path}")

    result: dict[str, dict[str, str]] = {}
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        video = pred.get("video", "")
        index = pred.get("index", "")
        # Use index+video as key to handle potential duplicate videos
        key = f"{index}:{video}"
        result[key] = pred
    return result


def compute_weights(
    real_preds: dict[str, dict[str, str]],
    silent_preds: dict[str, dict[str, str]],
    w_helps: float,
    w_neutral: float,
    w_hurts: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute per-sample weights from real vs silent predictions.

    Args:
        real_preds: Predictions with real audio.
        silent_preds: Predictions with silent audio.
        w_helps: Weight for audio-helps category.
        w_neutral: Weight for neutral category.
        w_hurts: Weight for audio-hurts category.

    Returns:
        Tuple of (weights dict mapping index:video → weight, category counts).
    """
    weights: dict[str, float] = {}
    category_counts: dict[str, int] = Counter()

    for key, real_pred in real_preds.items():
        silent_pred = silent_preds.get(key)
        if silent_pred is None:
            weights[key] = w_neutral
            category_counts["missing_silent"] += 1
            continue

        real_correct = real_pred.get("correct", "").lower() == "true"
        silent_correct = silent_pred.get("correct", "").lower() == "true"

        if real_correct and not silent_correct:
            # Audio genuinely helps — reinforce audio usage
            weights[key] = w_helps
            category_counts["audio_helps"] += 1
        elif real_correct and silent_correct:
            # Both correct — audio not needed for this sample
            weights[key] = w_neutral
            category_counts["audio_neutral"] += 1
        elif not real_correct and silent_correct:
            # Audio actually hurts — downweight
            weights[key] = w_hurts
            category_counts["audio_hurts"] += 1
        else:
            # Both wrong — hard sample, normal weight
            weights[key] = w_neutral
            category_counts["both_wrong"] += 1

    return weights, dict(category_counts)


def main() -> None:
    """Compute and save per-sample audio utility weights."""
    args = parse_args()

    logger.info("Loading real-audio predictions from %s", args.real_results)
    real_preds = load_predictions(args.real_results)
    logger.info("Loaded %d real-audio predictions", len(real_preds))

    logger.info("Loading silent-audio predictions from %s", args.silent_results)
    silent_preds = load_predictions(args.silent_results)
    logger.info("Loaded %d silent-audio predictions", len(silent_preds))

    weights, category_counts = compute_weights(
        real_preds=real_preds,
        silent_preds=silent_preds,
        w_helps=args.weight_audio_helps,
        w_neutral=args.weight_audio_neutral,
        w_hurts=args.weight_audio_hurts,
    )

    # Build output: list of {index, video, weight, category} aligned with training data order
    output_weights: list[dict[str, object]] = []
    for key, weight in sorted(weights.items(), key=lambda kv: int(kv[0].split(":")[0])):
        parts = key.split(":", 1)
        index = int(parts[0])
        video = parts[1] if len(parts) > 1 else ""
        output_weights.append({"index": index, "video": video, "weight": weight})

    output_data = {
        "total_samples": len(output_weights),
        "category_counts": category_counts,
        "weight_config": {
            "audio_helps": args.weight_audio_helps,
            "audio_neutral": args.weight_audio_neutral,
            "audio_hurts": args.weight_audio_hurts,
        },
        "real_accuracy": sum(1 for p in real_preds.values() if p.get("correct", "").lower() == "true")
        / max(len(real_preds), 1)
        * 100,
        "silent_accuracy": sum(1 for p in silent_preds.values() if p.get("correct", "").lower() == "true")
        / max(len(silent_preds), 1)
        * 100,
        "weights": output_weights,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d sample weights to %s", len(output_weights), output_path)
    logger.info("Category breakdown:")
    for cat, count in sorted(category_counts.items()):
        pct = count / max(len(output_weights), 1) * 100
        logger.info("  %s: %d (%.1f%%)", cat, count, pct)
    logger.info("Real accuracy on train: %.1f%%", output_data["real_accuracy"])
    logger.info("Silent accuracy on train: %.1f%%", output_data["silent_accuracy"])


if __name__ == "__main__":
    main()
