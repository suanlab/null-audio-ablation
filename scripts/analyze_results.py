#!/usr/bin/env python3
"""Comprehensive analysis of AGTA audio ablation experiments.

Compares QB v1 (broken bridge) vs QB v2 (zero-init residual bridge)
across 5 audio modes: real, shuffled, shifted, noise, silent.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_results(prefix: str) -> dict[str, dict]:
    """Load all 5-mode results for a given prefix."""
    modes = ["real", "shuffled", "shifted", "noise", "silent"]
    results = {}
    for mode in modes:
        path = Path(f"eval_results/{prefix}_{mode}.json")
        if path.exists():
            with open(path) as f:
                results[mode] = json.load(f)
    return results


def load_test_data() -> list[dict]:
    """Load AVQA test data for question metadata."""
    test_data = []
    with open("data/instruct/avqa_test_clean.jsonl") as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data


def extract_question_type(conversation: str) -> str:
    """Extract question type from the conversation text."""
    q_lower = conversation.lower()
    if "what are the people doing" in q_lower or "what is the" in q_lower:
        return "activity_recognition"
    elif "what happened" in q_lower:
        return "event_recognition"
    elif "what sound" in q_lower or "what do you hear" in q_lower:
        return "sound_identification"
    elif "where" in q_lower:
        return "spatial"
    elif "how many" in q_lower:
        return "counting"
    elif "which" in q_lower:
        return "selection"
    else:
        return "other"


def analyze_per_sample(v1_results: dict, v2_results: dict) -> None:
    """Detailed per-sample comparison between v1 and v2."""
    print("\n" + "=" * 80)
    print("PER-SAMPLE ANALYSIS: QB v1 vs QB v2 (real audio mode)")
    print("=" * 80)

    v1_preds = {p["index"]: p for p in v1_results["real"]["predictions"]}
    v2_preds = {p["index"]: p for p in v2_results["real"]["predictions"]}

    # Categories: both correct, v2 fixed, v2 broke, both wrong
    both_correct = []
    v2_fixed = []
    v2_broke = []
    both_wrong = []

    for idx in v1_preds:
        v1_correct = v1_preds[idx]["correct"] == "True"
        v2_correct = v2_preds[idx]["correct"] == "True"

        sample = {
            "index": idx,
            "video": v1_preds[idx]["video"],
            "gt": v1_preds[idx]["gt"],
            "v1_pred": v1_preds[idx]["pred"],
            "v2_pred": v2_preds[idx]["pred"],
        }

        if v1_correct and v2_correct:
            both_correct.append(sample)
        elif not v1_correct and v2_correct:
            v2_fixed.append(sample)
        elif v1_correct and not v2_correct:
            v2_broke.append(sample)
        else:
            both_wrong.append(sample)

    total = len(v1_preds)
    print(f"\nTotal samples: {total}")
    print(f"  Both correct:    {len(both_correct):3d} ({100 * len(both_correct) / total:.1f}%)")
    print(f"  V2 fixed (v1✗→v2✓): {len(v2_fixed):3d} ({100 * len(v2_fixed) / total:.1f}%)")
    print(f"  V2 broke (v1✓→v2✗): {len(v2_broke):3d} ({100 * len(v2_broke) / total:.1f}%)")
    print(f"  Both wrong:      {len(both_wrong):3d} ({100 * len(both_wrong) / total:.1f}%)")

    # Show samples that v2 fixed
    if v2_fixed:
        print(f"\n--- Samples V2 FIXED ({len(v2_fixed)}) ---")
        for s in v2_fixed[:10]:
            print(f"  [{s['index']}] {s['video']}: gt={s['gt']}, v1={s['v1_pred']}→v2={s['v2_pred']}")

    # Show samples that v2 broke
    if v2_broke:
        print(f"\n--- Samples V2 BROKE ({len(v2_broke)}) ---")
        for s in v2_broke[:10]:
            print(f"  [{s['index']}] {s['video']}: gt={s['gt']}, v1={s['v1_pred']}→v2={s['v2_pred']}")


def analyze_audio_dependency(v2_results: dict) -> None:
    """Analyze how model behavior changes across audio modes."""
    print("\n" + "=" * 80)
    print("AUDIO DEPENDENCY ANALYSIS (QB v2)")
    print("=" * 80)

    modes = ["real", "shuffled", "shifted", "noise", "silent"]
    mode_preds = {}

    for mode in modes:
        if mode in v2_results:
            mode_preds[mode] = {p["index"]: p for p in v2_results[mode]["predictions"]}

    if "real" not in mode_preds:
        print("Missing real mode results")
        return

    # For each sample, check consistency across modes
    indices = list(mode_preds["real"].keys())

    # Count: how many modes get each sample correct?
    consistency = defaultdict(list)
    for idx in indices:
        correct_modes = []
        for mode in modes:
            if mode in mode_preds:
                is_correct = mode_preds[mode][idx]["correct"] == "True"
                if is_correct:
                    correct_modes.append(mode)
        consistency[len(correct_modes)].append(idx)

    print("\n--- Sample Difficulty Distribution ---")
    for n_correct in sorted(consistency.keys()):
        count = len(consistency[n_correct])
        pct = 100 * count / len(indices)
        print(f"  Correct in {n_correct}/5 modes: {count:3d} samples ({pct:.1f}%)")

    # Audio-essential samples: correct ONLY with real audio
    audio_essential = []
    for idx in indices:
        real_correct = mode_preds["real"][idx]["correct"] == "True"
        noise_correct = mode_preds.get("noise", {}).get(idx, {}).get("correct") == "True"
        silent_correct = mode_preds.get("silent", {}).get(idx, {}).get("correct") == "True"

        if real_correct and not noise_correct and not silent_correct:
            audio_essential.append(idx)

    print("\n--- Audio-Essential Samples ---")
    print(
        f"  Correct with real audio BUT wrong with noise AND silent: {len(audio_essential)} ({100 * len(audio_essential) / len(indices):.1f}%)"
    )

    # Audio-harmful samples: correct WITHOUT audio but wrong WITH
    audio_harmful = []
    for idx in indices:
        real_correct = mode_preds["real"][idx]["correct"] == "True"
        silent_correct = mode_preds.get("silent", {}).get(idx, {}).get("correct") == "True"

        if not real_correct and silent_correct:
            audio_harmful.append(idx)

    print(
        f"  Correct with silent BUT wrong with real: {len(audio_harmful)} ({100 * len(audio_harmful) / len(indices):.1f}%)"
    )

    # Shifted vs Real agreement
    if "shifted" in mode_preds:
        agree = 0
        for idx in indices:
            if mode_preds["real"][idx]["pred"] == mode_preds["shifted"][idx]["pred"]:
                agree += 1
        print("\n--- Shifted vs Real Agreement ---")
        print(f"  Same prediction: {agree}/{len(indices)} ({100 * agree / len(indices):.1f}%)")

    # Prediction distribution per mode
    print("\n--- Prediction Distribution per Mode ---")
    for mode in modes:
        if mode in mode_preds:
            preds = [mode_preds[mode][idx]["pred"] for idx in indices]
            counter = Counter(preds)
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))
            print(f"  {mode:10s}: {dist_str}")

    # GT distribution
    gts = [mode_preds["real"][idx]["gt"] for idx in indices]
    gt_counter = Counter(gts)
    gt_str = ", ".join(f"{k}:{v}" for k, v in sorted(gt_counter.items()))
    print(f"  {'GT':10s}: {gt_str}")


def analyze_question_types(v2_results: dict, test_data: list[dict]) -> None:
    """Analyze accuracy by question type."""
    print("\n" + "=" * 80)
    print("QUESTION TYPE ANALYSIS (QB v2)")
    print("=" * 80)

    # Map video to question type
    video_to_qtype = {}
    video_to_question = {}
    for item in test_data:
        video = item["video"]
        question = item["conversations"][0]["value"]
        video_to_qtype[video] = extract_question_type(question)
        video_to_question[video] = question.split("\n")[2] if "\n" in question else question

    modes = ["real", "shuffled", "shifted", "noise", "silent"]
    for mode in modes:
        if mode not in v2_results:
            continue

        print(f"\n--- {mode.upper()} ---")
        qtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for pred in v2_results[mode]["predictions"]:
            video = pred["video"]
            qtype = video_to_qtype.get(video, "unknown")
            qtype_stats[qtype]["total"] += 1
            if pred["correct"] == "True":
                qtype_stats[qtype]["correct"] += 1

        for qtype in sorted(qtype_stats.keys()):
            s = qtype_stats[qtype]
            acc = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
            print(f"  {qtype:25s}: {s['correct']:2d}/{s['total']:2d} = {acc:5.1f}%")


def statistical_significance(v2_results: dict) -> None:
    """McNemar's test for audio contribution significance."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print("=" * 80)

    if "real" not in v2_results or "noise" not in v2_results:
        print("Missing results for significance test")
        return

    real_preds = {p["index"]: p["correct"] == "True" for p in v2_results["real"]["predictions"]}
    noise_preds = {p["index"]: p["correct"] == "True" for p in v2_results["noise"]["predictions"]}
    silent_preds = {p["index"]: p["correct"] == "True" for p in v2_results["silent"]["predictions"]}

    # McNemar: compare real vs noise
    for comp_name, comp_preds in [("noise", noise_preds), ("silent", silent_preds)]:
        # b = real correct, comp wrong
        # c = real wrong, comp correct
        b = sum(1 for idx in real_preds if real_preds[idx] and not comp_preds.get(idx, False))
        c = sum(1 for idx in real_preds if not real_preds[idx] and comp_preds.get(idx, False))

        print(f"\n  Real vs {comp_name}:")
        print(f"    Real✓ {comp_name}✗ (b): {b}")
        print(f"    Real✗ {comp_name}✓ (c): {c}")

        if b + c > 0:
            # McNemar chi-squared (with continuity correction)
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            # For chi2 with 1 df: p < 0.05 if chi2 > 3.84, p < 0.01 if chi2 > 6.63
            print(f"    McNemar χ² (continuity-corrected): {chi2:.2f}")
            if chi2 > 6.63:
                print("    *** p < 0.01 — HIGHLY SIGNIFICANT ***")
            elif chi2 > 3.84:
                print("    ** p < 0.05 — SIGNIFICANT **")
            else:
                print("    p > 0.05 — NOT SIGNIFICANT")

            # Effect size: odds ratio
            if c > 0:
                odds_ratio = b / c
                print(f"    Odds ratio (b/c): {odds_ratio:.2f}")
            else:
                print("    Odds ratio: ∞ (c=0)")

    # Also report 95% CI for real accuracy using Wilson score
    n = len(real_preds)
    p_hat = sum(real_preds.values()) / n
    z = 1.96  # 95%
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom

    print(
        f"\n  Real accuracy: {100 * p_hat:.1f}% (95% CI: [{100 * (center - margin):.1f}%, {100 * (center + margin):.1f}%])"
    )
    print(f"  N = {n} samples")


def generate_summary_table(v1_results: dict, v2_results: dict) -> None:
    """Generate publication-ready summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Publication-Ready)")
    print("=" * 80)

    modes = ["real", "shuffled", "shifted", "noise", "silent"]
    mode_labels = {
        "real": "Real Audio",
        "shuffled": "Shuffled Audio",
        "shifted": "Time-Shifted Audio",
        "noise": "Gaussian Noise",
        "silent": "Silent (No Audio)",
    }

    print(f"\n{'Audio Condition':<25s} | {'QB v1 (Broken)':>15s} | {'QB v2 (AGTA)':>15s} | {'Δ':>8s}")
    print("-" * 70)
    for mode in modes:
        v1_acc = v1_results[mode]["accuracy"] if mode in v1_results else float("nan")
        v2_acc = v2_results[mode]["accuracy"] if mode in v2_results else float("nan")
        delta = v2_acc - v1_acc

        sign = "+" if delta >= 0 else ""
        print(f"{mode_labels[mode]:<25s} | {v1_acc:>14.1f}% | {v2_acc:>14.1f}% | {sign}{delta:>6.1f}pp")

    # Deltas
    print("-" * 70)
    if "real" in v2_results and "noise" in v2_results:
        delta_rn = v2_results["real"]["accuracy"] - v2_results["noise"]["accuracy"]
        print(f"{'Δ(Real - Noise)':<25s} | {'—':>15s} | {delta_rn:>14.1f}pp |")
    if "real" in v2_results and "silent" in v2_results:
        delta_rs = v2_results["real"]["accuracy"] - v2_results["silent"]["accuracy"]
        print(f"{'Δ(Real - Silent)':<25s} | {'—':>15s} | {delta_rs:>14.1f}pp |")
    if "real" in v2_results and "shuffled" in v2_results:
        delta_rsh = v2_results["real"]["accuracy"] - v2_results["shuffled"]["accuracy"]
        print(f"{'Δ(Real - Shuffled)':<25s} | {'—':>15s} | {delta_rsh:>14.1f}pp |")


def analyze_confusion_patterns(v2_results: dict) -> None:
    """Analyze what the model predicts when audio is wrong."""
    print("\n" + "=" * 80)
    print("CONFUSION PATTERN ANALYSIS")
    print("=" * 80)

    if "real" not in v2_results or "noise" not in v2_results:
        return

    real_preds = {p["index"]: p for p in v2_results["real"]["predictions"]}
    noise_preds = {p["index"]: p for p in v2_results["noise"]["predictions"]}

    # When noise makes model wrong, what does it predict?
    print("\n--- When NOISE breaks correct predictions ---")
    noise_confusion = Counter()
    for idx in real_preds:
        real_correct = real_preds[idx]["correct"] == "True"
        noise_correct = noise_preds[idx]["correct"] == "True"
        if real_correct and not noise_correct:
            gt = real_preds[idx]["gt"]
            noise_pred = noise_preds[idx]["pred"]
            noise_confusion[f"GT={gt}→Pred={noise_pred}"] += 1

    for pattern, count in noise_confusion.most_common(10):
        print(f"  {pattern}: {count}")

    # Does noise cause consistent wrong answers or random?
    noise_wrong_preds = []
    for idx in real_preds:
        if real_preds[idx]["correct"] == "True" and noise_preds[idx]["correct"] != "True":
            noise_wrong_preds.append(noise_preds[idx]["pred"])

    if noise_wrong_preds:
        pred_dist = Counter(noise_wrong_preds)
        print(f"\n  Noise wrong predictions distribution: {dict(pred_dist)}")
        entropy = -sum((c / len(noise_wrong_preds)) * np.log2(c / len(noise_wrong_preds)) for c in pred_dist.values())
        max_entropy = np.log2(len(pred_dist))
        print(f"  Prediction entropy: {entropy:.2f} / {max_entropy:.2f} max")
        print(f"  → {'Random-looking' if entropy > 0.8 * max_entropy else 'Systematic bias'}")


def main() -> None:
    """Run all analyses."""
    print("=" * 80)
    print("AGTA AUDIO ABLATION — COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Load data
    v1_results = load_results("qb")
    v2_results = load_results("qb_v2")
    test_data = load_test_data()

    if not v1_results:
        print("ERROR: No QB v1 results found")
        sys.exit(1)
    if not v2_results:
        print("ERROR: No QB v2 results found")
        sys.exit(1)

    print(f"\nLoaded QB v1 modes: {list(v1_results.keys())}")
    print(f"Loaded QB v2 modes: {list(v2_results.keys())}")
    print(f"Test data samples: {len(test_data)}")

    # 1. Summary table
    generate_summary_table(v1_results, v2_results)

    # 2. Per-sample comparison
    analyze_per_sample(v1_results, v2_results)

    # 3. Audio dependency analysis
    analyze_audio_dependency(v2_results)

    # 4. Question type analysis
    analyze_question_types(v2_results, test_data)

    # 5. Statistical significance
    statistical_significance(v2_results)

    # 6. Confusion patterns
    analyze_confusion_patterns(v2_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
