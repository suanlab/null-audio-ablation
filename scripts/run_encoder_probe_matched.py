"""Instrument probe with all three audio encoders on byte-identical audio (added 2026-11-09).

The probe in ``run_encoder_probe.py`` and ``run_encoder_probe_qwen.py`` did not give the encoders
the same audio. Whisper's feature extractor truncates to the first 30 s, Qwen's tower was fed the
first 30 s, and BEATs got the whole clip -- about 60 s for almost every MUSIC-AVQA video. So
54.8 vs 35.7 % compared encoders on different inputs. Here every encoder gets the first 30 s of
the same decoded waveform, and the script keeps per-clip out-of-fold predictions on shared folds,
so that differences between encoders come with paired intervals rather than fold SDs.

It also reports the probe on a label-audited subset. Each clip's label is the answer to the first
instrument-valued question about it, and many of those questions are visual ("which instrument is
on the left?"). A label taken from a question that names a sound is more likely to be audible, so
the subset keeps clips whose label question's type includes "Audio".

Usage::

    CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/run_encoder_probe_matched.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
SR = 16000
SECONDS = 30
SEED = 2027
N_BOOT = 10_000


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Matched-input three-encoder instrument probe")
    p.add_argument("--limit", type=int, default=1319)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=str,
                   default=str(REPO_ROOT / "eval_results" / "encoder_probe_instrument_matched.json"))
    return p.parse_args()


def label_question_types(paths: list[str]) -> list[str]:
    """Question type of the record each clip's label was taken from (first instrument answer)."""
    from encoder_probe import INSTRUMENTS, MUSIC_SOURCE

    first: dict[str, str] = {}
    for record in json.loads(MUSIC_SOURCE.read_text()):
        if record.get("question_deleted"):
            continue
        name = f"{record['video_id']}.mp4"
        if str(record["anser"]).strip() in INSTRUMENTS and name not in first:
            first[name] = str(record.get("type", ""))
    return [first.get(Path(p).name, "") for p in paths]


def oof_predictions(features: np.ndarray, labels: np.ndarray, folds: int) -> np.ndarray:
    """Out-of-fold predictions from the same pipeline and folds as ``run_encoder_probe.probe``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return cross_val_predict(clf, features, labels, cv=cv, n_jobs=folds)


def bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and percentile 95 % CI over clips (one clip per video, so clips are the unit)."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    """Extract three encoders on the first 30 s, probe, and compare them clip by clip."""
    args = parse_args()
    import run_encoder_probe as rep
    import run_encoder_probe_qwen as rq
    from encoder_probe import instrument_labels

    items = instrument_labels(args.limit)
    paths = [p for p, _ in items]
    names = sorted({y for _, y in items})
    labels = np.array([names.index(y) for _, y in items])
    majority = float(np.bincount(labels).max() / len(labels) * 100)
    print(f"items={len(items)} classes={len(names)} majority={majority:.1f}%", flush=True)

    # One decode, one window, reused by every encoder: patch the shared loader to return the
    # first 30 s of a cached waveform, which all three feature functions call.
    cache: dict[str, np.ndarray] = {}
    full_load = rep.load_audio

    def first_window(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = full_load(path)[: SECONDS * SR]
        return cache[path]

    setattr(rep, "load_audio", first_window)  # noqa: B010 -- patch the module attribute the extractors read
    extractors = {"BEATs": rep.beats_features, "Whisper-large-v3": rep.whisper_features,
                  "Qwen2.5-Omni audio tower": rq.qwen_features}

    correct: dict[str, np.ndarray] = {}
    results: dict[str, object] = {
        "items": len(items), "classes": len(names), "majority_baseline": majority,
        "task": "instrument identity (20-way)", "audio": f"first {SECONDS} s of each clip, all encoders",
        "seconds_fed": {"median": float(np.median([len(first_window(p)) / SR for p in paths])),
                        "clips_shorter_than_window": int(sum(len(first_window(p)) < SECONDS * SR for p in paths))},
    }
    encoders: dict[str, object] = {}
    results["encoders"] = encoders
    for name, fn in extractors.items():
        print(f"extracting {name} ...", flush=True)
        feats = fn(paths, args.device)
        pred = oof_predictions(feats, labels, args.folds)
        correct[name] = (pred == labels).astype(float)
        m, lo, hi = bootstrap(correct[name] * 100)
        encoders[name] = {"mean": m, "ci": [lo, hi], "dim": int(feats.shape[1])}
        print(f"  {name}: {m:.1f}% [{lo:.1f}, {hi:.1f}]", flush=True)

    pairs = [("BEATs", "Whisper-large-v3"), ("Qwen2.5-Omni audio tower", "Whisper-large-v3"),
             ("BEATs", "Qwen2.5-Omni audio tower")]
    results["paired_differences"] = {
        f"{a} - {b}": dict(zip(("mean", "lo", "hi"), bootstrap((correct[a] - correct[b]) * 100), strict=True))
        for a, b in pairs
    }
    types = label_question_types(paths)
    audible = np.array(["Audio" in t for t in types])
    results["label_audit"] = {
        "label_question_types": {t: types.count(t) for t in sorted(set(types))},
        "audio_typed_subset": {
            "items": int(audible.sum()),
            **{name: dict(zip(("mean", "lo", "hi"), bootstrap(c[audible] * 100), strict=True))
               for name, c in correct.items()},
        },
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results["paired_differences"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
