"""Delivery gain ``G_A`` on the held-out MUSIC-AVQA precheck cells, for every model.

``G_A`` is the eligibility gate: on a blank video, an item answerable from audio alone must
move when real audio replaces silence. It is the quantity that disqualified the AVUT domain
for both reported models, and it is computed here with the same video-cluster percentile
bootstrap the manuscript uses everywhere else (10,000 resamples, seed 2027), so a number
printed here is directly comparable with a number in a table.

Usage:
    python scripts/delivery_gain.py                 # every model with both cells present
    python scripts/delivery_gain.py qwenomni        # one model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from videollm.cmss import BOOTSTRAP_SEED, N_BOOT, cluster_bootstrap_mean  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "eval_results"
MODELS = ("vsalm2", "vl2", "qwenomni")


def _load(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    records = payload["predictions"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"{path.name}: expected a list of predictions, got {type(records).__name__}")
    return records


def gain(model: str) -> tuple[float, float, float, int, int, int] | None:
    """Return ``(G_A, lo, hi, n_items, n_videos, n_unparsed)`` in pp, or None if a cell is missing."""
    real_path = RESULTS / f"pre_music_{model}_blank_real.json"
    silent_path = RESULTS / f"pre_music_{model}_blank_silent.json"
    if not (real_path.exists() and silent_path.exists()):
        return None
    real, silent = _load(real_path), _load(silent_path)
    if len(real) != len(silent):
        raise ValueError(f"{model}: {len(real)} real vs {len(silent)} silent items are not paired")

    videos = [str(r["video"]) for r in real]
    if videos != [str(s["video"]) for s in silent]:
        raise ValueError(f"{model}: the two cells do not cover the same items in the same order")

    def correct(records: list[dict[str, object]]) -> np.ndarray:
        return np.array([str(r["correct"]).lower() == "true" for r in records], dtype=np.float64)

    diff = correct(real) - correct(silent)  # (N,) in {-1, 0, 1}, item-paired
    clusters = np.array(videos)
    point, lo, hi = cluster_bootstrap_mean(diff * 100.0, clusters, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
    unparsed = sum(1 for r in (*real, *silent) if not str(r["pred"]).strip())
    return point, lo, hi, len(real), int(np.unique(clusters).size), unparsed


def main() -> None:
    wanted = sys.argv[1:] or list(MODELS)
    print("Delivery gain G_A = (blank video, real audio) - (blank video, silent)")
    print(f"video-cluster percentile bootstrap, {N_BOOT:,} resamples, seed {BOOTSTRAP_SEED}\n")
    print(f"{'model':12} {'G_A (pp)':>10}  {'95% CI':>20}  {'items':>6} {'videos':>7} {'unparsed':>9}  gate")
    print("-" * 86)
    for model in wanted:
        result = gain(model)
        if result is None:
            print(f"{model:12} {'--':>10}  {'(cells not present)':>20}")
            continue
        point, lo, hi, n, c, unparsed = result
        passes = "PASS" if lo > 0 else "FAIL"
        print(f"{model:12} {point:+10.2f}  [{lo:+8.2f}, {hi:+8.2f}]  {n:6} {c:7} {unparsed:9}  {passes}")
    print("\nPASS means the 95% CI excludes zero from above: audio alone measurably answers")
    print("items that silence does not. A model that fails this gate is not disqualified from")
    print("the paper -- it is a delivery result, and is reported as one.")


if __name__ == "__main__":
    main()
