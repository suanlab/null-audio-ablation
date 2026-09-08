from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval_results"

MODELS = {
    "agta_full": {
        "real": EVAL_DIR / "full_v6_real.json",
        "shuffled": EVAL_DIR / "full_v6_shuffled.json",
        "shifted": EVAL_DIR / "full_v6_shifted.json",
        "noise": EVAL_DIR / "full_v6_noise.json",
        "silent": EVAL_DIR / "full_v6_silent.json",
    },
    "no_bridge": {
        "real": EVAL_DIR / "no_bridge_real.json",
        "shuffled": EVAL_DIR / "no_bridge_shuffled.json",
        "shifted": EVAL_DIR / "no_bridge_shifted.json",
        "noise": EVAL_DIR / "no_bridge_noise.json",
        "silent": EVAL_DIR / "no_bridge_silent.json",
    },
    "vision_only": {
        "real": EVAL_DIR / "vision_only_real.json",
        "shuffled": EVAL_DIR / "vision_only_shuffled.json",
        "shifted": EVAL_DIR / "vision_only_shifted.json",
        "noise": EVAL_DIR / "vision_only_noise.json",
        "silent": EVAL_DIR / "vision_only_silent.json",
    },
}


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(loaded)!r}")
    return cast("dict[str, object]", loaded)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (center - margin, center + margin)


def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def parse_predictions(payload: dict[str, object]) -> dict[str, bool]:
    preds_obj = payload.get("predictions")
    if not isinstance(preds_obj, list):
        raise ValueError("Missing predictions list")
    preds = cast("list[object]", preds_obj)
    result: dict[str, bool] = {}
    for item_obj in preds:
        if not isinstance(item_obj, dict):
            raise ValueError(f"Prediction entry must be dict, got {type(item_obj)!r}")
        item = cast("dict[str, object]", item_obj)
        idx = str(item["index"])
        result[idx] = str(item["correct"]).lower() == "true"
    return result


def analyze_model(name: str, paths: dict[str, Path]) -> dict[str, object]:
    loaded = {mode: load_json(path) for mode, path in paths.items()}
    parsed = {mode: parse_predictions(payload) for mode, payload in loaded.items()}

    base_indices = list(parsed["real"].keys())
    for mode, preds in parsed.items():
        if list(preds.keys()) != base_indices:
            raise ValueError(f"Index mismatch for {name}:{mode}")

    metrics: dict[str, dict[str, object]] = {}
    for mode, payload in loaded.items():
        correct_obj = payload.get("correct")
        total_obj = payload.get("total")
        accuracy_obj = payload.get("accuracy")
        if not isinstance(correct_obj, (int, float)):
            raise ValueError(f"Invalid correct count for {name}:{mode}")
        if not isinstance(total_obj, (int, float)):
            raise ValueError(f"Invalid total count for {name}:{mode}")
        if not isinstance(accuracy_obj, (int, float)):
            raise ValueError(f"Invalid accuracy for {name}:{mode}")
        k = int(correct_obj)
        n = int(total_obj)
        lo, hi = wilson_interval(k, n)
        metrics[mode] = {
            "correct": k,
            "total": n,
            "accuracy": float(accuracy_obj),
            "ci95": [100 * lo, 100 * hi],
        }

    pairwise: dict[str, dict[str, float | int]] = {}
    for mode in ("silent", "shuffled", "shifted", "noise"):
        b = 0
        c = 0
        for idx in base_indices:
            real_ok = parsed["real"][idx]
            mode_ok = parsed[mode][idx]
            if real_ok and not mode_ok:
                b += 1
            elif not real_ok and mode_ok:
                c += 1
        pairwise[mode] = {
            "discordant_real_better": b,
            "discordant_mode_better": c,
            "mcnemar_p": exact_mcnemar_p_value(b, c),
        }

    return {
        "model": name,
        "metrics": metrics,
        "pairwise_vs_real": pairwise,
        "gaps": {
            "real_minus_silent": cast("float", metrics["real"]["accuracy"]) - cast("float", metrics["silent"]["accuracy"]),
            "real_minus_noise": cast("float", metrics["real"]["accuracy"]) - cast("float", metrics["noise"]["accuracy"]),
            "real_minus_shuffled": cast("float", metrics["real"]["accuracy"])
            - cast("float", metrics["shuffled"]["accuracy"]),
        },
    }


def main() -> None:
    output = {name: analyze_model(name, paths) for name, paths in MODELS.items()}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
