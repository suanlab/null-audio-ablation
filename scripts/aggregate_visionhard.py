"""T1.2 modified: Music-AVQA-VisionHard analogue (no new GPU runs).

doesaudiomatter2025 builds Music-AVQA-Hard by filtering items a single-frame
VLM (GPT-4o on the middle frame) can solve. We construct a stricter
in-house analogue without API access: filter the matched-98 subset to the
items our **Vision-Only** baseline answers WRONG. Vision-Only is a strictly
weaker probe than a single-frame VLM (it has the full video at 50% accuracy,
i.e. random on the 4-way task), so the items it gets wrong are at minimum
items that vision-alone cannot reliably solve.

For each model we then recompute the R-S gap on this VisionHard subset
(typically n ~= 49 = 98 - vision-only-correct). If the headline dissociation
holds on the hardened items, the small video-present R-S is not driven by
visually-easy items only.

Outputs:  eval_results/matched_98/visionhard.json
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"
OUT = EV / "matched_98" / "visionhard.json"

# Build VisionHard index set: items where Vision-Only is WRONG on real audio
vo = json.loads((EV / "vision_only_real.json").read_text())["predictions"]
HARD = sorted({int(x["index"]) for x in vo if x["correct"] != "True"})
print(f"VisionHard subset: {len(HARD)} / 98 items where Vision-Only is wrong")

MODELS = {
    "AGTA":              "full_v6",
    "VideoLLaMA2.1-AV":  "videollama2",
    "video-SALMONN 2+":  "vsalm2",
    "Qwen2.5-Omni":      "qwenomni",
}


def vec(fp: str) -> dict[str, int]:
    d = json.loads((EV / f"{fp}.json").read_text())["predictions"]
    return {str(x["index"]): int(x["correct"] == "True")
            for x in d if int(x["index"]) in set(HARD)}


def mcnemar(a, b):
    k = sorted(set(a) & set(b))
    n10 = sum(1 for i in k if a[i] and not b[i])
    n01 = sum(1 for i in k if b[i] and not a[i])
    n = n10 + n01
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) * 0.5 ** n) if n else 1.0


def boot_ci(a, b, seed=2026, nb=10000):
    k = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in k], float)
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), (nb, len(d)))].mean(1) * 100
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), len(k)


def main() -> None:
    rows = []
    for name, fp in MODELS.items():
        real_vec, silent_vec = vec(f"{fp}_real"), vec(f"{fp}_silent")
        keys = sorted(set(real_vec) & set(silent_vec))
        r_acc = 100.0 * sum(real_vec[i] for i in keys) / len(keys)
        s_acc = 100.0 * sum(silent_vec[i] for i in keys) / len(keys)
        rs = r_acc - s_acc
        p = mcnemar(real_vec, silent_vec)
        lo, hi, n_eff = boot_ci(real_vec, silent_vec)
        rows.append({
            "model": name, "n_eff": n_eff,
            "real": round(r_acc, 2), "silent": round(s_acc, 2),
            "rs_pp": round(rs, 2),
            "ci95": [round(lo, 1), round(hi, 1)],
            "mcnemar_p": round(p, 4),
        })
    out = {"n": len(HARD),
           "subset": "VisionHard (Vision-Only wrong on real audio)",
           "indices": HARD, "rows": rows}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}\n")
    print(f"{'Model':22s} | n_eff | real% | silent% | R-S pp | 95% CI | McNemar p")
    print("-" * 88)
    for r in rows:
        print(f"{r['model']:22s} | {r['n_eff']:5d} | {r['real']:5.2f} | {r['silent']:6.2f} | "
              f"{r['rs_pp']:+6.2f} | [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}] | {r['mcnemar_p']:.4f}")


if __name__ == "__main__":
    main()
