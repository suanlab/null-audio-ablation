"""A2 + B1: matched 98-sample re-aggregation with bootstrap CIs and paired tests.

Recomputes the headline AVQA comparison across AGTA (full_v6), No-Bridge, and
Vision-Only on the **exact same 98 samples** that all three were evaluated on,
adds 95% bootstrap confidence intervals (10000 resamples), paired bootstrap
p-values, McNemar's test, and an R-S gap diff-in-diff between models.

Inputs: ``eval_results/{model}_{mode}.json`` for
    model in {full_v6, no_bridge, vision_only}
    mode  in {real, silent, noise, shuffled, shifted}

Outputs:
    eval_results/matched_98/summary_with_ci.json    -- full numeric results
    eval_results/matched_98/headline_table.md       -- paper-ready table

Usage::

    python scripts/aggregate_matched_98.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

RNG_SEED = 2026
N_BOOT = 10_000
MODELS = ("full_v6", "no_bridge", "vision_only", "videollama2", "qwenomni")
MODEL_LABELS = {
    "full_v6": "AGTA (full_v6)",
    "no_bridge": "No-Bridge",
    "vision_only": "Vision-Only",
    "videollama2": "VideoLLaMA2.1-AV (external)",
    "qwenomni": "Qwen2.5-Omni-7B (external)",
}
MODES = ("real", "silent", "noise", "shuffled", "shifted")
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_results"
OUT_DIR = EVAL_DIR / "matched_98"


def load_correct_vec(model: str, mode: str) -> np.ndarray:
    """Return a per-sample boolean correctness vector for (model, mode)."""
    payload = json.loads((EVAL_DIR / f"{model}_{mode}.json").read_text())
    return np.array([p["correct"] == "True" for p in payload["predictions"]], dtype=bool)


def boot_ci(values: np.ndarray, rng: np.random.Generator,
            n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean accuracy (returned in %)."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean() * 100), float(lo * 100), float(hi * 100)


def paired_boot_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                     n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Paired bootstrap of mean(a) - mean(b) on per-sample paired data.

    Returns (point_pp, lo_pp, hi_pp, two_sided_p).
    """
    n = len(a)
    diff = a.astype(int) - b.astype(int)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diff = diff[idx].mean(axis=1) * 100
    lo, hi = np.percentile(boot_diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_pos = float(np.mean(boot_diff <= 0))
    p_neg = float(np.mean(boot_diff >= 0))
    p = 2 * min(p_pos, p_neg)
    return float(diff.mean() * 100), float(lo), float(hi), float(p)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar test (binomial on discordant pairs).

    Uses the exact binomial test rather than the continuity-corrected
    chi-square approximation so the released script reproduces the exact
    p-values reported in the paper (n is small: <=98 paired samples).
    """
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "stat": None, "p": 1.0}
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(2.0 * tail, 1.0)
    return {"n01": n01, "n10": n10, "stat": None, "p": float(p)}


def boot_simple_diff(d: np.ndarray, rng: np.random.Generator,
                     n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Bootstrap CI on the mean of a per-sample difference series."""
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = d[idx].mean(axis=1) * 100
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return float(d.mean() * 100), float(lo), float(hi)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    vecs: dict[tuple[str, str], np.ndarray] = {
        (m, mode): load_correct_vec(m, mode) for m in MODELS for mode in MODES
    }

    # Sanity: all three models on the same 98 keys (assumed; verified upstream)
    n = len(next(iter(vecs.values())))
    assert n == 98, f"Expected 98 samples, got {n}"

    out: dict = {
        "n_samples": n,
        "n_boot": N_BOOT,
        "rng_seed": RNG_SEED,
        "models": {},
        "cross_model_real": {},
        "rs_gap_diff_in_diff": {},
    }

    for m in MODELS:
        model_out = {"label": MODEL_LABELS[m], "accuracy_pct": {}, "gaps_pp": {}}
        real = vecs[(m, "real")]
        for mode in MODES:
            pt, lo, hi = boot_ci(vecs[(m, mode)], rng)
            model_out["accuracy_pct"][mode] = {"point": pt, "ci95_lo": lo, "ci95_hi": hi}
        for other in ("silent", "noise", "shuffled", "shifted"):
            v = vecs[(m, other)]
            gap, glo, ghi, p = paired_boot_diff(real, v, rng)
            mc = mcnemar(real, v)
            model_out["gaps_pp"][f"real_minus_{other}"] = {
                "point_pp": gap, "ci95_lo": glo, "ci95_hi": ghi,
                "boot_p_two_sided": p, "mcnemar": mc,
            }
        out["models"][m] = model_out

    real_agta = vecs[("full_v6", "real")]
    for m in ("no_bridge", "vision_only", "videollama2", "qwenomni"):
        gap, glo, ghi, p = paired_boot_diff(real_agta, vecs[(m, "real")], rng)
        mc = mcnemar(real_agta, vecs[(m, "real")])
        out["cross_model_real"][f"AGTA_minus_{m}"] = {
            "point_pp": gap, "ci95_lo": glo, "ci95_hi": ghi,
            "boot_p_two_sided": p, "mcnemar": mc,
        }

    def rs_diff(model: str) -> np.ndarray:
        return vecs[(model, "real")].astype(int) - vecs[(model, "silent")].astype(int)

    agta_diff = rs_diff("full_v6")
    for label, d in (
        ("AGTA_RS_minus_NoBridge_RS", agta_diff - rs_diff("no_bridge")),
        ("AGTA_RS_minus_VisionOnly_RS", agta_diff - rs_diff("vision_only")),
        ("AGTA_RS_minus_VideoLLaMA2_RS", agta_diff - rs_diff("videollama2")),
        ("AGTA_RS_minus_QwenOmni_RS", agta_diff - rs_diff("qwenomni")),
    ):
        pt, lo, hi = boot_simple_diff(d, rng)
        out["rs_gap_diff_in_diff"][label] = {"point_pp": pt, "ci95_lo": lo, "ci95_hi": hi}

    (OUT_DIR / "summary_with_ci.json").write_text(json.dumps(out, indent=2))

    # --- paper-ready markdown table -----------------------------------------
    md_lines: list[str] = []
    md_lines.append("# Matched 98-sample AVQA — Headline Results\n")
    md_lines.append(f"Bootstrap: {N_BOOT} resamples, seed={RNG_SEED}. "
                    "Brackets show 95% percentile CI.\n")
    md_lines.append("## Per-mode accuracy (%, 95% CI)\n")
    md_lines.append("| Model | Real | Silent | Noise | Shuffled | Shifted |")
    md_lines.append("|---|---|---|---|---|---|")
    for m in MODELS:
        row = [MODEL_LABELS[m]]
        for mode in MODES:
            a = out["models"][m]["accuracy_pct"][mode]
            row.append(f"{a['point']:.1f} [{a['ci95_lo']:.1f}, {a['ci95_hi']:.1f}]")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    md_lines.append("## Paired gaps (real − mode, pp)\n")
    md_lines.append("| Model | R−Silent | R−Noise | R−Shuffled | R−Shifted |")
    md_lines.append("|---|---|---|---|---|")
    for m in MODELS:
        row = [MODEL_LABELS[m]]
        for other in ("silent", "noise", "shuffled", "shifted"):
            g = out["models"][m]["gaps_pp"][f"real_minus_{other}"]
            sig = "*" if g["boot_p_two_sided"] < 0.05 else ""
            row.append(f"{g['point_pp']:+.1f} [{g['ci95_lo']:+.1f}, {g['ci95_hi']:+.1f}]{sig}")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("\n*: paired bootstrap p < 0.05 (two-sided).\n")

    md_lines.append("## R-S gap diff-in-diff (model vs model)\n")
    md_lines.append("| Comparison | Δ R-S (pp) | 95% CI |")
    md_lines.append("|---|---|---|")
    for k, v in out["rs_gap_diff_in_diff"].items():
        md_lines.append(f"| {k} | {v['point_pp']:+.1f} | [{v['ci95_lo']:+.1f}, {v['ci95_hi']:+.1f}] |")
    md_lines.append("")

    md_lines.append("## AGTA vs baselines on REAL only\n")
    md_lines.append("| Comparison | Δ (pp) | 95% CI | Boot p | McNemar p |")
    md_lines.append("|---|---|---|---|---|")
    for k, v in out["cross_model_real"].items():
        md_lines.append(
            f"| {k} | {v['point_pp']:+.1f} | [{v['ci95_lo']:+.1f}, {v['ci95_hi']:+.1f}] | "
            f"{v['boot_p_two_sided']:.4f} | {v['mcnemar']['p']:.4f} |"
        )

    (OUT_DIR / "headline_table.md").write_text("\n".join(md_lines) + "\n")

    print(f"Wrote {OUT_DIR/'summary_with_ci.json'}")
    print(f"Wrote {OUT_DIR/'headline_table.md'}")


if __name__ == "__main__":
    main()
