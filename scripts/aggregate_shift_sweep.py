"""P1-C: aggregate the finer shift-magnitude sweep.

For AGTA (full_v6) and VideoLLaMA2.1-AV (videollama2), for each shift
magnitude, report accuracy on the matched 98-subset with a 95% bootstrap CI
(10k resamples, seed=2026) and a PAIRED exact two-sided binomial McNemar test
against that model's own real-audio predictions (joined by sample index).

Canonical audited JSONs are reused for points that already exist (AGTA
real/1.0/3.0; VL2 real/3.0); the new magnitudes come from
eval_results/shift_sweep/. Outputs:
  eval_results/shift_sweep/summary.json   (machine-checked by audit_claims.py)
  eval_results/shift_sweep/summary.md     (human table)
Missing condition files are skipped with a note so the script can be re-run
as the sweep fills in.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"
SW = EV / "shift_sweep"
GRID = [0.10, 0.25, 0.50, 1.0, 2.0, 3.0, 5.0]

# (model, shift_seconds) -> json path. None key = the model's real baseline.
PATHS: dict[str, dict] = {
    "full_v6": {
        "real": EV / "full_v6_real.json",
        0.10: SW / "full_v6_s0p10.json",
        0.25: SW / "full_v6_s0p25.json",
        0.50: SW / "full_v6_s0p50.json",
        1.0: EV / "full_v6_shifted_1s.json",
        2.0: SW / "full_v6_s2p0.json",
        3.0: EV / "full_v6_shifted.json",
        5.0: SW / "full_v6_s5p0.json",
    },
    "videollama2": {
        "real": EV / "videollama2_real.json",
        0.10: SW / "videollama2_s0p10.json",
        0.25: SW / "videollama2_s0p25.json",
        0.50: SW / "videollama2_s0p50.json",
        1.0: SW / "videollama2_s1p0.json",
        2.0: SW / "videollama2_s2p0.json",
        3.0: EV / "videollama2_shifted.json",
        5.0: SW / "videollama2_s5p0.json",
    },
}
LABEL = {"full_v6": "AGTA (full_v6)", "videollama2": "VideoLLaMA2.1-AV"}


def load_vec(p: Path) -> dict[str, int]:
    """index(str) -> 1/0 correctness."""
    d = json.loads(p.read_text())["predictions"]
    return {str(x["index"]): int(str(x["correct"]) == "True") for x in d}


def boot_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    n = len(vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = vals[idx].mean(axis=1) * 100.0
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def exact_mcnemar(real: dict[str, int], cond: dict[str, int]) -> tuple[int, int, float]:
    """Paired exact two-sided binomial McNemar on shared indices."""
    keys = sorted(set(real) & set(cond))
    b = sum(1 for k in keys if real[k] == 1 and cond[k] == 0)
    c = sum(1 for k in keys if real[k] == 0 and cond[k] == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return b, c, min(1.0, p)


def main() -> None:
    SW.mkdir(parents=True, exist_ok=True)
    out: dict = {"grid_seconds": GRID, "n_boot": 10000, "seed": 2026, "models": {}}
    missing: list[str] = []

    for model, pmap in PATHS.items():
        real_p = pmap["real"]
        if not real_p.exists():
            missing.append(f"{model}:real ({real_p.name})")
            continue
        real = load_vec(real_p)
        keys = sorted(real)
        rec: dict = {"label": LABEL[model], "n": len(keys),
                     "real_acc": round(100.0 * np.mean([real[k] for k in keys]), 4),
                     "points": {}}
        for s in GRID:
            fp = pmap[s]
            if not fp.exists():
                missing.append(f"{model}:{s}s ({fp.name})")
                continue
            cond = load_vec(fp)
            shared = sorted(set(real) & set(cond))
            acc_vec = np.array([cond[k] for k in shared], dtype=float)
            acc = round(100.0 * acc_vec.mean(), 4)
            lo, hi = boot_ci(acc_vec)
            b, c, p = exact_mcnemar(real, cond)
            rec["points"][f"{s:g}"] = {
                "shift_seconds": s, "n": len(shared),
                "accuracy": acc, "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "delta_vs_real_pp": round(acc - rec["real_acc"], 4),
                "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p, 6),
                "source": fp.name,
            }
        out["models"][model] = rec

    out["missing"] = missing
    (SW / "summary.json").write_text(json.dumps(out, indent=2))

    lines = ["# Shift-magnitude sweep (matched 98-subset)\n",
             "Paired exact two-sided binomial McNemar vs. each model's real audio. "
             "CI = 95% bootstrap (10k, seed 2026).\n"]
    for _model, rec in out["models"].items():
        lines.append(f"\n## {rec['label']}  (real={rec['real_acc']:.2f}%, n={rec['n']})\n")
        lines.append("| shift (s) | acc % | 95% CI | Δ vs real (pp) | McNemar p | b/c |")
        lines.append("|---|---|---|---|---|---|")
        for s in GRID:
            pt = rec["points"].get(f"{s:g}")
            if not pt:
                lines.append(f"| {s:g} | _pending_ | | | | |")
                continue
            lines.append(
                f"| {s:g} | {pt['accuracy']:.2f} | "
                f"[{pt['ci95_lo']:.1f}, {pt['ci95_hi']:.1f}] | "
                f"{pt['delta_vs_real_pp']:+.2f} | {pt['mcnemar_p']:.4f} | "
                f"{pt['mcnemar_b']}/{pt['mcnemar_c']} |")
    if missing:
        lines.append(f"\n_Missing (pending): {', '.join(missing)}_")
    (SW / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", SW / "summary.json", "and summary.md")
    print("missing:", missing if missing else "none")
    for _model, rec in out["models"].items():
        print(f"\n{rec['label']} real={rec['real_acc']:.2f}%")
        for s in GRID:
            pt = rec["points"].get(f"{s:g}")
            if pt:
                print(f"  {s:g}s acc={pt['accuracy']:.2f}% "
                      f"Δ={pt['delta_vs_real_pp']:+.2f}pp p={pt['mcnemar_p']:.4f}")


if __name__ == "__main__":
    main()
