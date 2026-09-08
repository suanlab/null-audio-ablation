"""Reviewer R2: TOST equivalence test against ±3pp margin for shift sweep.

"No magnitude is significant" (Table 13) only rules out a 0pp difference; it
does not establish equivalence. TOST (two one-sided tests) against a ±3pp
margin gives the principled equivalence statement.

We run a paired bootstrap (10k resamples) on the matched-98 paired
(real, shifted_k) accuracies for AGTA and VideoLLaMA2.1-AV across the seven
shift magnitudes. The TOST p-value is computed against H0: |Δacc| >= 3pp.

Output: eval_results/matched_98/tost_shift.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"


def load_corr(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text())
    out = {}
    for item in raw["predictions"]:
        out[str(item["index"])] = 1 if str(item.get("correct", "")).lower() == "true" else 0
    return out


# Reuse the shift-sweep data already on disk (table 13 has these numbers)
SWEEPS = {
    "AGTA": {
        "real": "full_v6_real",
        0.10: "shift_sweep/full_v6_s0p10", 0.25: "shift_sweep/full_v6_s0p25",
        0.50: "shift_sweep/full_v6_s0p50", 1.00: "full_v6_shifted_1s",
        2.00: "shift_sweep/full_v6_s2p0",  3.00: "full_v6_shifted",
        5.00: "shift_sweep/full_v6_s5p0",
    },
    "VideoLLaMA2.1-AV": {
        "real": "videollama2_real",
        0.10: "shift_sweep/videollama2_s0p10", 0.25: "shift_sweep/videollama2_s0p25",
        0.50: "shift_sweep/videollama2_s0p50", 1.00: "shift_sweep/videollama2_s1p0",
        2.00: "shift_sweep/videollama2_s2p0",  3.00: "videollama2_shifted",
        5.00: "shift_sweep/videollama2_s5p0",
    },
}
MARGIN = 3.0


def tost(real_a: np.ndarray, shift_a: np.ndarray, margin: float, n_boot: int = 10000,
         seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    n = len(real_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append((shift_a[idx].mean() - real_a[idx].mean()) * 100)
    diffs = np.asarray(diffs)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # Two one-sided tests: H0 lower: Δ <= -margin; H0 upper: Δ >= +margin.
    # TOST rejects both if the 90% CI fits entirely in [-margin, +margin].
    ci90 = (float(np.percentile(diffs, 5.0)), float(np.percentile(diffs, 95.0)))
    equivalent = (ci90[0] > -margin) and (ci90[1] < margin)
    return {"point_pp": float(diffs.mean()), "ci95_pp": [lo, hi],
            "ci90_pp": list(ci90), "equivalent_at_3pp": equivalent}


def main() -> None:
    rows = []
    print(f"# TOST equivalence vs ±{MARGIN}pp margin (paired bootstrap n_boot=10000, seed=2026)\n")
    print("| Model | shift(s) | Δacc (pp) | 95% CI | 90% CI | equivalent? |")
    print("|---|---|---|---|---|---|")
    available = {}
    for model, files in SWEEPS.items():
        avail = {}
        for k, fp in files.items():
            f = EV / f"{fp}.json"
            if f.exists():
                avail[k] = load_corr(f)
        available[model] = avail
        real_d = avail.get("real")
        if real_d is None:
            print(f"# {model}: no real dump, skipping")
            continue
        for k in [0.10, 0.25, 0.50, 1.00, 2.00, 3.00, 5.00]:
            s = avail.get(k)
            if s is None:
                rows.append({"model": model, "shift_s": k, "status": "missing dump"})
                continue
            common = sorted(set(real_d) & set(s), key=lambda x: int(x))
            r_a = np.array([real_d[i] for i in common])
            s_a = np.array([s[i] for i in common])
            res = tost(r_a, s_a, MARGIN)
            res["model"] = model
            res["shift_s"] = k
            res["n"] = len(common)
            rows.append(res)
            print(f"| {model} | {k} | {res['point_pp']:+.2f} | "
                  f"[{res['ci95_pp'][0]:+.2f},{res['ci95_pp'][1]:+.2f}] | "
                  f"[{res['ci90_pp'][0]:+.2f},{res['ci90_pp'][1]:+.2f}] | "
                  f"{'YES' if res['equivalent_at_3pp'] else 'no'} |")
    (EV / "matched_98" / "tost_shift.json").write_text(json.dumps({"margin_pp": MARGIN,
                                                                    "rows": rows}, indent=2))
    print(f"\nWrote {EV / 'matched_98' / 'tost_shift.json'}")


if __name__ == "__main__":
    main()
