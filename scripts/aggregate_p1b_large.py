"""P1-B: larger-n (n=548) robustness of the external positive-controlled
dissociation. For VideoLLaMA2.1-AV and video-SALMONN 2+: accuracy per mode,
R-S gap with 95% bootstrap CI (10k, seed 2026) + paired exact two-sided
binomial McNemar, R-shifted McNemar, vs. the matched-98 reference. Pool
overlaps in-house AGTA training, so AGTA is intentionally excluded (off-the-
shelf externals only). Outputs eval_results/large548/summary.{json,md}.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"
OUT = EV / "large548"
REF98 = {"videollama2": 4.1, "vsalm2": 1.0}  # matched-98 R-S (paper headline)
LABEL = {"videollama2": "VideoLLaMA2.1-AV", "vsalm2": "video-SALMONN 2+"}


def vec(p: Path) -> dict[str, int]:
    d = json.loads(p.read_text())["predictions"]
    return {str(x["index"]): int(str(x["correct"]) == "True") for x in d}


def accj(p: Path):
    d = json.loads(p.read_text())
    return d["accuracy"], d["correct"], d["total"]


def mcnemar(a: dict, b: dict):
    k = sorted(set(a) & set(b))
    n10 = sum(1 for i in k if a[i] and not b[i])
    n01 = sum(1 for i in k if b[i] and not a[i])
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) * 0.5 ** n) if n else 1.0
    return n10, n01, p


def boot(a: dict, b: dict, seed: int = 2026, nb: int = 10000):
    k = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in k], float)
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), (nb, len(d)))].mean(1) * 100
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), len(k)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"pool": "avqa_eval_large.jsonl", "n_pool": 548,
                 "note": "overlaps in-house AGTA training; off-the-shelf externals only",
                 "models": {}, "missing": []}
    for m in ("videollama2", "vsalm2"):
        fr = EV / f"large548_{m}_real.json"
        fs = EV / f"large548_{m}_silent.json"
        fsh = EV / f"large548_{m}_shifted.json"
        for f in (fr, fs, fsh):
            if not f.exists():
                res["missing"].append(f.name)
        if not (fr.exists() and fs.exists()):
            continue
        real_vec, silent_vec = vec(fr), vec(fs)
        ar, sr = accj(fr), accj(fs)
        rs = round(ar[0] - sr[0], 2)
        b, c, p = mcnemar(real_vec, silent_vec)
        lo, hi, n = boot(real_vec, silent_vec)
        rec = {"label": LABEL[m], "n_eff": n,
               "real": round(ar[0], 2), "real_c": ar[1], "real_t": ar[2],
               "silent": round(sr[0], 2), "silent_c": sr[1], "silent_t": sr[2],
               "RS_pp": rs, "RS_ci95": [round(lo, 2), round(hi, 2)],
               "RS_mcnemar_b": b, "RS_mcnemar_c": c, "RS_mcnemar_p": round(p, 6),
               "RS_matched98_ref_pp": REF98[m]}
        if fsh.exists():
            shuf_vec = vec(fsh)
            ash = accj(fsh)
            bb, cc, pp = mcnemar(real_vec, shuf_vec)
            rec.update({"shifted": round(ash[0], 2),
                        "Rsh_pp": round(ash[0] - ar[0], 2),
                        "Rshift_mcnemar_p": round(pp, 6)})
        res["models"][m] = rec

    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    lines = ["# P1-B larger-n (n=548) external dissociation robustness\n",
         f"Pool: avqa_eval_large.jsonl (548). {res['note']}.",
         "R-S = real-silent; CI = 95% bootstrap (10k, seed 2026); "
         "p = paired exact two-sided binomial McNemar.\n",
         "| model | n | real% | silent% | R-S pp | 95% CI | McNemar p | matched-98 R-S |",
         "|---|---|---|---|---|---|---|---|"]
    for _m, r in res["models"].items():
        lines.append(f"| {r['label']} | {r['n_eff']} | {r['real']:.2f} | {r['silent']:.2f} | "
                 f"{r['RS_pp']:+.2f} | [{r['RS_ci95'][0]:+.1f},{r['RS_ci95'][1]:+.1f}] | "
                 f"{r['RS_mcnemar_p']:.4f} | {r['RS_matched98_ref_pp']:+.1f} |")
    if res["missing"]:
        lines.append(f"\n_Missing (pending): {', '.join(res['missing'])}_")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("missing:", res["missing"] if res["missing"] else "none")
    for _m, r in res["models"].items():
        print(f"{r['label']}: n={r['n_eff']} real={r['real']:.2f} silent={r['silent']:.2f} "
              f"R-S={r['RS_pp']:+.2f}pp CI[{r['RS_ci95'][0]:+.1f},{r['RS_ci95'][1]:+.1f}] "
              f"McNemar p={r['RS_mcnemar_p']:.4f} (matched-98 ref {r['RS_matched98_ref_pp']:+.1f}pp)"
              + (f" | shifted={r.get('shifted')} R-sh={r.get('Rsh_pp'):+.2f} "
                 f"p={r.get('Rshift_mcnemar_p')}" if 'shifted' in r else ""))


if __name__ == "__main__":
    main()
