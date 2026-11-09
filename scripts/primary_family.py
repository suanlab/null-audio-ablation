"""The preregistered primary family, recomputed from the per-item cells (added 2026-11-09).

Preregistration Sec. 4 fixes the family: per model x domain, the SG > 0 test (silent null,
pooled visual families) and the delivery control (blank-video real vs silent), plus
tau_switch existence. No cell yields a switching threshold, so that arm has no statistic. The
family was registered for the two original systems. The amendment of 2026-08-16 then declared
both systems unvalidated on AVUT and ruled that no AVUT cell enters any confirmatory family, so
the registered family is the 2 x 2 = 4 MUSIC-AVQA tests. The 8-test version with the AVUT cells
is computed too, as a labelled sensitivity analysis. The third system arrived by amendment and
its tests stay outside both.

Raw p is a one-sided cluster-bootstrap p for a positive effect: (1 + #draws <= 0) / (1 + B),
B = 10,000 resamples of whole videos, seed 2027 -- so the smallest reportable p is ~1e-4.
Holm-Bonferroni at alpha = 0.05 over the eight. Writes eval_results/primary_family_holm.json.

Usage::

    $VENV/bin/python scripts/primary_family.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_crossover_figures import EVAL_DIR, N_BOOT, SEED, _paired_vector, _pooled_severe  # noqa: E402

ALPHA = 0.05
FAMILY = (("music", "vl2", "VideoLLaMA2.1-AV"), ("music", "vsalm2", "video-SALMONN 2+"),
          ("avut", "vl2", "VideoLLaMA2.1-AV"), ("avut", "vsalm2", "video-SALMONN 2+"))


def bootstrap_p(values: np.ndarray, cid: np.ndarray) -> tuple[float, float]:
    """Point estimate and one-sided cluster-bootstrap p for mean(values) > 0."""
    rng = np.random.default_rng(SEED)
    n_clusters = int(cid.max()) + 1
    sums = np.bincount(cid, weights=values, minlength=n_clusters)
    counts = np.bincount(cid, minlength=n_clusters).astype(float)
    pick = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
    draws = sums[pick].sum(1) / counts[pick].sum(1)
    return float(values.mean()), float((1 + (draws <= 0).sum()) / (1 + N_BOOT))


def main() -> None:
    """Compute the eight registered tests and apply Holm-Bonferroni."""
    tests: list[dict[str, object]] = []
    for domain, model, name in FAMILY:
        tag = f"cmss_{domain}_{model}"
        deliv, cid = _paired_vector(EVAL_DIR / f"poscontrol_{tag}_real.json",
                                    EVAL_DIR / f"poscontrol_{tag}_silent.json")
        severe, cid_s = _pooled_severe(tag)
        clean, cid_c = _paired_vector(EVAL_DIR / f"avqa_{tag}_real.json", EVAL_DIR / f"avqa_{tag}_silent.json")
        if not (np.array_equal(cid, cid_s) and np.array_equal(cid, cid_c)):
            raise ValueError(f"{tag}: cells are not item-aligned")
        for test, values in (("delivery control", deliv), ("SG > 0", severe - clean)):
            est, p = bootstrap_p(values, cid)
            tests.append({"system": name, "domain": domain, "test": test, "estimate_pp": est, "raw_p": p})
    out: dict[str, object] = {"alpha": ALPHA, "n_boot": N_BOOT, "seed": SEED}
    for label, members in (("registered", [t for t in tests if t["domain"] == "music"]),
                           ("sensitivity_with_avut", tests)):
        family = [dict(t) for t in members]
        pvals = np.array([t["raw_p"] for t in family], dtype=float)
        still_rejecting = True
        for rank, i in enumerate(np.argsort(pvals, kind="stable")):
            threshold = ALPHA / (len(family) - rank)
            still_rejecting = still_rejecting and bool(pvals[i] <= threshold)
            family[i]["holm_threshold"] = threshold
            family[i]["holm_reject"] = still_rejecting
        rejections = int(sum(bool(t["holm_reject"]) for t in family))
        out[label] = {"family_size": len(family), "rejections": rejections, "tests": family}
        print(f"{label}: {rejections} of {len(family)} rejected")
        for t in family:
            print(f"  {t['system']:18s} {t['domain']:5s} {t['test']:16s} {t['estimate_pp']:+7.2f} pp  "
                  f"p={t['raw_p']:.4f}  reject={t['holm_reject']}")
    path = EVAL_DIR / "primary_family_holm.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
