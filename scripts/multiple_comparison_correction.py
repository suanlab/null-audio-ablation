"""Rebuttal deliverable: multiple-comparison correction over the paired-test family.

Reviewer Vev1 asks whether the reused n=98 (and n=47 positive control) sample,
across a large number of paired significance tests, requires correction for
multiple comparisons. This script answers that directly.

It assembles the *complete pre-specified family* of exact-McNemar paired tests
that the paper computes on the shared samples, then applies both
Holm-Bonferroni (family-wise error rate, FWER) and Benjamini-Hochberg (false
discovery rate, FDR) at alpha=0.05.

Two families are corrected separately because they are two different samples and
two different confirmatory questions:

    * Family A -- matched-98 subset: every ``real vs {silent,noise,shuffled,
      shifted}`` paired test for all five models (20 tests) plus the four
      cross-model ``real`` comparisons (24 tests total).
    * Family B -- positive control, n=47 blanked-video: ``real vs blanked-silent``
      for the four models with a positive control run (4 tests).

A pooled correction over all 28 tests is also reported so the answer holds even
under the most conservative "one big family" reading of the reviewer's concern.

Interpretation note: multiple-comparison correction guards against *false
positives* (spuriously significant discoveries). The paper's null/invariance
claims (Vision-Only flat, Qwen2.5-Omni audio-invariant, external near-zero
audio dependence with video present) are claims that a test is *non*-significant;
correction only makes non-significance easier to obtain, so it cannot threaten
those claims. The tests that must survive correction are the positive audio-
grounding effects; this script shows they do.

Inputs:
    eval_results/matched_98/summary_with_ci.json   (Family A exact-McNemar p's)
    eval_results/poscontrol_{model}_{real,silent}.json  (Family B, n=47)

Outputs:
    eval_results/matched_98/multiple_comparison.json
    eval_results/matched_98/multiple_comparison.md

Usage::

    python scripts/multiple_comparison_correction.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_results"
MATCHED_DIR = EVAL_DIR / "matched_98"
ALPHA = 0.05

# Models with a positive-control (video-blanked) run at n=47.
POSCONTROL_MODELS = ("agta", "videollama2", "vsalm2", "qwenomni")
POSCONTROL_LABELS = {
    "agta": "AGTA (in-house)",
    "videollama2": "VideoLLaMA2.1-AV (external)",
    "vsalm2": "video-SALMONN 2+ (external)",
    "qwenomni": "Qwen2.5-Omni-7B (external)",
}


def mcnemar_exact_p(n01: int, n10: int) -> float:
    """Two-sided exact-binomial McNemar p-value on discordant pairs.

    Matches ``scripts/aggregate_matched_98.py`` so numbers reconcile exactly.
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(2.0 * tail, 1.0)


def load_poscontrol_pair(model: str) -> tuple[int, int, float, float]:
    """Return (n01, n10, acc_real_pct, acc_silent_pct) aligned by sample index."""
    real = json.loads((EVAL_DIR / f"poscontrol_{model}_real.json").read_text())["predictions"]
    silent = json.loads((EVAL_DIR / f"poscontrol_{model}_silent.json").read_text())["predictions"]
    real_by_idx = {p["index"]: str(p["correct"]) == "True" for p in real}
    silent_by_idx = {p["index"]: str(p["correct"]) == "True" for p in silent}
    keys = sorted(set(real_by_idx) & set(silent_by_idx))
    n01 = sum(1 for k in keys if (not real_by_idx[k]) and silent_by_idx[k])
    n10 = sum(1 for k in keys if real_by_idx[k] and (not silent_by_idx[k]))
    acc_real = 100.0 * sum(real_by_idx[k] for k in keys) / len(keys)
    acc_silent = 100.0 * sum(silent_by_idx[k] for k in keys) / len(keys)
    return n01, n10, acc_real, acc_silent


def holm(tests: list[dict], alpha: float = ALPHA) -> None:
    """Annotate each test in-place with Holm-Bonferroni reject/threshold."""
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i]["p"])
    still_rejecting = True
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        tests[i]["holm_threshold"] = thr
        reject = still_rejecting and tests[i]["p"] <= thr
        tests[i]["holm_reject"] = reject
        if not reject:
            still_rejecting = False  # once we fail to reject, all later fail too


def benjamini_hochberg(tests: list[dict], alpha: float = ALPHA) -> None:
    """Annotate each test in-place with BH-FDR reject/threshold."""
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i]["p"])
    # Largest rank whose p <= (rank/m)*alpha; reject that and all smaller.
    max_reject_rank = -1
    for rank, i in enumerate(order, start=1):
        thr = rank / m * alpha
        tests[i]["bh_threshold"] = thr
        if tests[i]["p"] <= thr:
            max_reject_rank = rank
    for rank, i in enumerate(order, start=1):
        tests[i]["bh_reject"] = rank <= max_reject_rank


def build_family_a() -> list[dict]:
    summary = json.loads((MATCHED_DIR / "summary_with_ci.json").read_text())
    tests: list[dict] = []
    for model, mdata in summary["models"].items():
        for mode in ("silent", "noise", "shuffled", "shifted"):
            g = mdata["gaps_pp"][f"real_minus_{mode}"]
            tests.append({
                "family": "A_matched98",
                "name": f"{model}: real vs {mode}",
                "gap_pp": g["point_pp"],
                "n01": g["mcnemar"]["n01"],
                "n10": g["mcnemar"]["n10"],
                "p": g["mcnemar"]["p"],
                "supports_positive_claim": model in ("full_v6", "no_bridge")
                and mode in ("silent", "noise", "shuffled"),
            })
    for name, c in summary["cross_model_real"].items():
        tests.append({
            "family": "A_matched98",
            "name": f"cross-model real: {name}",
            "gap_pp": c["point_pp"],
            "n01": c["mcnemar"]["n01"],
            "n10": c["mcnemar"]["n10"],
            "p": c["mcnemar"]["p"],
            "supports_positive_claim": name in ("AGTA_minus_videollama2", "AGTA_minus_qwenomni"),
        })
    return tests


def build_family_b() -> list[dict]:
    tests: list[dict] = []
    for model in POSCONTROL_MODELS:
        n01, n10, acc_r, acc_s = load_poscontrol_pair(model)
        p = mcnemar_exact_p(n01, n10)
        tests.append({
            "family": "B_poscontrol_n47",
            "name": f"{POSCONTROL_LABELS[model]}: blanked-video real vs silent",
            "gap_pp": acc_r - acc_s,
            "n01": n01,
            "n10": n10,
            "p": p,
            # Externals that pass the control must show a *significant* blanked R-S.
            "supports_positive_claim": model in ("videollama2", "vsalm2"),
        })
    return tests


def summarize(tests: list[dict], title: str) -> list[str]:
    holm(tests, ALPHA)
    benjamini_hochberg(tests, ALPHA)
    tests_sorted = sorted(tests, key=lambda t: t["p"])
    lines = [f"### {title}  (m = {len(tests)} tests, alpha = {ALPHA})", ""]
    lines.append("| Test | gap (pp) | n01/n10 | raw p | Holm reject | BH reject |")
    lines.append("|---|---|---|---|---|---|")
    for t in tests_sorted:
        lines.append(
            f"| {t['name']} | {t['gap_pp']:+.1f} | {t['n01']}/{t['n10']} | "
            f"{t['p']:.2e} | {'YES' if t['holm_reject'] else 'no'} | "
            f"{'YES' if t['bh_reject'] else 'no'} |"
        )
    lines.append("")
    # Headline check: do all positive-claim tests survive both corrections?
    pos = [t for t in tests if t["supports_positive_claim"]]
    surviving = [t for t in pos if t["holm_reject"] and t["bh_reject"]]
    lines.append(
        f"**Positive-claim tests: {len(surviving)}/{len(pos)} survive BOTH "
        f"Holm-Bonferroni and BH-FDR at alpha={ALPHA}.**"
    )
    if len(surviving) < len(pos):
        failed = [t["name"] for t in pos if not (t["holm_reject"] and t["bh_reject"])]
        lines.append(f"NOT surviving: {failed}")
    lines.append("")
    return lines


def main() -> None:
    fam_a = build_family_a()
    fam_b = build_family_b()
    pooled = [dict(t) for t in fam_a] + [dict(t) for t in fam_b]

    md: list[str] = ["# Multiple-Comparison Correction (rebuttal deliverable)", ""]
    md.append(
        "Exact-McNemar paired tests, corrected for multiplicity. Reconciles with "
        "`matched_98/summary_with_ci.json`. Positive-claim tests are the audio-"
        "grounding effects the paper's claims *depend on being significant*; "
        "null/invariance claims are unaffected by correction (see script docstring).\n"
    )
    md += summarize([dict(t) for t in fam_a], "Family A — matched-98 subset")
    md += summarize([dict(t) for t in fam_b], "Family B — positive control (n=47, blanked video)")
    md += summarize(pooled, "Pooled — all tests as a single conservative family")

    # Recompute annotated copies for JSON persistence.
    fam_a_j, fam_b_j, pooled_j = [dict(t) for t in fam_a], [dict(t) for t in fam_b], [dict(t) for t in pooled]
    for group in (fam_a_j, fam_b_j, pooled_j):
        holm(group, ALPHA)
        benjamini_hochberg(group, ALPHA)

    out = {
        "alpha": ALPHA,
        "method": "exact McNemar per test; Holm-Bonferroni (FWER) + Benjamini-Hochberg (FDR)",
        "family_A_matched98": fam_a_j,
        "family_B_poscontrol_n47": fam_b_j,
        "pooled": pooled_j,
    }
    (MATCHED_DIR / "multiple_comparison.json").write_text(json.dumps(out, indent=2))
    (MATCHED_DIR / "multiple_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {MATCHED_DIR/'multiple_comparison.json'}")
    print(f"Wrote {MATCHED_DIR/'multiple_comparison.md'}")


if __name__ == "__main__":
    main()
