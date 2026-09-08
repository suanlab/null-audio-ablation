"""Reviewer R1/R2/R5: 10k-bootstrap CIs on the Shapley audio-share.

The two-player Shapley audio-share is reported as a point estimate in
Table 12; reviewers asked for uncertainty quantification and pairwise
distinguishability.

For each model and each bootstrap resample of the n=47 audio-explicit subset
(with replacement, seed 2026), we recompute:
  R-S(v+) = real_acc(v+) - silent_acc(v+)        on the resampled n=47
  R-S(v-) = real_acc(v-) - silent_acc(v-)        on the resampled n=47
  Sh(audio) = 0.5*R-S(v+) + 0.5*R-S(v-)
  Sh(video) = 0.5*(real_v+ - real_v-) + 0.5*(silent_v+ - silent_v-)
  audio_share = |Sh(audio)| / (|Sh(audio)| + |Sh(video)|)

Outputs:
  eval_results/matched_98/shapley_ci.json -- point estimates + 95% CI + leave-1-out range
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"
AE_IDX = {str(i) for i in json.loads((EV / "matched_98" / "audio_explicit_idx.json").read_text())}

# (display, v+ files, v- files)
MODELS = [
    ("AGTA",
        {"real": "full_v6_real", "silent": "full_v6_silent",
         "real_v-": "poscontrol_agta_real",   "silent_v-": "poscontrol_agta_silent"}),
    ("VideoLLaMA2.1-AV",
        {"real": "videollama2_real", "silent": "videollama2_silent",
         "real_v-": "poscontrol_videollama2_real", "silent_v-": "poscontrol_videollama2_silent"}),
    ("video-SALMONN 2+",
        {"real": "vsalm2_real", "silent": "vsalm2_silent",
         "real_v-": "poscontrol_vsalm2_real",     "silent_v-": "poscontrol_vsalm2_silent"}),
    ("Qwen2.5-Omni",
        {"real": "qwenomni_real", "silent": "qwenomni_silent",
         "real_v-": "poscontrol_qwenomni_real",   "silent_v-": "poscontrol_qwenomni_silent"}),
]


def load_corr(path: Path) -> dict[str, int]:
    """Return {index: 0/1} restricted to audio-explicit subset (n=47)."""
    raw = json.loads(path.read_text())
    out = {}
    for item in raw["predictions"]:
        idx = str(item["index"])
        if idx in AE_IDX:
            out[idx] = 1 if str(item.get("correct", "")).lower() == "true" else 0
    return out


def metrics(real_vp: np.ndarray, sil_vp: np.ndarray,
            real_vm: np.ndarray, sil_vm: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute (R-S_v+, R-S_v-, Sh_audio, Sh_video, audio_share) on a single sample.

    All four input arrays are aligned by item index over the same subset.
    """
    rs_vp = (real_vp.mean() - sil_vp.mean()) * 100
    rs_vm = (real_vm.mean() - sil_vm.mean()) * 100
    sh_audio = 0.5 * rs_vp + 0.5 * rs_vm
    sh_video = 0.5 * ((real_vp.mean() - real_vm.mean()) * 100) + \
               0.5 * ((sil_vp.mean()  - sil_vm.mean())  * 100)
    denom = abs(sh_audio) + abs(sh_video)
    share = 100 * abs(sh_audio) / denom if denom > 0 else 0.0
    return rs_vp, rs_vm, sh_audio, sh_video, share


def main() -> None:
    rng = np.random.default_rng(2026)
    n_boot = 10000
    all_rows = []
    print(f"# Shapley audio-share bootstrap CIs (n_boot={n_boot}, seed=2026)\n")
    print("| Model | Sh(audio) | Sh(video) | audio-share | 95% CI |")
    print("|---|---|---|---|---|")
    for name, files in MODELS:
        rv = load_corr(EV / f"{files['real']}.json")
        sv = load_corr(EV / f"{files['silent']}.json")
        rb = load_corr(EV / f"{files['real_v-']}.json")
        sb = load_corr(EV / f"{files['silent_v-']}.json")
        common = sorted(set(rv) & set(sv) & set(rb) & set(sb), key=lambda s: int(s))
        n = len(common)
        if n == 0:
            print(f"| {name} | -- no overlap -- |")
            continue
        rv_a = np.array([rv[i] for i in common])
        sv_a = np.array([sv[i] for i in common])
        rb_a = np.array([rb[i] for i in common])
        sb_a = np.array([sb[i] for i in common])
        # Point estimates
        pt = metrics(rv_a, sv_a, rb_a, sb_a)
        # Bootstrap CIs
        boot_share, boot_sh_a, boot_sh_v = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            _, _, sh_a, sh_v, share = metrics(rv_a[idx], sv_a[idx], rb_a[idx], sb_a[idx])
            boot_share.append(share)
            boot_sh_a.append(sh_a)
            boot_sh_v.append(sh_v)
        ci_share = (float(np.percentile(boot_share, 2.5)),
                    float(np.percentile(boot_share, 97.5)))
        ci_sh_a  = (float(np.percentile(boot_sh_a, 2.5)),  float(np.percentile(boot_sh_a, 97.5)))
        ci_sh_v  = (float(np.percentile(boot_sh_v, 2.5)),  float(np.percentile(boot_sh_v, 97.5)))
        # Leave-one-out stability
        loo_share = []
        for k in range(n):
            mask = np.ones(n, bool)
            mask[k] = False
            _, _, _, _, share = metrics(rv_a[mask], sv_a[mask], rb_a[mask], sb_a[mask])
            loo_share.append(share)
        loo_lo, loo_hi = float(min(loo_share)), float(max(loo_share))
        row = {"model": name, "n": n,
               "Sh_audio_pp": pt[2], "Sh_audio_ci": ci_sh_a,
               "Sh_video_pp": pt[3], "Sh_video_ci": ci_sh_v,
               "audio_share_pct": pt[4], "audio_share_ci_pct": ci_share,
               "audio_share_loo_range_pct": [loo_lo, loo_hi]}
        all_rows.append(row)
        print(f"| {name} | "
              f"{pt[2]:+.1f}pp [{ci_sh_a[0]:+.1f},{ci_sh_a[1]:+.1f}] | "
              f"{pt[3]:+.1f}pp [{ci_sh_v[0]:+.1f},{ci_sh_v[1]:+.1f}] | "
              f"{pt[4]:.1f}% [{ci_share[0]:.1f},{ci_share[1]:.1f}] | "
              f"LOO {loo_lo:.1f}--{loo_hi:.1f}% |")
    # Pairwise distinguishability for audio-share
    print("\n# Pairwise audio-share differences (10k paired bootstrap, same indices each step)\n")
    # We re-run a paired bootstrap because models share the same n=47 indices
    rng2 = np.random.default_rng(2027)
    cached = {}
    for name, files in MODELS:
        rv = load_corr(EV / f"{files['real']}.json")
        sv = load_corr(EV / f"{files['silent']}.json")
        rb = load_corr(EV / f"{files['real_v-']}.json")
        sb = load_corr(EV / f"{files['silent_v-']}.json")
        cached[name] = (rv, sv, rb, sb)
    common_all = sorted(set.intersection(*(set(c[0]) & set(c[1]) & set(c[2]) & set(c[3])
                                            for c in cached.values())),
                        key=lambda s: int(s))
    m_arrays = {name: tuple(np.array([cached[name][k][i] for i in common_all]) for k in range(4))
                for name in cached}
    pairs = [("AGTA", "VideoLLaMA2.1-AV"), ("AGTA", "video-SALMONN 2+"),
             ("VideoLLaMA2.1-AV", "video-SALMONN 2+"),
             ("video-SALMONN 2+", "Qwen2.5-Omni")]
    n = len(common_all)
    pair_results = []
    print("| Pair | Δ(audio-share, pp) | 95% CI | distinguishable at 0.05? |")
    print("|---|---|---|---|")
    for a, b in pairs:
        diffs = []
        for _ in range(n_boot):
            idx = rng2.integers(0, n, n)
            _, _, _, _, s_a = metrics(*[m_arrays[a][k][idx] for k in range(4)])
            _, _, _, _, s_b = metrics(*[m_arrays[b][k][idx] for k in range(4)])
            diffs.append(s_a - s_b)
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        pt_diff = float(np.mean(diffs))
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        pair_results.append({"a": a, "b": b, "diff_pp": pt_diff, "ci": [lo, hi], "sig": sig})
        print(f"| {a} - {b} | {pt_diff:+.1f}pp | [{lo:+.1f},{hi:+.1f}] | {sig} |")
    out = {"per_model": all_rows, "pairwise": pair_results,
           "n_audio_explicit": len(common_all), "n_boot": n_boot, "seed": 2026}
    (EV / "matched_98" / "shapley_ci.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {EV / 'matched_98' / 'shapley_ci.json'}")


if __name__ == "__main__":
    main()
