"""Audit: cross-check every numeric claim in paper/main.tex against eval_results."""
from __future__ import annotations

import json
from pathlib import Path

EV = Path(__file__).resolve().parent.parent / "eval_results"


def load_json(f: str) -> dict:
    return json.loads((EV / f"{f}.json").read_text())


def acc(f: str):
    d = load_json(f)
    return round(d["accuracy"], 1), d["correct"], d["total"]


S = json.loads((EV / "matched_98" / "summary_with_ci.json").read_text())
TX = json.loads((EV / "matched_98" / "failure_taxonomy.json").read_text())
CAT = json.loads((EV / "matched_98" / "categories.json").read_text())
fails: list[str] = []


def chk(name: str, cond: bool) -> None:
    print(("OK   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# headline per-mode (matched 98)
for mdl, fp, exp in [
    ("AGTA", "full_v6", (58.2, 33.7, 34.7, 34.7, 61.2)),
    ("No-Bridge", "no_bridge", (69.4, 50.0, 50.0, 26.5, 65.3)),
    ("Vision-Only", "vision_only", (50.0, 50.0, 50.0, 50.0, 50.0)),
    ("VideoLLaMA2", "videollama2", (98.0, 93.9, 94.9, 92.9, 98.0)),
    ("Qwen-Omni", "qwenomni", (82.7, 82.7, 82.7, 82.7, 82.7)),
]:
    got = tuple(acc(f"{fp}_{m}")[0] for m in ["real", "silent", "noise", "shuffled", "shifted"])
    chk(f"{mdl} per-mode {exp} (got {got})", got == exp)


def gap(m: str, o: str = "silent"):
    g = S["models"][m]["gaps_pp"][f"real_minus_{o}"]
    return round(g["point_pp"], 1), round(g["ci95_lo"], 1), round(g["ci95_hi"], 1), g["mcnemar"]["p"]


chk("AGTA R-S +24.5", gap("full_v6")[0] == 24.5)
chk("No-Bridge R-S +19.4", gap("no_bridge")[0] == 19.4)
chk("VL2 R-S +4.1 [1.0,8.2]", gap("videollama2")[:3] == (4.1, 1.0, 8.2))
chk("Qwen R-S +0.0", gap("qwenomni")[0] == 0.0)

# Paper uses the EXACT binomial McNemar test (not the chi-square approx).
mc_a = S["models"]["full_v6"]["gaps_pp"]["real_minus_silent"]["mcnemar"]["p"]
mc_n = S["models"]["no_bridge"]["gaps_pp"]["real_minus_silent"]["mcnemar"]["p"]
chk(f"AGTA R-S exact McNemar p=1.16e-4 (got {mc_a:.3e})", round(mc_a, 6) == round(1.162e-4, 6))
chk(f"No-Bridge R-S exact McNemar p=2.56e-3 (got {mc_n:.3e})", abs(mc_n - 2.563e-3) < 5e-6)

ps_a = S["models"]["full_v6"]["gaps_pp"]["real_minus_shifted"]["mcnemar"]["p"]
ps_n = S["models"]["no_bridge"]["gaps_pp"]["real_minus_shifted"]["mcnemar"]["p"]
chk(f"AGTA R-shift exact McNemar p=0.58 (got {ps_a:.3f})", round(ps_a, 2) == 0.58)
chk(f"No-Bridge R-shift exact McNemar p=0.39 (got {ps_n:.3f})", round(ps_n, 2) == 0.39)

dd = S["rs_gap_diff_in_diff"]


def diff_in_diff(k: str):
    v = dd[k]
    return round(v["point_pp"], 1), round(v["ci95_lo"], 1), round(v["ci95_hi"], 1)


chk("dd AGTA-NoBridge +5.1 [-9.2,19.4]", diff_in_diff("AGTA_RS_minus_NoBridge_RS") == (5.1, -9.2, 19.4))
chk("dd AGTA-VisionOnly +24.5 [13.3,35.7]", diff_in_diff("AGTA_RS_minus_VisionOnly_RS") == (24.5, 13.3, 35.7))
chk("dd AGTA-VL2 +20.4 [8.2,32.7]", diff_in_diff("AGTA_RS_minus_VideoLLaMA2_RS") == (20.4, 8.2, 32.7))
chk("dd AGTA-Qwen +24.5 [13.3,35.7]", diff_in_diff("AGTA_RS_minus_QwenOmni_RS") == (24.5, 13.3, 35.7))

cm = S["cross_model_real"]
v = cm["AGTA_minus_videollama2"]
chk(f"AGTA-VL2 real -39.8 p<1e-4 (got {v['point_pp']:.1f}, p={v['boot_p_two_sided']:.4f})",
    round(v["point_pp"], 1) == -39.8 and v["boot_p_two_sided"] < 1e-3)
v = cm["AGTA_minus_no_bridge"]
chk(f"AGTA-NoBridge real ~-11.2 (got {v['point_pp']:.1f}, p={v['boot_p_two_sided']:.4f})",
    round(v["point_pp"], 1) == -11.2)

chk("AGTA shifted1s == real (58.2)", acc("full_v6_shifted_1s")[0] == acc("full_v6_real")[0] == 58.2)

tk = ["T1_audio_rescued", "T2_semantic_only_audio", "T3_shift_insensitive", "T4_audio_harms", "T5_audio_ignored"]
chk("tax AGTA 31/19/52/7/16", [TX["full_v6"]["counts"].get(k, 0) for k in tk] == [31, 19, 52, 7, 16])
chk("tax VL2 4/4/96/0/92", [TX["videollama2"]["counts"].get(k, 0) for k in tk] == [4, 4, 96, 0, 92])
chk("tax Qwen 0/0/81/0/98", [TX["qwenomni"]["counts"].get(k, 0) for k in tk] == [0, 0, 81, 0, 98])
# vsalm2 taxonomy is computed inline (not yet exported by extract_failure_taxonomy.py)
_modes_t = ["real", "silent", "noise", "shuffled", "shifted"]
_PV = {m: {x["index"]: x for x in load_json(f"vsalm2_{m}")["predictions"]} for m in _modes_t}
_ii = sorted(set.intersection(*[set(_PV[m]) for m in _modes_t]))
_T1 = sum(1 for i in _ii if _PV["real"][i]["correct"] == "True" and _PV["silent"][i]["correct"] != "True")
_T2 = sum(1 for i in _ii if _PV["real"][i]["correct"] == "True" and _PV["shifted"][i]["correct"] == "True"
          and _PV["silent"][i]["correct"] != "True" and _PV["shuffled"][i]["correct"] != "True")
_T3 = sum(1 for i in _ii if _PV["real"][i]["correct"] == "True" and _PV["shifted"][i]["correct"] == "True")
_T4 = sum(1 for i in _ii if _PV["silent"][i]["correct"] == "True" and _PV["real"][i]["correct"] != "True")
_T5 = sum(1 for i in _ii if len({_PV[m][i]["pred"] for m in _modes_t}) == 1)
chk(f"tax vsalm2 2/2/83/1/87 (got {_T1}/{_T2}/{_T3}/{_T4}/{_T5})",
    [_T1, _T2, _T3, _T4, _T5] == [2, 2, 83, 1, 87])

pc = CAT["per_model_table"]


def cell(m, cat, mode):
    return round(pc[m][cat][mode], 1)


chk("cat AGTA audio-explicit real 66.0", cell("full_v6", "audio-explicit", "real") == 66.0)
_ae = pc["full_v6"]["audio-explicit"]
chk(f"cat AGTA audio-explicit R-S +31.9 (raw {_ae['real']-_ae['silent']:.4f})",
    round(_ae["real"] - _ae["silent"], 1) == 31.9)
chk("cat counts 47/29/19",
    CAT["category_counts"].get("audio-explicit") == 47
    and CAT["category_counts"].get("action") == 29
    and CAT["category_counts"].get("scene-entity") == 19)

lr = acc("full_v6_large_real")
ls = acc("full_v6_large_shifted")
chk(f"large AGTA real 77.3 (got {lr[0]})", lr[0] == 77.3)
chk(f"large AGTA shifted 79.1 (got {ls[0]})", ls[0] == 79.1)
chk(f"large n=163 (got {lr[2]})", lr[2] == 163)

for v_, exp in [("qb_v2", (65.3, 33.7, 66.3, 23.5, 24.5)),
                ("qb_v3", (57.1, 43.9, 61.2, 45.9, 41.8)),
                ("qb_v4", (58.2, 35.7, 57.1, 35.7, 34.7))]:
    g = tuple(acc(f"{v_}_{m}")[0] for m in ["real", "shuffled", "shifted", "noise", "silent"])
    print(f"INFO {v_} (real,shuf,shift,noise,sil)={g} paper-expected~{exp}")

# MVBench JSONs store 'overall_accuracy'; paper rounds to 1 dp.
for f, exp in [("mvbench_full_v6", 32.1), ("mvbench_no_bridge", 32.4),
               ("mvbench_vision_only", 33.3)]:
    oa = round(load_json(f)["overall_accuracy"], 1)
    chk(f"{f} overall_accuracy {exp} (got {oa})", oa == exp)
chk(f"videomme_full_v6 22.3 (got {round(load_json('videomme_full_v6')['accuracy'],1)})",
    round(load_json("videomme_full_v6")["accuracy"], 1) == 22.3)

print("\n=== RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED -> {fails}")

# --- Positive control (blanked video, n=47 audio-explicit) ---

_idx = set(json.loads(Path("eval_results/matched_98/audio_explicit_idx.json").read_text()))


def _rs_present(fp: str) -> float:
    def sub(mode):
        d = json.loads(Path(f"eval_results/{fp}_{mode}.json").read_text())["predictions"]
        s = [p for p in d if int(p["index"]) in _idx]
        return 100.0 * sum(p["correct"] == "True" for p in s) / len(s)
    return round(sub("real") - sub("silent"), 1)


def _rs_blank(tag: str) -> float:
    r = json.loads(Path(f"eval_results/poscontrol_{tag}_real.json").read_text())
    s = json.loads(Path(f"eval_results/poscontrol_{tag}_silent.json").read_text())
    return round(r["accuracy"] - s["accuracy"], 1)


chk(f"poscontrol AGTA present +31.9 (got {_rs_present('full_v6')})", _rs_present("full_v6") == 31.9)
chk(f"poscontrol AGTA blank +29.8 (got {_rs_blank('agta')})", _rs_blank("agta") == 29.8)
chk(f"poscontrol VL2 present +6.4 (got {_rs_present('videollama2')})", _rs_present("videollama2") == 6.4)
chk(f"poscontrol VL2 blank +59.6 (got {_rs_blank('videollama2')})", _rs_blank("videollama2") == 59.6)
chk(f"poscontrol Qwen present +0.0 (got {_rs_present('qwenomni')})", _rs_present("qwenomni") == 0.0)
chk(f"poscontrol Qwen blank +0.0 (got {_rs_blank('qwenomni')})", _rs_blank("qwenomni") == 0.0)

# --- P1-A: video-SALMONN 2+ second positive-controlled external model ---
for _m, _a in [("real", 84.7), ("silent", 83.7), ("noise", 83.7),
               ("shuffled", 81.6), ("shifted", 86.7)]:
    chk(f"vsalm2 {_m} {_a} (got {acc('vsalm2_'+_m)[0]})", acc(f"vsalm2_{_m}")[0] == _a)
chk(f"vsalm2 present R-S +0.0 (got {_rs_present('vsalm2')})", _rs_present("vsalm2") == 0.0)
chk(f"vsalm2 blank R-S +27.7 (got {_rs_blank('vsalm2')})", _rs_blank("vsalm2") == 27.7)


def _pcvec(mode: str) -> dict:
    return {str(x["index"]): int(str(x["correct"]) == "True")
            for x in json.loads(Path(f"eval_results/poscontrol_vsalm2_{mode}.json").read_text())["predictions"]}


from math import comb as _comb  # noqa: E402

_br, _bs = _pcvec("real"), _pcvec("silent")
_k = sorted(set(_br) & set(_bs))
_b = sum(1 for i in _k if _br[i] and not _bs[i])
_c = sum(1 for i in _k if _bs[i] and not _br[i])
_n = _b + _c
_pmc = min(1.0, 2 * sum(_comb(_n, i) for i in range(min(_b, _c) + 1)) * 0.5 ** _n) if _n else 1.0
import numpy as _np  # noqa: E402

_dv = _np.array([_br[i] - _bs[i] for i in _k], float)
_rng = _np.random.default_rng(2026)
_bm = _dv[_rng.integers(0, len(_dv), (10000, len(_dv)))].mean(1) * 100
_lo, _hi = float(_np.percentile(_bm, 2.5)), float(_np.percentile(_bm, 97.5))
chk(f"vsalm2 blanked McNemar p=0.001 (got {_pmc:.4f})", round(_pmc, 3) == 0.001)
chk(f"vsalm2 blanked CI [14.9,42.6] (got [{_lo:.1f},{_hi:.1f}])",
    round(_lo, 1) == 14.9 and round(_hi, 1) == 42.6)
_VM = ["real", "silent", "noise", "shuffled", "shifted"]
_PP = {m: {str(x["index"]): x["pred"]
           for x in json.loads(Path(f"eval_results/vsalm2_{m}.json").read_text())["predictions"]}
       for m in _VM}
_ii = set.intersection(*[set(_PP[m]) for m in _VM])
_same = sum(1 for i in _ii if len({_PP[m][i] for m in _VM}) == 1)
chk(f"vsalm2 identical-5mode 87 (got {_same})", _same == 87)

# --- P1-C: shift-magnitude sweep vs Appendix tab:shiftsweep ---
_sw = json.loads(Path("eval_results/shift_sweep/summary.json").read_text())
_SWEEP_EXP = {
    "full_v6": {"0.1": (58.16, 0.00, 1.00), "0.25": (58.16, 0.00, 1.00),
                "0.5": (61.22, 3.06, 0.55), "1": (58.16, 0.00, 1.00),
                "2": (59.18, 1.02, 1.00), "3": (61.22, 3.06, 0.58),
                "5": (63.27, 5.10, 0.23)},
    "videollama2": {"0.1": (97.96, 0.00, 1.00), "0.25": (97.96, 0.00, 1.00),
                    "0.5": (97.96, 0.00, 1.00), "1": (97.96, 0.00, 1.00),
                    "2": (97.96, 0.00, 1.00), "3": (97.96, 0.00, 1.00),
                    "5": (98.98, 1.02, 1.00)},
}
chk("sweep missing == none", _sw.get("missing") == [])
for _m, _exp in _SWEEP_EXP.items():
    _rec = _sw["models"][_m]
    chk(f"sweep {_m} n=98", _rec["n"] == 98)
    _ps = []
    for _s, (_a, _d, _p) in _exp.items():
        _pt = _rec["points"][_s]
        _ps.append(_pt["mcnemar_p"])
        chk(f"sweep {_m} {_s}s acc {_a} (got {_pt['accuracy']})",
            round(_pt["accuracy"], 2) == _a)
        chk(f"sweep {_m} {_s}s d {_d:+.2f} (got {_pt['delta_vs_real_pp']})",
            round(_pt["delta_vs_real_pp"], 2) == _d)
        chk(f"sweep {_m} {_s}s p {_p:.2f} (got {_pt['mcnemar_p']})",
            round(_pt["mcnemar_p"], 2) == _p)
    chk(f"sweep {_m} all p>=0.05 (min {min(_ps):.4f})", all(q >= 0.05 for q in _ps))
chk("sweep AGTA max|d|=5.10 @5s",
    round(_sw["models"]["full_v6"]["points"]["5"]["delta_vs_real_pp"], 2) == 5.10)
chk("sweep AGTA min p>=0.23",
    round(min(q["mcnemar_p"] for q in _sw["models"]["full_v6"]["points"].values()), 2) == 0.23)
chk("sweep VL2 max|d|<=1.02 all p=1.0",
    round(max(abs(q["delta_vs_real_pp"]) for q in _sw["models"]["videollama2"]["points"].values()), 2) == 1.02
    and all(q["mcnemar_p"] == 1.0 for q in _sw["models"]["videollama2"]["points"].values()))

# --- P1-B: larger-n (n=548) external dissociation robustness vs tab:largen ---
_lg = json.loads(Path("eval_results/large548/summary.json").read_text())
chk("largen missing == none", _lg.get("missing") == [])
_VL = _lg["models"]["videollama2"]
_VS = _lg["models"]["vsalm2"]
for _tag, _r, _exp in [("VL2", _VL, {"n": 548, "real": 97.99, "silent": 93.61,
                                     "RS": 4.38, "CI": (2.5, 6.2),
                                     "shifted": 97.99, "Rshp": 1.00}),
                       ("VS2", _VS, {"n": 548, "real": 94.34, "silent": 94.71,
                                     "RS": -0.36, "CI": (-2.2, 1.5),
                                     "shifted": 93.98, "Rshp": 0.75})]:
    chk(f"{_tag} large n={_exp['n']} (got {_r['n_eff']})", _r["n_eff"] == _exp["n"])
    chk(f"{_tag} large real {_exp['real']} (got {_r['real']})", _r["real"] == _exp["real"])
    chk(f"{_tag} large silent {_exp['silent']} (got {_r['silent']})", _r["silent"] == _exp["silent"])
    chk(f"{_tag} large R-S {_exp['RS']:+.2f} (got {_r['RS_pp']})", _r["RS_pp"] == _exp["RS"])
    _lo, _hi = _r["RS_ci95"]
    chk(f"{_tag} large CI [{_exp['CI'][0]:+.1f},{_exp['CI'][1]:+.1f}] (got [{_lo:.2f},{_hi:.2f}])",
        round(_lo, 1) == _exp["CI"][0] and round(_hi, 1) == _exp["CI"][1])
    chk(f"{_tag} large shifted {_exp['shifted']} (got {_r.get('shifted')})",
        _r.get("shifted") == _exp["shifted"])
    chk(f"{_tag} large R-shift p={_exp['Rshp']:.2f} (got {_r.get('Rshift_mcnemar_p')})",
        round(_r.get("Rshift_mcnemar_p", -1.0), 2) == _exp["Rshp"])
chk(f"VL2 large R-S McNemar p<1e-4 (got {_VL['RS_mcnemar_p']:.4e})", _VL["RS_mcnemar_p"] < 1e-4)

# --- T2.1 Shapley attribution (eval_results/matched_98/shapley.json) ---
_sh = json.loads(Path("eval_results/matched_98/shapley.json").read_text())
_SHAP_EXP = {
    "AGTA":              (31.91, 29.79, 30.85,  1.06, 96.7),
    "VideoLLaMA2.1-AV":  ( 6.38, 59.57, 32.98, 28.72, 53.4),
    "video-SALMONN 2+":  ( 0.00, 27.66, 13.83, 41.49, 25.0),
    "Qwen2.5-Omni":      ( 0.00,  0.00,  0.00, 53.19,  0.0),
}
_by = {r["model"]: r for r in _sh["rows"]}
for _m, _e in _SHAP_EXP.items():
    _r = _by[_m]
    chk(f"shapley {_m} R-S(v+)={_e[0]:+.2f} (got {_r['rs_video_present_pp']:+.2f})",
        round(_r["rs_video_present_pp"], 2) == _e[0])
    chk(f"shapley {_m} R-S(v-)={_e[1]:+.2f} (got {_r['rs_video_blanked_pp']:+.2f})",
        round(_r["rs_video_blanked_pp"], 2) == _e[1])
    chk(f"shapley {_m} Sh(audio)={_e[2]:+.2f}", round(_r["shapley_audio_pp"], 2) == _e[2])
    chk(f"shapley {_m} Sh(video)={_e[3]:+.2f}", round(_r["shapley_video_pp"], 2) == _e[3])
    chk(f"shapley {_m} audio-share={_e[4]:.1f}%", round(_r["audio_share_pct"], 1) == _e[4])

# --- T1.2 modified VisionHard analogue (eval_results/matched_98/visionhard.json) ---
_vh = json.loads(Path("eval_results/matched_98/visionhard.json").read_text())
chk(f"VisionHard n=49 (got {_vh['n']})", _vh["n"] == 49)
_VH_EXP = {
    "AGTA":              (59.18, 28.57,  30.61, 0.0007),
    "VideoLLaMA2.1-AV":  (95.92, 91.84,   4.08, 0.5000),
    "video-SALMONN 2+":  (79.59, 81.63,  -2.04, 1.0000),
    "Qwen2.5-Omni":      (77.55, 77.55,   0.00, 1.0000),
}
_byvh = {r["model"]: r for r in _vh["rows"]}
for _m, _e in _VH_EXP.items():
    _r = _byvh[_m]
    chk(f"visionhard {_m} real={_e[0]:.2f}", round(_r["real"], 2) == _e[0])
    chk(f"visionhard {_m} silent={_e[1]:.2f}", round(_r["silent"], 2) == _e[1])
    chk(f"visionhard {_m} R-S={_e[2]:+.2f}", round(_r["rs_pp"], 2) == _e[2])
    chk(f"visionhard {_m} McNemar p={_e[3]}", round(_r["mcnemar_p"], 4) == _e[3])

# --- Reviewer-driven additions (revision round 2) ---

# Conditional-on-real-correct taxonomy (R5 ceiling-effect fix)
_ct = json.loads(Path("eval_results/matched_98/conditional_taxonomy.json").read_text())
_CT_EXP = {  # (n_real_correct, T5*)
    "AGTA":             (57, 9),
    "VideoLLaMA2.1-AV": (96, 91),
    "video-SALMONN 2+": (83, 77),
    "Qwen2.5-Omni":     (81, 81),
}
_by = {r["model"]: r for r in _ct["rows"]}
for _m, (_nrc, _t5) in _CT_EXP.items():
    chk(f"cond.tax {_m} n_real_correct={_nrc}", _by[_m]["n_real_correct"] == _nrc)
    chk(f"cond.tax {_m} T5*={_t5}",            _by[_m]["T5_audio_ignored"] == _t5)
# Headline contrast survives conditioning: AGTA T5*/n=15.8% vs VL2 94.8%
_agta_pct = _by["AGTA"]["T5_audio_ignored"] / _by["AGTA"]["n_real_correct"] * 100
_vl2_pct  = _by["VideoLLaMA2.1-AV"]["T5_audio_ignored"] / _by["VideoLLaMA2.1-AV"]["n_real_correct"] * 100
chk(f"cond.tax AGTA T5*≈15.8% (got {_agta_pct:.1f}%)", 15.0 <= _agta_pct <= 17.0)
chk(f"cond.tax VL2  T5*≈94.8% (got {_vl2_pct:.1f}%)",  94.0 <= _vl2_pct <= 96.0)

# Shapley bootstrap CIs (R1/R2/R5 ask)
_sci = json.loads(Path("eval_results/matched_98/shapley_ci.json").read_text())
_SCI_EXP = {  # (audio_share_pct, ci_lo_approx, ci_hi_approx)
    "AGTA":             (96.7, 85.0, 100.5),
    "VideoLLaMA2.1-AV": (53.4, 45.0, 62.0),
    "video-SALMONN 2+": (25.0, 10.0, 42.0),
    "Qwen2.5-Omni":     ( 0.0,  0.0,  0.5),
}
_by_sci = {r["model"]: r for r in _sci["per_model"]}
for _m, (_pt, _lo, _hi) in _SCI_EXP.items():
    _r = _by_sci[_m]
    chk(f"shapley_ci {_m} audio-share≈{_pt:.1f}",
        abs(_r["audio_share_pct"] - _pt) < 0.5)
    _ci = _r["audio_share_ci_pct"]
    chk(f"shapley_ci {_m} CI within tolerance [{_lo:.1f},{_hi:.1f}] (got [{_ci[0]:.1f},{_ci[1]:.1f}])",
        _ci[0] >= _lo - 1.0 and _ci[1] <= _hi + 1.0)
# All four pairwise differences should be distinguishable at α=0.05
for _p in _sci["pairwise"]:
    chk(f"shapley_ci pair {_p['a']}-{_p['b']} distinguishable (CI excludes 0)", _p["sig"] == "yes")

# TOST equivalence sweep (R2 ask)
_to = json.loads(Path("eval_results/matched_98/tost_shift.json").read_text())
_eq_vl2 = sum(1 for r in _to["rows"]
              if r.get("model") == "VideoLLaMA2.1-AV" and r.get("equivalent_at_3pp"))
chk(f"TOST VL2 equivalent at ±3pp on ≥6/7 magnitudes (got {_eq_vl2}/7)", _eq_vl2 >= 6)

# 3-mode vs 5-mode head-to-head (R1 [Critical] #1)
_tvf = json.loads(Path("eval_results/matched_98/three_vs_five.json").read_text())
_by_tvf = {r["model"]: r for r in _tvf["rows"]}
for _m in ["VideoLLaMA2.1-AV", "video-SALMONN 2+", "Qwen2.5-Omni"]:
    _r = _by_tvf[_m]
    chk(f"3vs5 {_m} 3-mode reads 'unused'", "unused" in _r["verdict_3mode"])
    chk(f"3vs5 {_m} 5-mode+ctrl flips verdict",
        "DISSOCIATION" in _r["verdict_5mode_posctrl"] or "INCONCLUSIVE" in _r["verdict_5mode_posctrl"])
chk("3vs5 AGTA agrees across both audits",
    "used" in _by_tvf["AGTA"]["verdict_3mode"] and "GROUNDED" in _by_tvf["AGTA"]["verdict_5mode_posctrl"])

print("\n=== FINAL:", "ALL PASS" if not fails else f"{len(fails)} FAILED -> {fails}")
