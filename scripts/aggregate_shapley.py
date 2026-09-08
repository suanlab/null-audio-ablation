"""T2.1: Shapley-style audio attribution from the existing 2x2 ablation grid.

For each model we already have 4 conditions on the n=47 audio-explicit subset:
    (a) real audio,  full video       = real_full
    (b) silent,      full video       = silent_full
    (c) real audio,  blanked video    = poscontrol_real
    (d) silent,      blanked video    = poscontrol_silent

These form a 2x2 ablation grid {video on/off} x {audio on/off}. The two-player
Shapley value for the audio modality is the average marginal contribution of
audio across the two coalitions (video on / video off):

    Shapley(audio) = 0.5 * (real_full   - silent_full)
                   + 0.5 * (poscontrol_real - poscontrol_silent)

and symmetrically for video. The Shapley audio-share = |Sh(audio)| /
(|Sh(audio)| + |Sh(video)|) -- a principled scalar attribution that
complements the raw R-S gap with a proper modality-contribution decomposition
(MM-SHAP-style averaging). No new GPU work needed; all four conditions are
already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"
OUT = EV / "matched_98" / "shapley.json"
IDX = set(json.loads((EV / "matched_98" / "audio_explicit_idx.json").read_text()))

MODELS = {
    "AGTA":              ("full_v6",     "agta"),
    "VideoLLaMA2.1-AV":  ("videollama2", "videollama2"),
    "video-SALMONN 2+":  ("vsalm2",      "vsalm2"),
    "Qwen2.5-Omni":      ("qwenomni",    "qwenomni"),
}


def acc_subset(fp: str) -> float:
    d = json.loads((EV / f"{fp}.json").read_text())["predictions"]
    s = [p for p in d if int(p["index"]) in IDX]
    return 100.0 * sum(p["correct"] == "True" for p in s) / len(s)


def acc_full(fp: str) -> float:
    return json.loads((EV / f"{fp}.json").read_text())["accuracy"]


def main() -> None:
    rows = []
    for name, (fp, pctag) in MODELS.items():
        r_full = acc_subset(f"{fp}_real")
        s_full = acc_subset(f"{fp}_silent")
        r_blank = acc_full(f"poscontrol_{pctag}_real")
        s_blank = acc_full(f"poscontrol_{pctag}_silent")
        rs_vp = r_full - s_full           # R-S | video present
        rs_vb = r_blank - s_blank         # R-S | video blanked
        sh_audio = 0.5 * rs_vp + 0.5 * rs_vb
        sh_video = 0.5 * (r_full - r_blank) + 0.5 * (s_full - s_blank)
        total = abs(sh_audio) + abs(sh_video)
        audio_share = 100.0 * abs(sh_audio) / total if total > 0 else 0.0
        rows.append({
            "model": name, "n": 47,
            "rs_video_present_pp": round(rs_vp, 2),
            "rs_video_blanked_pp": round(rs_vb, 2),
            "shapley_audio_pp": round(sh_audio, 2),
            "shapley_video_pp": round(sh_video, 2),
            "audio_share_pct": round(audio_share, 1),
        })
    out = {"n": 47, "subset": "audio_explicit",
           "method": "two-player Shapley = 0.5*marginal(coalition=v+)+0.5*marginal(coalition=v-)",
           "rows": rows}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}\n")
    print(f"{'Model':22s} | R-S (v+) | R-S (v-) | Sh(audio) | Sh(video) | audio-share%")
    print("-" * 92)
    for r in rows:
        print(f"{r['model']:22s} | {r['rs_video_present_pp']:+7.2f} | {r['rs_video_blanked_pp']:+7.2f} | "
              f"{r['shapley_audio_pp']:+8.2f} | {r['shapley_video_pp']:+8.2f} | {r['audio_share_pct']:5.1f}%")


if __name__ == "__main__":
    main()
