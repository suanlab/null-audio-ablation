"""Reviewer R1 [Critical] #1: head-to-head 3-mode vs 5-mode audit.

The paper claims the 5-mode protocol + positive control adds qualitative
information over 3-mode audits (e.g., real/silent/noise of silence2026icassp).
This script demonstrates the marginal value: for each of the 4 models, we
compute what a 3-mode audit (real/silent/noise) would conclude vs. what the
full protocol concludes.

Specifically we report, per model:
  3-mode verdict columns: R-S, R-N
  5-mode added columns  : R-Sh (shuffled), R-Shift (shifted), blanked R-S
  qualitative_added     : a textual one-line summary of what only 5-mode + pos. ctrl
                          can decide

Output: eval_results/matched_98/three_vs_five.json + markdown table.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"

MODELS = [
    ("AGTA",             {"real": "full_v6_real",    "silent": "full_v6_silent",
                          "noise": "full_v6_noise",  "shuffled": "full_v6_shuffled",
                          "shifted": "full_v6_shifted"}),
    ("VideoLLaMA2.1-AV", {"real": "videollama2_real",  "silent": "videollama2_silent",
                          "noise": "videollama2_noise", "shuffled": "videollama2_shuffled",
                          "shifted": "videollama2_shifted"}),
    ("video-SALMONN 2+", {"real": "vsalm2_real",       "silent": "vsalm2_silent",
                          "noise": "vsalm2_noise",     "shuffled": "vsalm2_shuffled",
                          "shifted": "vsalm2_shifted"}),
    ("Qwen2.5-Omni",     {"real": "qwenomni_real",     "silent": "qwenomni_silent",
                          "noise": "qwenomni_noise",   "shuffled": "qwenomni_shuffled",
                          "shifted": "qwenomni_shifted"}),
]
POS_CTRL = {
    "AGTA":             {"real_v-": "poscontrol_agta_real",       "silent_v-": "poscontrol_agta_silent"},
    "VideoLLaMA2.1-AV": {"real_v-": "poscontrol_videollama2_real", "silent_v-": "poscontrol_videollama2_silent"},
    "video-SALMONN 2+": {"real_v-": "poscontrol_vsalm2_real",      "silent_v-": "poscontrol_vsalm2_silent"},
    "Qwen2.5-Omni":     {"real_v-": "poscontrol_qwenomni_real",    "silent_v-": "poscontrol_qwenomni_silent"},
}


def acc(path: Path) -> float:
    return float(json.loads(path.read_text())["accuracy"])


def main() -> None:
    rows = []
    print("# 3-mode vs 5-mode head-to-head\n")
    print("| Model | (3-mode) R-S, R-N | (5-mode added) R-Sh, R-Shift | (pos. ctrl) Blanked R-S | "
          "Verdict from 3-mode only | Verdict adding 5-mode + pos. ctrl |")
    print("|---|---|---|---|---|---|")
    for name, files in MODELS:
        a = {k: acc(EV / f"{v}.json") for k, v in files.items()}
        rs = a["real"] - a["silent"]
        rn = a["real"] - a["noise"]
        rsh = a["real"] - a["shuffled"]
        rshift = a["real"] - a["shifted"]
        pc = POS_CTRL[name]
        rv_b = acc(EV / f"{pc['real_v-']}.json")
        sv_b = acc(EV / f"{pc['silent_v-']}.json")
        rs_blank = rv_b - sv_b
        # 3-mode verdict (real/silent/noise)
        if abs(rs) < 5 and abs(rn) < 5:
            v3 = "audio appears unused (R-S, R-N near 0)"
        elif rs > 5 and rn > 5:
            v3 = "audio appears used (R-S and R-N both > 5pp)"
        else:
            v3 = "mixed: small R-S, larger R-N or vice versa"
        # 5-mode + pos ctrl verdict
        delivery_ok = abs(rs_blank) > 5
        small_present = abs(rs) < 5
        large_blank = rs_blank > 10
        if not delivery_ok and small_present:
            v5 = "INCONCLUSIVE: positive control fails; cannot separate audio-ignoring from delivery failure"
        elif small_present and large_blank:
            v5 = "DISSOCIATION: audio reaches model and is usable, but video dominates at benchmark"
        elif not small_present and large_blank:
            v5 = "AUDIO-GROUNDED: behavioral audio dependence under both video conditions"
        elif rsh > 10 and rshift < 5:
            v5 = "CONTENT-DRIVEN: model uses audio content, not synchrony"
        else:
            v5 = "AUDIO-INSENSITIVE: small effects in all conditions"
        rows.append({"model": name, "rs_pp": rs, "rn_pp": rn, "rsh_pp": rsh,
                     "rshift_pp": rshift, "rs_blanked_pp": rs_blank,
                     "verdict_3mode": v3, "verdict_5mode_posctrl": v5})
        print(f"| {name} | "
              f"R-S {rs:+.1f}pp, R-N {rn:+.1f}pp | "
              f"R-Sh {rsh:+.1f}pp, R-Shift {rshift:+.1f}pp | "
              f"{rs_blank:+.1f}pp | "
              f"{v3} | "
              f"{v5} |")
    out = {"rows": rows,
           "note": ("3-mode = real/silent/noise; 5-mode adds shuffled+shifted; "
                    "positive control adds blanked-video R-S.")}
    (EV / "matched_98" / "three_vs_five.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {EV / 'matched_98' / 'three_vs_five.json'}")
    # Disagreement summary
    print("\n# Models for which 5-mode + pos. ctrl gives a different verdict than 3-mode")
    for r in rows:
        if "INCONCLUSIVE" in r["verdict_5mode_posctrl"] or "DISSOCIATION" in r["verdict_5mode_posctrl"]:
            print(f"- {r['model']}: 3-mode -> '{r['verdict_3mode']}'; "
                  f"full -> '{r['verdict_5mode_posctrl']}'")


if __name__ == "__main__":
    main()
