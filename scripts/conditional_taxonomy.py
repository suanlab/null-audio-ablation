"""Reviewer R5 [Critical] #2: ceiling-effect-corrected taxonomy.

Recomputes Table 10 (per-sample failure taxonomy) restricted to items where
*real audio* was answered correctly. The original Table 10 mixes accuracy and
audio sensitivity: high-accuracy models with few wrong items inevitably score
high on T3/T5 patterns. Conditional taxonomy gives a denominator comparable
across accuracy levels.

Definitions (conditional on real correct):
  T1*  audio-rescued    : silent wrong  (real was right, audio actively helped)
  T2*  semantic-only    : sil and shuf wrong, shift right (content matters, alignment does not)
  T4*  audio-harms      : N/A in this conditional (real must be right)
  T5*  audio-ignored    : all 5 modes give same pred (= same right answer)
  T3*  shift-insensitive: shift right (trivially since real right)  -- reported but not headline

Output: eval_results/matched_98/conditional_taxonomy.json + a small markdown table.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"

# (display name, file prefix, list of mode files)
MODELS = [
    ("AGTA",              {"real": "full_v6_real", "silent": "full_v6_silent",
                            "noise": "full_v6_noise", "shuffled": "full_v6_shuffled",
                            "shifted": "full_v6_shifted"}),
    ("VideoLLaMA2.1-AV",  {"real": "videollama2_real", "silent": "videollama2_silent",
                            "noise": "videollama2_noise", "shuffled": "videollama2_shuffled",
                            "shifted": "videollama2_shifted"}),
    ("video-SALMONN 2+",  {"real": "vsalm2_real", "silent": "vsalm2_silent",
                            "noise": "vsalm2_noise", "shuffled": "vsalm2_shuffled",
                            "shifted": "vsalm2_shifted"}),
    ("Qwen2.5-Omni",      {"real": "qwenomni_real", "silent": "qwenomni_silent",
                            "noise": "qwenomni_noise", "shuffled": "qwenomni_shuffled",
                            "shifted": "qwenomni_shifted"}),
]
MODES = ["real", "silent", "noise", "shuffled", "shifted"]


def load_preds(path: Path) -> dict[str, dict]:
    """Return {index: {pred, correct(bool), gt}} from a dump file."""
    raw = json.loads(path.read_text())
    out = {}
    for item in raw["predictions"]:
        idx = str(item["index"])
        out[idx] = {
            "pred": item["pred"],
            "correct": str(item.get("correct", "")).lower() == "true",
            "gt": item["gt"],
        }
    return out


def classify(real_c: bool, silent_c: bool, noise_c: bool,
             shuf_c: bool, shift_c: bool, preds: dict[str, str]) -> dict[str, bool]:
    """Return {tag: bool} for the 5 conditional patterns. Caller filters on real_c."""
    same_pred = len(set(preds.values())) == 1
    return {
        "T1_audio_rescued":   (not silent_c),                                  # silent wrong
        "T2_semantic_only":   ((not silent_c) and (not shuf_c) and shift_c),   # sil+shuf wrong, shift right
        "T3_shift_insens":    shift_c,                                          # shift right
        "T4_audio_harms":     False,                                            # by construction excluded
        "T5_audio_ignored":   same_pred,                                        # all 5 modes same answer
    }


def main() -> None:
    out_rows = []
    md_lines = ["| Pattern (conditional on real correct) | AGTA | VL2.1-AV | vsalm2+ | Qwen2.5-Omni |",
                "|---|---|---|---|---|"]
    counts: dict[str, dict[str, int]] = {}
    real_correct_n: dict[str, int] = {}
    for name, files in MODELS:
        modes = {m: load_preds(EV / f"{files[m]}.json") for m in MODES}
        idx_set = set(modes["real"].keys())
        for m in MODES[1:]:
            idx_set &= set(modes[m].keys())
        items = sorted(idx_set, key=lambda s: int(s))
        # Restrict to real-correct
        rc = [i for i in items if modes["real"][i]["correct"]]
        real_correct_n[name] = len(rc)
        tag_counts = Counter()
        for i in rc:
            preds = {m: modes[m][i]["pred"] for m in MODES}
            tags = classify(
                modes["real"][i]["correct"], modes["silent"][i]["correct"],
                modes["noise"][i]["correct"], modes["shuffled"][i]["correct"],
                modes["shifted"][i]["correct"], preds,
            )
            for t, v in tags.items():
                if v:
                    tag_counts[t] += 1
        counts[name] = dict(tag_counts)
        out_rows.append({"model": name, "n_real_correct": len(rc), **dict(tag_counts)})
    # Markdown table
    rows = [
        ("T1* audio-rescued (silent wrong)",     "T1_audio_rescued"),
        ("T2* semantic-only (sil/shuf wrong, shift right)", "T2_semantic_only"),
        ("T3* shift-insens. (shift right)",      "T3_shift_insens"),
        ("T5* audio-ignored (identical across 5 modes)", "T5_audio_ignored"),
    ]
    md_lines.append(f"| n (real correct)       | {real_correct_n['AGTA']} | "
                    f"{real_correct_n['VideoLLaMA2.1-AV']} | {real_correct_n['video-SALMONN 2+']} | "
                    f"{real_correct_n['Qwen2.5-Omni']} |")
    for label, k in rows:
        md_lines.append(
            f"| {label} | "
            f"{counts['AGTA'].get(k,0)} | "
            f"{counts['VideoLLaMA2.1-AV'].get(k,0)} | "
            f"{counts['video-SALMONN 2+'].get(k,0)} | "
            f"{counts['Qwen2.5-Omni'].get(k,0)} |"
        )

    out = {"n_universe": 98, "n_real_correct": real_correct_n,
           "counts_conditional_real_correct": counts, "rows": out_rows}
    (EV / "matched_98" / "conditional_taxonomy.json").write_text(json.dumps(out, indent=2))
    print("\n".join(md_lines))
    print()
    print(f"Wrote {EV / 'matched_98' / 'conditional_taxonomy.json'}")


if __name__ == "__main__":
    main()
