"""D1: failure taxonomy + qualitative examples from the matched 98-sample run.

Joins AVQA questions with per-sample predictions of AGTA (full_v6) and the
external VideoLLaMA2.1-AV across the 5 modes, then surfaces concrete examples
for each failure pattern in the protocol's taxonomy:

  T1 audio-rescued        : real correct, silent wrong  (clean audio grounding)
  T2 semantic-only audio  : real & shifted correct, silent & shuffled wrong
  T3 shift-insensitive    : real correct AND shifted correct (no timing effect)
  T4 audio-harms          : silent correct, real wrong   (audio misleads)
  T5 audio-ignored        : identical pred across all 5 modes (audio inert)

Output: eval_results/matched_98/failure_taxonomy.json  (+ console summary)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval_results"
DATA = REPO / "data" / "instruct" / "avqa_test_clean.jsonl"
MODES = ("real", "silent", "noise", "shuffled", "shifted")


def load_preds(model: str) -> list[dict[str, dict]]:
    per_mode = {m: json.loads((EVAL / f"{model}_{m}.json").read_text())["predictions"] for m in MODES}
    n = len(per_mode["real"])
    rows = []
    for i in range(n):
        rows.append({m: per_mode[m][i] for m in MODES})
    return rows


def question_text(rec: dict) -> tuple[str, str]:
    raw = rec["conversations"][0]["value"].replace("<video>", "").replace("<audio>", "").strip()
    gt = rec["conversations"][1]["value"]
    return raw, gt


def ok(p: dict) -> bool:
    return p["correct"] == "True"


def classify(row: dict[str, dict]) -> list[str]:
    r, s, n, sh, sf = (ok(row[m]) for m in MODES)
    tags = []
    if r and not s:
        tags.append("T1_audio_rescued")
    if r and sf and (not s) and (not sh):
        tags.append("T2_semantic_only_audio")
    if r and sf:
        tags.append("T3_shift_insensitive")
    if s and not r:
        tags.append("T4_audio_harms")
    preds = {row[m]["pred"] for m in MODES}
    if len(preds) == 1:
        tags.append("T5_audio_ignored")
    return tags


def main() -> None:
    recs = [json.loads(line) for line in DATA.read_text().splitlines()]
    out: dict[str, dict] = {}
    for model in ("full_v6", "videollama2", "qwenomni"):
        rows = load_preds(model)
        buckets: dict[str, list[dict]] = {}
        for i, row in enumerate(rows):
            q, gt = question_text(recs[i])
            for tag in classify(row):
                buckets.setdefault(tag, []).append({
                    "index": i,
                    "video": recs[i]["video"],
                    "question": q.split("Choices:")[0].strip()[:120],
                    "gt": row["real"]["gt"],
                    "preds": {m: row[m]["pred"] for m in MODES},
                })
        counts = {k: len(v) for k, v in sorted(buckets.items())}
        out[model] = {"counts": counts, "examples": {k: v[:3] for k, v in buckets.items()}}
        print(f"\n=== {model} ===")
        for k, c in counts.items():
            print(f"  {k:26s}: {c}")

    (EVAL / "matched_98" / "failure_taxonomy.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {EVAL/'matched_98'/'failure_taxonomy.json'}")
    # print 2 concrete examples per key category for AGTA
    print("\n--- AGTA representative examples ---")
    for tag in ("T1_audio_rescued", "T2_semantic_only_audio", "T4_audio_harms", "T5_audio_ignored"):
        exs = out["full_v6"]["examples"].get(tag, [])[:2]
        for e in exs:
            print(f"[{tag}] #{e['index']} gt={e['gt']} preds={e['preds']}")
            print(f"   Q: {e['question']}")


if __name__ == "__main__":
    main()
