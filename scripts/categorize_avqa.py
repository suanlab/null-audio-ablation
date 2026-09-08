"""B3: Rule-based AVQA question categorization + per-category 5-mode breakdown.

Categorizes the 98 matched AVQA samples into three question types based on
the question text. Categories were chosen after inspecting the dataset (the
original PLAN.md's "temporal" category is essentially absent from AVQA):

    - audio-explicit : question mentions sound / noise / call / audio source
    - action         : "doing", "happen(ed)", "what's going on" (action queries)
    - scene-entity   : where / weather / what animal / what is shown / what color

For each model (full_v6, no_bridge, vision_only) and each category, reports the
5-mode accuracy stratified per category.

Inputs : data/instruct/avqa_test_clean.jsonl, eval_results/{model}_{mode}.json
Output : eval_results/matched_98/categories.json
         eval_results/matched_98/per_category_table.md

Usage : python scripts/categorize_avqa.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "instruct" / "avqa_test_clean.jsonl"
EVAL = REPO / "eval_results"
OUT = EVAL / "matched_98"

MODELS = ("full_v6", "no_bridge", "vision_only")
MODEL_LABELS = {"full_v6": "AGTA", "no_bridge": "No-Bridge", "vision_only": "Vision-Only"}
MODES = ("real", "silent", "noise", "shuffled", "shifted")

# Order matters: first matching rule wins.
_AUDIO_KEYS = re.compile(
    r"\b(sound|sounds|noise|noises|call|calling|audio|music|song|"
    r"singing|speak|speaking|voice|voices|cry|crying|bark|barking)\b",
    re.IGNORECASE,
)
_ACTION_KEYS = re.compile(
    r"\b(doing|happen|happened|happening|what's going on|going on)\b",
    re.IGNORECASE,
)
_SCENE_KEYS = re.compile(
    r"\b(where|weather|shown|animal|color|colors|place|location|"
    r"setting|environment)\b",
    re.IGNORECASE,
)


def categorize(question: str) -> str:
    """Return one of {audio-explicit, action, scene-entity, other}.

    Args:
        question: The human prompt text (with <video>/<audio> tokens stripped).
    """
    if _AUDIO_KEYS.search(question):
        return "audio-explicit"
    if _ACTION_KEYS.search(question):
        return "action"
    if _SCENE_KEYS.search(question):
        return "scene-entity"
    return "other"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]

    # Match the per-sample order used by eval scripts (positional, 0..97).
    cats: list[str] = []
    samples: list[dict[str, str]] = []
    for i, ex in enumerate(rows):
        q_raw = ex["conversations"][0]["value"]
        q_main = q_raw.replace("<video>", "").replace("<audio>", "").strip().split("\n")[0]
        c = categorize(q_main)
        cats.append(c)
        samples.append({"index": str(i), "video": ex["video"], "question": q_main, "category": c})

    # Sanity: distribution
    dist = Counter(cats)
    print("Category distribution:")
    for cat, n in dist.most_common():
        print(f"  {cat:15s}: {n:3d}  ({100*n/len(cats):.1f}%)")

    # Cross-check: indexing must match eval JSONs (which use index 0..97).
    eval_sample = json.loads((EVAL / "full_v6_real.json").read_text())["predictions"]
    assert len(eval_sample) == len(cats), "Sample count mismatch with eval JSONs"
    mismatches = sum(1 for i, p in enumerate(eval_sample) if p["video"] != rows[i]["video"])
    assert mismatches == 0, f"Video-id order mismatch: {mismatches} positions differ"

    # Per-category 5-mode accuracy per model
    table: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for m in MODELS:
        for mode in MODES:
            preds = json.loads((EVAL / f"{m}_{mode}.json").read_text())["predictions"]
            by_cat: dict[str, list[bool]] = defaultdict(list)
            for i, p in enumerate(preds):
                by_cat[cats[i]].append(p["correct"] == "True")
            for cat, vec in by_cat.items():
                acc = 100 * sum(vec) / len(vec)
                table[m][cat][mode] = acc
                table[m][cat]["n"] = len(vec)

    payload = {
        "n_samples": len(cats),
        "category_counts": dict(dist),
        "per_sample_categories": samples,
        "per_model_table": {m: dict(table[m]) for m in MODELS},
    }
    (OUT / "categories.json").write_text(json.dumps(payload, indent=2))

    # Markdown
    md = ["# Per-Category 5-Mode Accuracy (matched 98)\n"]
    md.append("Rule-based categorization; first matching rule wins (audio → action → scene-entity → other).\n")
    md.append("Category counts: " + ", ".join(f"{c}={n}" for c, n in dist.most_common()) + "\n")
    for m in MODELS:
        md.append(f"## {MODEL_LABELS[m]}\n")
        md.append("| Category | n | Real | Silent | Noise | Shuffled | Shifted | R−S |")
        md.append("|---|---|---|---|---|---|---|---|")
        for cat, _ in dist.most_common():
            row = table[m].get(cat)
            if row is None:
                continue
            n = int(row["n"])
            rs = row["real"] - row["silent"]
            md.append(
                f"| {cat} | {n} | {row['real']:.1f} | {row['silent']:.1f} | "
                f"{row['noise']:.1f} | {row['shuffled']:.1f} | {row['shifted']:.1f} | "
                f"{rs:+.1f} |"
            )
        md.append("")
    (OUT / "per_category_table.md").write_text("\n".join(md) + "\n")

    print(f"\nWrote {OUT/'categories.json'}")
    print(f"Wrote {OUT/'per_category_table.md'}")


if __name__ == "__main__":
    main()
