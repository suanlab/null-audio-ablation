"""Regenerate paper/figures/audio_grounding_spectrum.pdf — the 5-mode bar chart.

User-requested adjustments (2026-05-21):
  - tighter bar spacing within each mode group
  - larger text sizes throughout
Data is read live from `eval_results/{full_v6, no_bridge, vision_only}_{mode}.json`
so the figure always matches the audited paper values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results"
OUT = ROOT / "paper" / "figures" / "audio_grounding_spectrum.pdf"

MODES = ["real", "silent", "noise", "shuffled", "shifted"]
MODELS = [("AGTA (full)", "full_v6", "#1f77b4"),
          ("No-Bridge",   "no_bridge", "#ff7f0e"),
          ("Vision-Only", "vision_only", "#7f7f7f")]


def acc(fp: str) -> float:
    return json.loads((EV / f"{fp}.json").read_text())["accuracy"]


# Build (n_models, n_modes) matrix
M = np.array([[acc(f"{fp}_{m}") for m in MODES] for _, fp, _ in MODELS])

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6.5, 3.4))  # single-column friendly when scaled

n_models = len(MODELS)
n_modes = len(MODES)
group_w = 0.72                                 # narrower group => tighter bars
bar_w = group_w / n_models
x = np.arange(n_modes)                         # group centers
for i, (name, _, color) in enumerate(MODELS):
    pos = x - group_w / 2 + (i + 0.5) * bar_w
    bars = ax.bar(pos, M[i], bar_w * 0.98, color=color, edgecolor="black",
                  linewidth=0.5, label=name)
    for xi, v in zip(pos, M[i], strict=False):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color="#333333")

# 25% chance line
ax.axhline(25.0, color="#888888", linewidth=0.7, linestyle=":", zorder=0)
ax.text(n_modes - 0.45, 25.5, "chance (25%)", color="#666666",
        fontsize=9, ha="right", va="bottom", style="italic")

ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in MODES], fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 85)
ax.set_xlim(-0.55, n_modes - 0.45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, fontsize=11, handletextpad=0.5, columnspacing=1.5,
          labelspacing=0.3)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
print(f"wrote {OUT}")
print("data (rows=models, cols=modes):")
for (name, _, _), row in zip(MODELS, M, strict=False):
    print(f"  {name:14s}", " ".join(f"{v:5.1f}" for v in row))
