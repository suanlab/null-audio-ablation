"""T1.4: Headline figure — the accuracy / audio-grounding dissociation.

Produces paper/figures/dissociation.pdf. The x-axis is the video-present R-S
(audio dependence when video is available); the y-axis is the blanked-video
R-S (audio dependence when video is removed); both on the n=47 audio-explicit
subset so the two coordinates are like-for-like. Points off the diagonal show
the dissociation: a model far above y=x has audio that is "available but
behaviorally unused under benchmark conditions" -- the paper's headline.

Marker conventions:
 - red filled circle:  positive-controlled, large dissociation (VideoLLaMA2.1-AV)
 - orange circle:      positive-controlled, moderate dissociation (video-SALMONN 2+)
 - blue triangle:      audio-grounded in-house (AGTA), near-diagonal
 - grey X:             inconclusive (Qwen2.5-Omni, failed positive control)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "eval_results" / "matched_98" / "shapley.json").read_text())
OUT = ROOT / "paper" / "figures" / "dissociation.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (x = R-S video-present, y = R-S video-blanked) from shapley.json
points = {r["model"]: (r["rs_video_present_pp"], r["rs_video_blanked_pp"]) for r in DATA["rows"]}

# Visual style
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed Type-1
})

fig, ax = plt.subplots(figsize=(4.2, 4.0))

# Diagonal y=x: "audio used equally whether video present or absent"
ax.plot([-3, 65], [-3, 65], color="#888888", linewidth=0.7, linestyle="--", zorder=1,
        label=r"$y{=}x$: audio used equally")

# Shaded region above diagonal (dissociation zone)
ax.fill_between([-3, 65], [-3, 65], 65, color="#fde9d9", alpha=0.5, zorder=0)
ax.text(28, 56, "dissociation\nzone", color="#a85a14", fontsize=8, style="italic",
        ha="center", va="center")

# Points (note: AGTA in-house is included as a calibration reference)
styles = {
    "AGTA":              dict(marker="^", color="#1f77b4", ms=10, mec="black", mew=0.6,
                              label="AGTA (in-house, audio-grounded)"),
    "VideoLLaMA2.1-AV":  dict(marker="o", color="#d62728", ms=12, mec="black", mew=0.6,
                              label="VideoLLaMA2.1-AV (pos. ctrl PASS)"),
    "video-SALMONN 2+":  dict(marker="o", color="#ff7f0e", ms=10, mec="black", mew=0.6,
                              label="video-SALMONN 2+ (pos. ctrl PASS)"),
    "Qwen2.5-Omni":      dict(marker="x", color="#7f7f7f", ms=10, mew=2,
                              label="Qwen2.5-Omni (pos. ctrl FAIL, inconcl.)"),
}
offsets = {  # text label dx, dy in data units
    "AGTA":              ( 2.0, -1.0),
    "VideoLLaMA2.1-AV":  ( 2.0, -1.5),
    "video-SALMONN 2+":  ( 2.0, -2.5),
    "Qwen2.5-Omni":      ( 1.5,  2.0),
}
for name, (x, y) in points.items():
    ax.plot(x, y, linestyle="", zorder=3, **styles[name])
    dx, dy = offsets[name]
    ax.text(x + dx, y + dy, name.replace(" ", " "), fontsize=7.5, va="center", zorder=4)

ax.set_xlim(-4, 38)
ax.set_ylim(-5, 65)
ax.set_xlabel(r"video-present R-S (pp)  $\rightarrow$ audio used when video available")
ax.set_ylabel(r"blanked-video R-S (pp)  $\rightarrow$ audio used when video removed")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
          frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.0,
          labelspacing=0.3)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT}")
print(f"data: {points}")
