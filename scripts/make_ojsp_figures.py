"""Build every figure and every quoted number for the OJ-SP submission from the cell JSONs.

The paper's claims are all differences of accuracies over the same 300 items, so the
figures and the prose must not be able to drift apart. This script is the single source
for both: it emits the PDFs *and* `numbers.tex`, a file of LaTeX macros that the
manuscript uses instead of hard-coded digits. Change a cell, re-run, and the text moves
with the plot.

Outputs (into ``paper_ojsp/``):

* ``figures/fig1_delta_curves.pdf`` -- Delta_A(v) against visual severity, one panel per
  domain, both models, pooled over the four corruption families. This is the sign flip.
* ``figures/fig2_delivery.pdf`` -- the delivery controls against the registered 20 pp bar,
  showing that the two AVUT failures are qualitatively different.
* ``figures/fig3_redundancy.pdf`` -- single-channel versus joint accuracy, i.e. how much
  headroom each benchmark leaves for a modality effect to exist in.
* ``numbers.tex`` -- \\newcommand macros for every number quoted in the manuscript.

Usage::

    python scripts/make_ojsp_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - must follow the Agg backend selection

from videollm.cmss import cluster_bootstrap_mean  # noqa: E402 - local package after mpl setup

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval_results"
OUT_DIR = REPO_ROOT / "paper_ojsp"
FAMILIES = ("motion_blur", "occlusion", "frame_drop", "downscale")
SEVERITIES = ("0.33", "0.66", "1")
N_BOOT = 10000
SEED = 2027
CHANCE = 25.0  # 4-way multiple choice, both domains

# (display name, filename tag) per domain and model.
DOMAINS = (("MUSIC-AVQA", ""), ("AVUT", "avut_"))
MODELS = (("video-SALMONN 2+", "vsalm2"), ("VideoLLaMA2.1-AV", "vl2"))

# Colour-blind safe, and distinguishable in greyscale print.
COLOURS = {"vsalm2": "#0072B2", "vl2": "#D55E00"}
MARKERS = {"vsalm2": "o", "vl2": "s"}


def cell(domain_tag: str, model_tag: str, audio: str, suffix: str = "") -> dict[str, object]:
    """Load one evaluation cell."""
    path = EVAL_DIR / f"avqa_cmss_{domain_tag}{model_tag}_{audio}{suffix}.json"
    return json.loads(path.read_text())


def poscontrol(domain_tag: str, model_tag: str, audio: str) -> dict[str, object]:
    """Load one blank-video delivery control."""
    path = EVAL_DIR / f"poscontrol_cmss_{domain_tag}{model_tag}_{audio}.json"
    return json.loads(path.read_text())


def correctness(payload: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    """Return per-item correctness in {0,1} and the video each item belongs to."""
    preds = payload["predictions"]  # type: ignore[index]
    ok = np.array([1.0 if str(p["correct"]) == "True" else 0.0 for p in preds])
    videos = [str(p["video"]) for p in preds]
    return ok, videos


def paired_delta(domain_tag: str, model_tag: str, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-item Delta_A (pp) for one visual condition, plus its video-cluster ids."""
    real, videos_r = correctness(cell(domain_tag, model_tag, "real", suffix))
    silent, videos_s = correctness(cell(domain_tag, model_tag, "silent", suffix))
    if videos_r != videos_s:
        raise ValueError(f"item order differs between real and silent for {suffix or 'clean'}")
    index = {v: i for i, v in enumerate(sorted(set(videos_r)))}
    return (real - silent) * 100.0, np.array([index[v] for v in videos_r])


def delta_curve(domain_tag: str, model_tag: str) -> tuple[list[float], list[float], list[float]]:
    """Delta_A pooled over families at each severity, with cluster-bootstrap CIs."""
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for severity in ("clean", *SEVERITIES):
        if severity == "clean":
            diff, cid = paired_delta(domain_tag, model_tag, "")
        else:
            pieces = [paired_delta(domain_tag, model_tag, f"_vis-{f}-s{severity}") for f in FAMILIES]
            diff = np.concatenate([p[0] for p in pieces])  # (n_items * n_families,)
            cid = np.concatenate([p[1] for p in pieces])
        mean, lo, hi = cluster_bootstrap_mean(diff, cid, n_boot=N_BOOT, seed=SEED)
        means.append(mean)
        los.append(lo)
        his.append(hi)
    return means, los, his


def figure_delta_curves(curves: dict[tuple[str, str], tuple[list[float], list[float], list[float]]]) -> None:
    """Delta_A(v) per domain: the sign flip, in one picture."""
    x = [0.0, 0.33, 0.66, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (domain_name, domain_tag) in zip(axes, DOMAINS, strict=True):
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=1)
        for model_name, model_tag in MODELS:
            mean, lo, hi = curves[(domain_tag, model_tag)]
            ax.fill_between(x, lo, hi, color=COLOURS[model_tag], alpha=0.15, lw=0, zorder=2)
            ax.plot(
                x,
                mean,
                color=COLOURS[model_tag],
                marker=MARKERS[model_tag],
                ms=4,
                lw=1.6,
                label=model_name,
                zorder=3,
            )
        ax.set_title(domain_name, fontsize=9)
        ax.set_xlabel("visual degradation severity $v$", fontsize=8)
        ax.set_xticks(x)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"$\Delta_A(v)$  (pp)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    # A legend inside either panel lands on a curve, so it goes above the axes.
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "figures" / "fig1_delta_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_delivery(gains: dict[tuple[str, str], tuple[float, float, float]]) -> None:
    """Audio delivery gain against the registered 20 pp bar, with CIs."""
    labels: list[str] = []
    means: list[float] = []
    errs: list[list[float]] = [[], []]
    colours: list[str] = []
    for domain_name, domain_tag in DOMAINS:
        for _model_name, model_tag in MODELS:
            mean, lo, hi = gains[(domain_tag, model_tag)]
            short = "vSALMONN2+" if model_tag == "vsalm2" else "VideoLLaMA2.1"
            labels.append(f"{short}\n{domain_name}")
            means.append(mean)
            errs[0].append(mean - lo)
            errs[1].append(hi - mean)
            colours.append(COLOURS[model_tag])

    fig, ax = plt.subplots(figsize=(3.4, 2.4))  # one IEEE column
    pos = np.arange(len(means))
    ax.bar(pos, means, color=colours, alpha=0.85, width=0.6, zorder=2)
    ax.errorbar(pos, means, yerr=errs, fmt="none", ecolor="0.2", elinewidth=1.0, capsize=3, zorder=3)
    ax.axhline(20.0, color="#009E73", lw=1.2, ls="--", zorder=4)
    ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)
    ax.text(3.45, 24.0, "20 pp bar", fontsize=6, color="#009E73", ha="right")  # over the empty bar
    for p, m, up in zip(pos, means, errs[1], strict=True):
        ax.text(p, m + up + 2.5, f"{m:+.2f}", ha="center", fontsize=6)  # clear of the whisker cap
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("audio delivery gain (pp)", fontsize=7)
    ax.set_ylim(-9, 86)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "figures" / "fig2_delivery.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_redundancy(single: dict[tuple[str, str], dict[str, float]]) -> None:
    """Single-channel versus joint accuracy: how much headroom the benchmark leaves."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))  # one IEEE column
    groups = [(d_name, d_tag, m_name, m_tag) for d_name, d_tag in DOMAINS for m_name, m_tag in MODELS]
    width = 0.22
    pos = np.arange(len(groups))
    series = (
        ("audio only", "audio_only", "#56B4E9"),
        ("vision only", "vision_only", "#E69F00"),
        ("both", "both", "#009E73"),
    )
    for offset, (label, key, colour) in zip((-width, 0.0, width), series, strict=True):
        vals = [single[(d_tag, m_tag)][key] for _, d_tag, _, m_tag in groups]
        ax.bar(pos + offset, vals, width=width, label=label, color=colour, zorder=2)
    ax.axhline(CHANCE, color="0.35", lw=1.0, ls=":", zorder=3)
    ax.text(-0.48, CHANCE + 2.0, "chance", fontsize=6, color="0.35")
    ax.set_xticks(pos)
    short = {"vsalm2": "vSALMONN2+", "vl2": "VideoLLaMA2.1"}
    ax.set_xticklabels([f"{short[m_tag]}\n{d_name}" for d_name, _, _, m_tag in groups], fontsize=5.5)
    ax.set_ylabel("accuracy (%)", fontsize=7)
    ax.set_ylim(0, 118)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", columnspacing=1.0, handlelength=1.2)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "figures" / "fig3_redundancy.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Compute every quoted number, write the macros, and render the figures."""
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    macros: list[str] = []

    def macro(name: str, value: str) -> None:
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    curves: dict[tuple[str, str], tuple[list[float], list[float], list[float]]] = {}
    gains: dict[tuple[str, str], tuple[float, float, float]] = {}
    single: dict[tuple[str, str], dict[str, float]] = {}

    for _domain_name, domain_tag in DOMAINS:
        for _model_name, model_tag in MODELS:
            key = (domain_tag, model_tag)
            curves[key] = delta_curve(domain_tag, model_tag)

            # Delivery: paired difference of the two blank-video conditions.
            real_ok, videos = correctness(poscontrol(domain_tag, model_tag, "real"))
            silent_ok, _ = correctness(poscontrol(domain_tag, model_tag, "silent"))
            index = {v: i for i, v in enumerate(sorted(set(videos)))}
            cid = np.array([index[v] for v in videos])
            gains[key] = cluster_bootstrap_mean((real_ok - silent_ok) * 100.0, cid, n_boot=N_BOOT, seed=SEED)

            single[key] = {
                "audio_only": float(poscontrol(domain_tag, model_tag, "real")["accuracy"]),  # type: ignore[arg-type]
                "floor": float(poscontrol(domain_tag, model_tag, "silent")["accuracy"]),  # type: ignore[arg-type]
                "vision_only": float(cell(domain_tag, model_tag, "silent")["accuracy"]),  # type: ignore[arg-type]
                "both": float(cell(domain_tag, model_tag, "real")["accuracy"]),  # type: ignore[arg-type]
            }

            tag = f"{'Avut' if domain_tag else 'Music'}{'Vsalm' if model_tag == 'vsalm2' else 'Vltwo'}"
            mean, lo, hi = gains[key]
            macro(f"Deliv{tag}", f"{mean:+.2f}")
            macro(f"DelivCI{tag}", f"[{lo:+.2f}, {hi:+.2f}]")
            macro(f"AudioOnly{tag}", f"{single[key]['audio_only']:.2f}")
            macro(f"VisionOnly{tag}", f"{single[key]['vision_only']:.2f}")
            macro(f"Both{tag}", f"{single[key]['both']:.2f}")
            macro(f"Floor{tag}", f"{single[key]['floor']:.2f}")
            macro(f"DeltaClean{tag}", f"{curves[key][0][0]:+.2f}")
            macro(f"DeltaSevere{tag}", f"{curves[key][0][-1]:+.2f}")
            headroom = 100.0 * (single[key]["audio_only"] - single[key]["floor"]) / (100.0 - single[key]["floor"])
            macro(f"Headroom{tag}", f"{headroom:.1f}")

    (OUT_DIR / "numbers.tex").write_text("\n".join(sorted(macros)) + "\n")
    figure_delta_curves(curves)
    figure_delivery(gains)
    figure_redundancy(single)

    print(f"wrote {OUT_DIR / 'numbers.tex'} ({len(macros)} macros)")
    for name in ("fig1_delta_curves", "fig2_delivery", "fig3_redundancy"):
        print(f"wrote {OUT_DIR / 'figures' / (name + '.pdf')}")
    print()
    for domain_name, domain_tag in DOMAINS:
        for model_name, model_tag in MODELS:
            mean, lo, hi = gains[(domain_tag, model_tag)]
            curve = curves[(domain_tag, model_tag)][0]
            print(
                f"{domain_name:11s} {model_name:18s} delivery {mean:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  "
                f"Delta_A {curve[0]:+6.2f} -> {curve[-1]:+6.2f}"
            )


if __name__ == "__main__":
    main()
