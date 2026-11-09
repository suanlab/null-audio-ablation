"""Figures 1 and 2 of the restructured paper, computed from the evaluation cells.

Figure 1 is the crossover: audio delivery gain for two audio front ends across two
benchmarks whose audio is of different kinds. An interaction plot is used rather than
grouped bars because the claim *is* the crossing -- each front end extracts audio
information only in its own domain -- and crossing lines carry that at a glance.

Figure 2 separates two things a standard ablation reports identically. Delivery gain is
what audio alone can contribute; ``Delta_A(0)`` is what it contributes on top of vision.
Points on the diagonal use everything they were given; the vertical distance below it is
audio information that reached the model and was discarded in fusion.

Numbers are recomputed here rather than copied, and written to ``numbers_crossover.tex``
so the manuscript cites macros instead of digits.

Usage::

    python scripts/make_crossover_figures.py
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
OUT_DIR = REPO_ROOT / "paper_taslp"
N_BOOT = 10000
SEED = 2027
GATE_PP = 20.0
FAMILIES = ("motion_blur", "occlusion", "frame_drop", "downscale")

# (display, front end, filename tag) -- the tag differs per domain by construction.
# Series order for the figures: systems, not front-end families.
CELLS_ORDERED = (
    ("MUSIC-AVQA", "music", "video-SALMONN 2+", "Whisper", "cmss_music_vsalm2"),
    ("MUSIC-AVQA", "music", "VideoLLaMA2.1-AV", "BEATs", "cmss_music_vl2"),
    ("MUSIC-AVQA", "music", "Qwen2.5-Omni", "Whisper", "cmss_music_qwenomni"),
)

CELLS = (
    ("MUSIC-AVQA", "music", "VideoLLaMA2.1-AV", "BEATs", "cmss_music_vl2"),
    ("MUSIC-AVQA", "music", "video-SALMONN 2+", "Whisper", "cmss_music_vsalm2"),
    ("AVUT", "speech", "VideoLLaMA2.1-AV", "BEATs", "cmss_avut_vl2"),
    ("AVUT", "speech", "video-SALMONN 2+", "Whisper", "cmss_avut_vsalm2"),
    ("MUSIC-AVQA", "music", "Qwen2.5-Omni", "Whisper", "cmss_music_qwenomni"),
    ("AVUT", "speech", "Qwen2.5-Omni", "Whisper", "cmss_avut_qwenomni"),
)
# Colour carries the front-end family, marker carries the system. Two systems now share
# the Whisper family and behave differently, which is the paper's point: same colour,
# different marker states it on the page. Keying the figures on the family alone would be
# worse than a colour clash -- the row lookups below use next(), so the second Whisper
# system would be silently dropped from every panel.
FRONTEND_COLOUR = {"Whisper": "#0072B2", "BEATs": "#D55E00"}
SYSTEM_MARKER = {"video-SALMONN 2+": "o", "VideoLLaMA2.1-AV": "s", "Qwen2.5-Omni": "^"}


# Macro suffix per system. Keyed on the system, not the front-end family: two systems now
# share the Whisper family, and a family-keyed suffix silently overwrote video-SALMONN 2+'s
# macros with Qwen's -- the body would have cited one model's numbers under the other's
# name with no build error. The first two keep their historical suffixes so existing
# \DelivMusicWhisper-style references in the body stay valid.
SYSTEM_SUFFIX = {
    "video-SALMONN 2+": "Whisper",
    "VideoLLaMA2.1-AV": "BEATs",
    "Qwen2.5-Omni": "Qwen",
}


def macro_suffix(domain: str, model: str) -> str:
    """Unique macro suffix for one model x domain cell."""
    if model not in SYSTEM_SUFFIX:
        raise ValueError(f"no macro suffix registered for system {model!r}")
    return f"{'Music' if domain == 'MUSIC-AVQA' else 'Avut'}{SYSTEM_SUFFIX[model]}"


def style_for(model: str, frontend: str) -> tuple[str, str]:
    """Colour from the front-end family, marker from the system."""
    if frontend not in FRONTEND_COLOUR:
        raise ValueError(f"no colour registered for front end {frontend!r}")
    if model not in SYSTEM_MARKER:
        raise ValueError(f"no marker registered for system {model!r}")
    return FRONTEND_COLOUR[frontend], SYSTEM_MARKER[model]

# (domain, model tag, macro suffix) for the per-model macro generators below. Qwen2.5-Omni
# is a Whisper-family front end like video-SALMONN 2+, so its suffix names the system, not
# the family: the two dissociate, which is the point. Generators skip a row whose cells are
# absent, so listing a model before its grid finishes is safe.
MODEL_CELLS = (
    ("music", "vl2", "MusicBEATs"),
    ("music", "vsalm2", "MusicWhisper"),
    ("music", "qwenomni", "MusicQwen"),
    ("avut", "vl2", "AvutBEATs"),
    ("avut", "vsalm2", "AvutWhisper"),
    ("avut", "qwenomni", "AvutQwen"),
)


def correctness(path: Path) -> tuple[np.ndarray, list[str]]:
    """Per-item correctness in {0,1} and the video each item belongs to."""
    preds = json.loads(path.read_text())["predictions"]
    ok = np.array([1.0 if str(p["correct"]) == "True" else 0.0 for p in preds])
    return ok, [str(p["video"]) for p in preds]


def paired_gain(a: Path, b: Path) -> tuple[float, float, float]:
    """Cluster-bootstrapped mean of (a - b) in percentage points."""
    x, videos = correctness(a)
    y, videos_b = correctness(b)
    if videos != videos_b:
        raise ValueError(f"item order differs between {a.name} and {b.name}")
    index = {v: i for i, v in enumerate(sorted(set(videos)))}
    cid = np.array([index[v] for v in videos])
    return cluster_bootstrap_mean((x - y) * 100.0, cid, n_boot=N_BOOT, seed=SEED)


def collect() -> list[dict[str, object]]:
    """Delivery gain and Delta_A(0) for every model x domain cell that is on disk."""
    rows: list[dict[str, object]] = []
    for domain, audio_kind, model, frontend, tag in CELLS:
        deliv_a = EVAL_DIR / f"poscontrol_{tag}_real.json"
        deliv_b = EVAL_DIR / f"poscontrol_{tag}_silent.json"
        clean_a = EVAL_DIR / f"avqa_{tag}_real.json"
        clean_b = EVAL_DIR / f"avqa_{tag}_silent.json"
        if not all(p.exists() for p in (deliv_a, deliv_b, clean_a, clean_b)):
            print(f"  [skip] {model} / {domain}: cells missing")
            continue
        dm, dlo, dhi = paired_gain(deliv_a, deliv_b)
        am, alo, ahi = paired_gain(clean_a, clean_b)
        rows.append(
            {
                "domain": domain, "audio_kind": audio_kind, "model": model, "frontend": frontend,
                "deliv": dm, "deliv_lo": dlo, "deliv_hi": dhi,
                "dA": am, "dA_lo": alo, "dA_hi": ahi,
                "tag": tag,
                "floor": json.loads(deliv_b.read_text())["accuracy"],
                "audio_only": json.loads(deliv_a.read_text())["accuracy"],
                "vision_only": json.loads(clean_b.read_text())["accuracy"],
                "both": json.loads(clean_a.read_text())["accuracy"],
            }
        )
    return rows


def figure_crossover(rows: list[dict[str, object]]) -> None:
    """Interaction plot: delivery gain by domain, one line per audio front end."""
    domains = ["MUSIC-AVQA", "AVUT"]
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--", zorder=1)
    ax.axhline(GATE_PP, color="#009E73", lw=1.1, ls=":", zorder=1)
    ax.text(0.5, GATE_PP + 0.8, "20 pp gate", fontsize=6, color="#009E73", ha="center")
    # Point labels are placed per system and domain: on MUSIC-AVQA two systems sit near the
    # gate and on AVUT the two Whisper-family systems sit within 1 pp of each other.
    label_offsets = {
        ("Qwen2.5-Omni", "MUSIC-AVQA"): (-24, 5), ("VideoLLaMA2.1-AV", "MUSIC-AVQA"): (-24, -4),
        ("video-SALMONN 2+", "MUSIC-AVQA"): (0, -13), ("video-SALMONN 2+", "AVUT"): (24, 5),
        ("Qwen2.5-Omni", "AVUT"): (24, -5), ("VideoLLaMA2.1-AV", "AVUT"): (0, -13),
    }

    for model in [c[2] for c in CELLS_ORDERED]:
        pts = [next((r for r in rows if r["model"] == model and r["domain"] == d), None) for d in domains]
        if any(p is None for p in pts):
            continue
        frontend = str(pts[0]["frontend"])  # type: ignore[index]
        colour, marker = style_for(model, frontend)
        xs = np.arange(len(domains))
        ys = [float(p["deliv"]) for p in pts]  # type: ignore[index]
        err = [[float(p["deliv"]) - float(p["deliv_lo"]) for p in pts],  # type: ignore[index]
               [float(p["deliv_hi"]) - float(p["deliv"]) for p in pts]]  # type: ignore[index]
        ax.errorbar(xs, ys, yerr=err, color=colour, marker=marker, ms=5, lw=1.8,
                    capsize=3, elinewidth=1.0, label=f"{model} ({frontend})", zorder=3)
        for x, y, d in zip(xs, ys, domains, strict=True):
            dx, dy = label_offsets[(model, d)]
            ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=6.5, color=colour)

    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels([f"{d}\n({'music' if d == 'MUSIC-AVQA' else 'speech'} audio)" for d in domains], fontsize=7)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("audio delivery gain (pp)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    # The top must clear the highest CI (Qwen2.5-Omni / MUSIC-AVQA reaches ~28 pp), and the
    # legend sits below the axes so it cannot cover the near-zero point labels.
    top = max(float(r["deliv_hi"]) for r in rows) + 2.5
    ax.set_ylim(-7.0, max(27.0, top))
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "figures" / "fig1_crossover.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_fusion(rows: list[dict[str, object]]) -> None:
    """Delivery gain against Delta_A(0), with the marginal gap between them drawn as a contrast.

    The vertical gap G_A - Delta_A(0) is a difference of two marginal accuracies. It is not
    discarded information and not a fusion loss: it is labelled as the contrast it is, and the
    diagonal is a reference line, not a ceiling. Delta_A(0) can exceed G_A where the two
    channels are complementary.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    lim = (-8.0, 26.0)
    ax.plot(lim, lim, color="0.55", lw=0.9, ls="--", zorder=1)
    ax.annotate(r"$\Delta_A(0) = G_A$", (17.0, 17.0), rotation=34, fontsize=6,
                color="0.4", ha="center", va="bottom")
    ax.axhline(0.0, color="0.8", lw=0.7, zorder=0)
    ax.axvline(0.0, color="0.8", lw=0.7, zorder=0)

    # Labels name the domain only; the legend names the system. Offsets are per system and
    # domain because the two Whisper-family systems sit on top of each other on AVUT.
    offsets = {
        ("video-SALMONN 2+", "AVUT"): (-8, 6, "right"),
        ("Qwen2.5-Omni", "AVUT"): (8, 6, "left"),
        ("VideoLLaMA2.1-AV", "MUSIC-AVQA"): (0, -14, "center"),
        ("Qwen2.5-Omni", "MUSIC-AVQA"): (0, -14, "center"),
        ("video-SALMONN 2+", "MUSIC-AVQA"): (0, 13, "center"),
        ("VideoLLaMA2.1-AV", "AVUT"): (0, -15, "center"),
    }
    seen: set[str] = set()
    for row in rows:
        model, domain = str(row["model"]), str(row["domain"])
        colour, marker = style_for(model, str(row["frontend"]))
        x, y = float(row["deliv"]), float(row["dA"])
        if x - y > 5.0:
            ax.plot([x, x], [x, y], color=colour, lw=1.0, ls=":", zorder=2)
            # The two gaps are 4 pp apart, so the left one is labelled on its left.
            left = x < max(float(r["deliv"]) for r in rows) - 1.0
            # The left label sits low on its gap so it clears the AVUT points' error bars.
            ax.annotate(f"$G_A-\\Delta_A(0)$\n{x - y:.2f} pp", (x, y + (0.3 if left else 0.5) * (x - y)),
                        textcoords="offset points",
                        xytext=(-4 if left else 4, 0), ha="right" if left else "left", va="center",
                        fontsize=6, color=colour)
        ax.errorbar(x, y, xerr=[[x - float(row["deliv_lo"])], [float(row["deliv_hi"]) - x]],
                    yerr=[[y - float(row["dA_lo"])], [float(row["dA_hi"]) - y]],
                    fmt=marker, color=colour, ms=6, capsize=2, elinewidth=0.8, zorder=3,
                    label=f"{model} ({row['frontend']})" if model not in seen else None)
        seen.add(model)
        if (model, domain) not in offsets:
            raise ValueError(f"no label offset registered for {model} / {domain}")
        dx, dy, ha = offsets[(model, domain)]
        ax.annotate(domain, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6, color=colour, va="center", ha=ha)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              columnspacing=0.8, handletextpad=0.3)

    ax.set_xlim(-11.0, 30.0)
    ax.set_ylim(-11.0, max(21.0, max(float(r["deliv"]) for r in rows) + 1.5))  # show each gap's top
    ax.set_xlabel("audio delivery gain $G_A$ (pp), blank video", fontsize=7.5)
    ax.set_ylabel(r"$\Delta_A(0)$ (pp), clean video", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "figures" / "fig2_fusion.pdf", bbox_inches="tight")
    plt.close(fig)


def _paired_vector(a: Path, b: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-item (a - b) in pp and the matching video-cluster ids."""
    x, videos = correctness(a)
    y, videos_b = correctness(b)
    if videos != videos_b:
        raise ValueError(f"item order differs between {a.name} and {b.name}")
    index = {v: i for i, v in enumerate(sorted(set(videos)))}
    return (x - y) * 100.0, np.array([index[v] for v in videos])


def _pooled_severe(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-item Delta_A at severity 1, averaged over the four families, and its cluster ids."""
    per_family = []
    cid: np.ndarray | None = None
    for family in FAMILIES:
        diff, ids = _paired_vector(EVAL_DIR / f"avqa_{tag}_real_vis-{family}-s1.json",
                                   EVAL_DIR / f"avqa_{tag}_silent_vis-{family}-s1.json")
        per_family.append(diff)
        cid = ids
    assert cid is not None
    return np.stack(per_family).mean(axis=0), cid


def _interaction_macros() -> list[str]:
    """Interaction contrast at matched audio coverage, estimate and interval together.

    ``(G_Whisper,AVUT - G_Whisper,MUSIC) - (G_BEATs,AVUT - G_BEATs,MUSIC)``. The BEATs arm
    uses the full-coverage delivery cells so that both arms see the whole track; the
    original 8x2 s sampling gave the BEATs-fronted model roughly a quarter of the audio the
    Whisper-fronted one received, which is confounded with the effect being measured.
    """
    # Within a domain the two systems answer the same items, so the per-item contrast
    # (G_Whisper - G_BEATs) is formed first and its video clusters resampled once; drawing
    # the two systems independently would treat paired outputs as independent. The two
    # domains are independent samples and are resampled separately.
    rng = np.random.default_rng(SEED)

    def domain_contrast(domain: str) -> tuple[np.ndarray, float]:
        w, cid = _paired_vector(EVAL_DIR / f"poscontrol_cmss_{domain}_vsalm2_real.json",
                                EVAL_DIR / f"poscontrol_cmss_{domain}_vsalm2_silent.json")
        bg, cid_b = _paired_vector(EVAL_DIR / f"full_poscontrol_cmss_{domain}_vl2_real.json",
                                   EVAL_DIR / f"full_poscontrol_cmss_{domain}_vl2_silent.json")
        if not np.array_equal(cid, cid_b):
            raise ValueError(f"{domain}: the two systems' delivery cells are not item-aligned")
        diff = w - bg
        n_clusters = int(cid.max()) + 1
        sums = np.zeros(n_clusters)
        counts = np.zeros(n_clusters)
        np.add.at(sums, cid, diff)
        np.add.at(counts, cid, 1)
        pick = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
        return sums[pick].sum(1) / counts[pick].sum(1), float(diff.mean())

    avut, avut_m = domain_contrast("avut")
    music, music_m = domain_contrast("music")
    draws = avut - music
    point = avut_m - music_m
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [
        f"\\newcommand{{\\InteractionMatched}}{{{point:+.2f}}}",
        f"\\newcommand{{\\InteractionCI}}{{[{lo:+.2f}, {hi:+.2f}]}}",
    ]


def _probe_macros() -> list[str]:
    """Probe numbers, emitted here so no macro file is maintained by hand.

    numbers_probe.tex was previously hand-written, which is how an interval computed at one
    audio-coverage condition came to sit beside an estimate computed at another. Everything
    the manuscript cites is now generated in a single pass from artefacts on disk.
    """
    path = EVAL_DIR / "encoder_probe_instrument.json"
    if not path.exists():
        return []
    probe = json.loads(path.read_text())
    beats = probe["encoders"]["BEATs"]
    whisper = probe["encoders"]["Whisper"]
    base = float(probe["majority_baseline"])
    return [
        f"\\newcommand{{\\ProbeItems}}{{{probe['items']}}}",
        f"\\newcommand{{\\ProbeClasses}}{{{probe['classes']}}}",
        f"\\newcommand{{\\ProbeBaseline}}{{{base:.1f}}}",
        f"\\newcommand{{\\ProbeBEATs}}{{{beats['mean']:.1f}}}",
        f"\\newcommand{{\\ProbeBEATsSd}}{{{beats['std']:.1f}}}",
        f"\\newcommand{{\\ProbeBEATsRatio}}{{{beats['mean'] / base:.1f}}}",
        f"\\newcommand{{\\ProbeWhisper}}{{{whisper['mean']:.1f}}}",
        f"\\newcommand{{\\ProbeWhisperSd}}{{{whisper['std']:.1f}}}",
        f"\\newcommand{{\\ProbeWhisperRatio}}{{{whisper['mean'] / base:.1f}}}",
        f"\\newcommand{{\\ProbeGap}}{{{beats['mean'] - whisper['mean']:.1f}}}",
    ] + _qwen_probe_macros(base) + _matched_probe_macros(base)


def _matched_probe_macros(base: float) -> list[str]:
    """The probe with all three encoders on the same first 30 s (run_encoder_probe_matched.py).

    The original probe gave BEATs the whole clip and Whisper/Qwen the first 30 s, so its numbers
    are not an encoder comparison. This one is, with intervals over clips and paired contrasts.
    """
    path = EVAL_DIR / "encoder_probe_instrument_matched.json"
    if not path.exists():
        return []
    probe = json.loads(path.read_text())
    if abs(float(probe["majority_baseline"]) - base) > 1e-6:
        raise ValueError("matched probe ran on a different clip set than the original probe")
    names = {"BEATs": "BEATs", "Whisper-large-v3": "Whisper", "Qwen2.5-Omni audio tower": "Qwen"}
    out: list[str] = []
    for key, tag in names.items():
        enc = probe["encoders"][key]
        lo, hi = enc["ci"]
        out += [f"\\newcommand{{\\ProbeM{tag}}}{{{enc['mean']:.1f}}}",
                f"\\newcommand{{\\ProbeMCI{tag}}}{{[{lo:.1f}, {hi:.1f}]}}",
                f"\\newcommand{{\\ProbeMRatio{tag}}}{{{enc['mean'] / base:.1f}}}"]
        sub = probe["label_audit"]["audio_typed_subset"][key]
        out.append(f"\\newcommand{{\\ProbeAud{tag}}}{{{sub['mean']:.1f}}}")
    for key, tag in (("BEATs - Whisper-large-v3", "BEATsWhisper"),
                     ("Qwen2.5-Omni audio tower - Whisper-large-v3", "QwenWhisper"),
                     ("BEATs - Qwen2.5-Omni audio tower", "BEATsQwen")):
        d = probe["paired_differences"][key]
        out += [f"\\newcommand{{\\ProbeMDiff{tag}}}{{{d['mean']:+.1f}}}",
                f"\\newcommand{{\\ProbeMDiffCI{tag}}}{{[{d['lo']:+.1f}, {d['hi']:+.1f}]}}"]
    types = probe["label_audit"]["label_question_types"]
    out += [f"\\newcommand{{\\ProbeAudItems}}{{{probe['label_audit']['audio_typed_subset']['items']}}}",
            f"\\newcommand{{\\ProbeVisualLabels}}{{{sum(n for t, n in types.items() if 'Audio' not in t)}}}",
            f"\\newcommand{{\\ProbeShortClips}}{{{probe['seconds_fed']['clips_shorter_than_window']}}}"]
    return out


def _qwen_probe_macros(base: float) -> list[str]:
    """Probe on Qwen2.5-Omni's own audio tower (scripts/run_encoder_probe_qwen.py).

    video-SALMONN 2+'s encoder is Whisper-large-v3 bit for bit, so the Whisper probe speaks for
    it; Qwen's tower was trained on from that initialisation, so it needs its own probe before
    the manuscript can say where the two systems' delivery gap lies.
    """
    path = EVAL_DIR / "encoder_probe_instrument_qwen.json"
    if not path.exists():
        return []
    probe = json.loads(path.read_text())
    if abs(float(probe["majority_baseline"]) - base) > 1e-6:
        raise ValueError("Qwen probe ran on a different clip set than the Whisper/BEATs probe")
    qwen = probe["encoders"]["Qwen2.5-Omni audio tower"]
    weights = EVAL_DIR / "encoder_weight_check.json"
    diff = json.loads(weights.read_text())["Qwen2.5-Omni"]["relative_difference"] if weights.exists() else None
    return ([f"\\newcommand{{\\EncoderDiffQwen}}{{{diff:.2f}}}"] if diff is not None else []) + [
        f"\\newcommand{{\\ProbeQwen}}{{{qwen['mean']:.1f}}}",
        f"\\newcommand{{\\ProbeQwenSd}}{{{qwen['std']:.1f}}}",
        f"\\newcommand{{\\ProbeQwenRatio}}{{{qwen['mean'] / base:.1f}}}",
    ]


def _matched_coverage_macros() -> list[str]:
    """Delivery gains for the BEATs-fronted model at full audio coverage.

    The graded grid was run on the default sampled path, so the manuscript reports that
    path in its tables. Matched coverage exists only for the delivery cells, and the two
    conditions must be distinguishable in the text: reporting whichever is more favourable
    per claim is the selective practice this paper argues against.
    """
    out: list[str] = []
    for domain in ("music", "avut"):
        real = EVAL_DIR / f"full_poscontrol_cmss_{domain}_vl2_real.json"
        silent = EVAL_DIR / f"full_poscontrol_cmss_{domain}_vl2_silent.json"
        if not (real.exists() and silent.exists()):
            continue
        mean, lo, hi = paired_gain(real, silent)
        tag = "Music" if domain == "music" else "Avut"
        out += [
            f"\\newcommand{{\\FullDeliv{tag}}}{{{mean:+.2f}}}",
            f"\\newcommand{{\\FullDelivCI{tag}}}{{[{lo:+.2f}, {hi:+.2f}]}}",
        ]
        # Fusion loss at matched coverage, derived here rather than in the prose: a
        # hand-entered derived digit is what desynchronised the interaction last round.
        clean_real = EVAL_DIR / f"avqa_cmss_{domain}_vl2_real.json"
        clean_silent = EVAL_DIR / f"avqa_cmss_{domain}_vl2_silent.json"
        if clean_real.exists() and clean_silent.exists():
            d_clean, _, _ = paired_gain(clean_real, clean_silent)
            out.append(f"\\newcommand{{\\FullFusionLoss{tag}}}{{{mean - d_clean:.2f}}}")
    return out


def _shuffled_macros() -> list[str]:
    """Delivery gain measured against SHUFFLED audio instead of silence.

    Fusion loss depends on G_A, which compares real audio with silence. If silence itself
    degraded the model, G_A would be inflated and the "discarded information" would be an
    artefact of the null rather than a property of fusion. Shuffled audio -- a donor clip's
    real waveform, matched to this item's duration -- has the acoustic character of the real
    condition and none of its answer content, so a gain that survives it is content-driven.
    """
    out: list[str] = []
    for domain, model, tag in MODEL_CELLS:
        cell = EVAL_DIR / f"shuf_poscontrol_cmss_{domain}_{model}.json"
        real = EVAL_DIR / f"poscontrol_cmss_{domain}_{model}_real.json"
        if not (cell.exists() and real.exists()):
            continue
        mean, lo, hi = paired_gain(real, cell)
        out += [
            f"\\newcommand{{\\ShufDeliv{tag}}}{{{mean:+.2f}}}",
            f"\\newcommand{{\\ShufDelivCI{tag}}}{{[{lo:+.2f}, {hi:+.2f}]}}",
        ]
    return out


def _census_macros() -> list[str]:
    """Cell counts and the small-sample replication, generated so prose cannot drift.

    Two numbers in this paper were previously typed by hand and both went stale as cells
    were added: the census of cells cited, and the size of the replication used to argue
    that the delivery gate's straddle is not a sample-size problem. Deriving them here keeps
    them true.
    """
    import glob as _glob

    paths = set()
    for pattern in ("avqa_cmss_music_*.json", "poscontrol_cmss_music_*.json",
                    "avqa_cmss_avut_*.json", "poscontrol_cmss_avut_*.json",
                    "full_poscontrol_*.json", "shuf_poscontrol_*.json"):
        paths |= set(_glob.glob(str(EVAL_DIR / pattern)))
    for extra in ("poscontrol_cmss_vl2_real.json", "poscontrol_cmss_vl2_silent.json",
                  "contig_poscontrol_cmss_vl2_real.json", "contig_poscontrol_cmss_avut_vl2_real.json"):
        if (EVAL_DIR / extra).exists():
            paths.add(str(EVAL_DIR / extra))
    total = sum(len(json.loads(Path(p).read_text())["predictions"]) for p in paths)

    out = [
        f"\\newcommand{{\\CitedCells}}{{{len(paths)}}}",
        f"\\newcommand{{\\CitedPredictions}}{{{total:,}}}".replace(",", "{,}"),
    ]

    # The grid proper: three systems x two domains x 28 cells. Counted here for the same
    # reason as the census above -- the body previously carried "112-cell grid" and an
    # unparseable count as literals, and a third system silently made both wrong.
    grid = set()
    for dom in ("music", "avut"):
        for model in ("vsalm2", "vl2", "qwenomni"):
            grid |= set(_glob.glob(str(EVAL_DIR / f"avqa_cmss_{dom}_{model}*.json")))
            grid |= set(_glob.glob(str(EVAL_DIR / f"poscontrol_cmss_{dom}_{model}_*.json")))
    grid_preds = 0
    grid_unparsed = 0
    for path in grid:
        preds = json.loads(Path(path).read_text())["predictions"]
        grid_preds += len(preds)
        grid_unparsed += sum(1 for r in preds if not str(r["pred"]).strip())
    out += [
        f"\\newcommand{{\\GridCells}}{{{len(grid)}}}",
        f"\\newcommand{{\\GridPredictions}}{{{grid_preds:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\GridUnparsed}}{{{grid_unparsed}}}",
        f"\\newcommand{{\\GridUnparsedPct}}{{{100.0 * grid_unparsed / max(grid_preds, 1):.3f}}}",
    ]
    rep_a = EVAL_DIR / "pre_music_vl2_blank_real.json"
    rep_b = EVAL_DIR / "pre_music_vl2_blank_silent.json"
    if rep_a.exists() and rep_b.exists():
        mean, lo, hi = paired_gain(rep_a, rep_b)
        _, videos = correctness(rep_a)
        out += [
            f"\\newcommand{{\\RepItems}}{{{len(videos)}}}",
            f"\\newcommand{{\\RepVideos}}{{{len(set(videos))}}}",
            f"\\newcommand{{\\RepGain}}{{{mean:+.2f}}}",
            f"\\newcommand{{\\RepCI}}{{[{lo:+.2f}, {hi:+.2f}]}}",
        ]
    # Qwen2.5-Omni's registered entry gate (amendment of 2026-09-14): G_A significantly
    # positive on the same 60 held-out MUSIC-AVQA items, run before its grid.
    gate_a = EVAL_DIR / "pre_music_qwenomni_blank_real.json"
    gate_b = EVAL_DIR / "pre_music_qwenomni_blank_silent.json"
    if gate_a.exists() and gate_b.exists():
        mean, lo, hi = paired_gain(gate_a, gate_b)
        out += [
            f"\\newcommand{{\\QwenGate}}{{{mean:+.2f}}}",
            f"\\newcommand{{\\QwenGateCI}}{{[{lo:+.2f}, {hi:+.2f}]}}",
        ]
    # Visual delivery gain (clean silent - blank silent): M1's second eligibility condition.
    for domain, model, tag in MODEL_CELLS:
        vision = EVAL_DIR / f"avqa_cmss_{domain}_{model}_silent.json"
        floor = EVAL_DIR / f"poscontrol_cmss_{domain}_{model}_silent.json"
        if vision.exists() and floor.exists():
            gain = json.loads(vision.read_text())["accuracy"] - json.loads(floor.read_text())["accuracy"]
            out.append(f"\\newcommand{{\\VisDeliv{tag}}}{{{gain:+.2f}}}")
    return out


def _degeneration_macros() -> list[str]:
    """How much of the delivery gain is content, and how much is silence de-degenerating.

    G_A compares real audio with silence. If silence itself collapses the model's output
    distribution, part of G_A is the model recovering from that collapse rather than using
    audio content. Shuffled audio -- a donor clip's waveform, matched in duration, carrying
    no answer -- has the acoustic character of the real condition, so `shuffled - silent`
    isolates the de-degeneration component and `real - shuffled` the content component.

    Reporting only `real - shuffled`, as an earlier draft did, hides the decomposition and
    happens to flatter the headline: on the anchor cell 3.00 of the 18.00 pp is
    de-degeneration, and in the AVUT cell shuffled audio is an active distractor, so the
    content component there *exceeds* the silence-referenced gain.
    """
    out: list[str] = []
    for domain, model, tag in MODEL_CELLS:
        shuf = EVAL_DIR / f"shuf_poscontrol_cmss_{domain}_{model}.json"
        silent = EVAL_DIR / f"poscontrol_cmss_{domain}_{model}_silent.json"
        clean_real = EVAL_DIR / f"avqa_cmss_{domain}_{model}_real.json"
        clean_silent = EVAL_DIR / f"avqa_cmss_{domain}_{model}_silent.json"
        if not (shuf.exists() and silent.exists()):
            continue
        mean, lo, hi = paired_gain(shuf, silent)
        out += [
            f"\\newcommand{{\\Degen{tag}}}{{{mean:+.2f}}}",
            f"\\newcommand{{\\DegenCI{tag}}}{{[{lo:+.2f}, {hi:+.2f}]}}",
        ]
        if clean_real.exists() and clean_silent.exists():
            # Fusion loss referenced to shuffled rather than silence.
            shuf_gain, _, _ = paired_gain(EVAL_DIR / f"poscontrol_cmss_{domain}_{model}_real.json", shuf)
            d_clean, _, _ = paired_gain(clean_real, clean_silent)
            out.append(f"\\newcommand{{\\FusionLossShuf{tag}}}{{{shuf_gain - d_clean:.2f}}}")
    return out


def _access_macros() -> list[str]:
    """The audio-access intervention, generated rather than transcribed.

    Three numbers in the access section were still typed by hand: the AVUT gain under the
    default sampled path, the same cell under a contiguous 30 s window, and the short-clip
    inertness control that must be identical under both paths. The last is the one that
    licenses reading the AVUT result as a front-end effect, so it is exactly the number that
    should not depend on a human copying it correctly.
    """
    out: list[str] = []
    pairs = {
        "AvutSampled": ("poscontrol_cmss_avut_vl2_real.json", "poscontrol_cmss_avut_vl2_silent.json"),
        "AvutContig": ("contig_poscontrol_cmss_avut_vl2_real.json", "poscontrol_cmss_avut_vl2_silent.json"),
        "ShortClipSampled": ("poscontrol_cmss_vl2_real.json", "poscontrol_cmss_vl2_silent.json"),
        "ShortClipContig": ("contig_poscontrol_cmss_vl2_real.json", "poscontrol_cmss_vl2_silent.json"),
    }
    for tag, (a, b) in pairs.items():
        pa, pb = EVAL_DIR / a, EVAL_DIR / b
        if not (pa.exists() and pb.exists()):
            continue
        mean, _lo, _hi = paired_gain(pa, pb)
        out.append(f"\\newcommand{{\\Access{tag}}}{{{mean:+.2f}}}")
    return out


def _contrast_macros() -> list[str]:
    """Item-paired Delta_A contrast between the two Whisper-family systems, per cell.

    The revised claim in Sec. V rests on these: the two systems that share an encoder
    family separate on music and not on speech. Stating it from two conditions in prose
    invites the reader to wonder about the other six, so all eight are generated here and
    tabulated. Contrasts are differenced within item before bootstrapping, which is what
    makes them a test of a difference rather than a comparison of two intervals.
    """
    out: list[str] = []
    for domain in ("music", "avut"):
        for family in FAMILIES:
            paths = {}
            for model in ("qwenomni", "vsalm2"):
                paths[model] = (
                    EVAL_DIR / f"avqa_cmss_{domain}_{model}_real_vis-{family}-s1.json",
                    EVAL_DIR / f"avqa_cmss_{domain}_{model}_silent_vis-{family}-s1.json",
                )
            if not all(f.exists() for pair in paths.values() for f in pair):
                continue
            diffs = {}
            videos = None
            for model, (real_p, silent_p) in paths.items():
                real, vids = correctness(real_p)
                silent, _ = correctness(silent_p)
                if videos is None:
                    videos = vids
                elif vids != videos:
                    raise ValueError(f"{domain}/{family}: cells are not item-aligned across models")
                diffs[model] = real - silent
            index = {v: i for i, v in enumerate(sorted(set(videos or [])))}
            cid = np.array([index[v] for v in (videos or [])])
            mean, lo, hi = cluster_bootstrap_mean(
                (diffs["qwenomni"] - diffs["vsalm2"]) * 100.0, cid, n_boot=N_BOOT, seed=SEED
            )
            tag = f"{'Music' if domain == 'music' else 'Avut'}{family.title().replace('_', '')}"
            out += [
                f"\\newcommand{{\\Contrast{tag}}}{{{mean:+.2f}}}",
                f"\\newcommand{{\\ContrastCI{tag}}}{{[{lo:+.2f}, {hi:+.2f}]}}",
            ]
    return out


def _ceiling_macros() -> list[str]:
    """Item-level selection headroom C, its blank-channel floor C0, and the net-gain split.

    ``G_A - Delta_A(0)`` is a difference of two *marginal* accuracies and is silent on
    whether the items audio answers alone are the items vision already answers. C is the
    share audio alone gets right and vision alone gets wrong: what choosing the better
    channel item by item could add. It bounds selection, not integration. In item counts
    the net gain splits exactly as Delta_A(0) = H + S - L: H items of C recovered with both
    channels, S synergy items neither channel answers alone, L items vision alone answered
    that the joint input loses. Delta_A(0)/C mixes all three; H/C is the recovery rate.
    """
    out: list[str] = []
    for domain, model, tag in MODEL_CELLS:
        audio = EVAL_DIR / f"poscontrol_cmss_{domain}_{model}_real.json"
        vision = EVAL_DIR / f"avqa_cmss_{domain}_{model}_silent.json"
        both = EVAL_DIR / f"avqa_cmss_{domain}_{model}_real.json"
        if not all(f.exists() for f in (audio, vision, both)):
            continue
        a, _ = correctness(audio)
        v, _ = correctness(vision)
        b, videos = correctness(both)
        unique = (a == 1) & (v == 0)          # audio answers, vision does not
        ceiling = 100.0 * unique.mean()
        realised = 100.0 * (b - v).mean()
        index = {x: i for i, x in enumerate(sorted(set(videos)))}
        cid = np.array([index[x] for x in videos])
        _m, lo, hi = cluster_bootstrap_mean(unique * 100.0, cid, n_boot=N_BOOT, seed=SEED)
        # C counts items audio-only gets right by guessing too. The same count with no
        # information at all -- blank video and silence -- is its chance analogue, C0;
        # C - C0 is the audio-unique signal beyond guessing, bootstrapped item-paired.
        floor_path = EVAL_DIR / f"poscontrol_cmss_{domain}_{model}_silent.json"
        f, _ = correctness(floor_path)
        chance = (f == 1) & (v == 0)
        excess_m, excess_lo, excess_hi = cluster_bootstrap_mean(
            (unique.astype(float) - chance.astype(float)) * 100.0, cid, n_boot=N_BOOT, seed=SEED)
        recovered = unique & (b == 1)
        synergy = (b == 1) & (a == 0) & (v == 0)
        harm = (v == 1) & (b == 0)
        if int(recovered.sum() + synergy.sum() - harm.sum()) != int((b - v).sum()):
            raise ValueError(f"{domain}/{model}: net gain does not decompose as H + S - L")
        out += [
            f"\\newcommand{{\\Recovered{tag}}}{{{100.0 * recovered.sum() / max(int(unique.sum()), 1):.0f}}}",
            f"\\newcommand{{\\RecoveredN{tag}}}{{{int(recovered.sum())}}}",
            f"\\newcommand{{\\SynergyN{tag}}}{{{int(synergy.sum())}}}",
            f"\\newcommand{{\\HarmN{tag}}}{{{int(harm.sum())}}}",
            f"\\newcommand{{\\Harm{tag}}}{{{100.0 * harm.mean():.2f}}}",
            f"\\newcommand{{\\NetN{tag}}}{{{int((b - v).sum()):+d}}}",
            f"\\newcommand{{\\Ceiling{tag}}}{{{ceiling:.2f}}}",
            f"\\newcommand{{\\CeilingCI{tag}}}{{[{lo:.2f}, {hi:.2f}]}}",
            f"\\newcommand{{\\CeilingN{tag}}}{{{int(unique.sum())}}}",
            f"\\newcommand{{\\CeilingChance{tag}}}{{{100.0 * chance.mean():.2f}}}",
            f"\\newcommand{{\\CeilingChanceN{tag}}}{{{int(chance.sum())}}}",
            # Items neither channel answers alone but both together do: what lets Delta_A(0)
            # exceed C in principle, so C bounds per-item channel selection, not integration.
            f"\\newcommand{{\\Synergy{tag}}}{{{100.0 * synergy.mean():.2f}}}",
            f"\\newcommand{{\\CeilingExcess{tag}}}{{{excess_m:+.2f}}}",
            f"\\newcommand{{\\CeilingExcessCI{tag}}}{{[{excess_lo:+.2f}, {excess_hi:+.2f}]}}",
            # A realisation ratio is reported only where the headroom clears its blank-channel floor:
            # below that the denominator is guessing and the ratio is noise over noise.
            f"\\newcommand{{\\Realised{tag}}}{{{100.0 * realised / ceiling:.0f}}}" if excess_lo > 0
            else f"\\newcommand{{\\Realised{tag}}}{{n/a}}",
        ]
    return out


def _fulltrack_macros() -> list[str]:
    """Qwen2.5-Omni re-run on the whole audio track (scripts/run_qwen_fulltrack.sh).

    Every Qwen2.5-Omni cell in the grid fed the model the first 30 s of each track. The rerun
    changes two things at once -- the audio is the whole track and the silence is as long as
    that item's track -- so a difference from the as-run cells is attributed to the input
    contract, not to "more audio" alone. Per domain: G_A, Delta_A(0), C - C0, and the
    item-paired change in G_A against the 30 s cells.
    """
    out: list[str] = []
    for domain, tag in (("music", "Music"), ("avut", "Avut")):
        ft = {k: EVAL_DIR / f"{p}_cmss_{domain}_qwenomni_fulltrack_{m}.json"
              for k, (p, m) in {"ar": ("poscontrol", "real"), "as": ("poscontrol", "silent"),
                                "vr": ("avqa", "real"), "vs": ("avqa", "silent")}.items()}
        if not all(f.exists() for f in ft.values()):
            continue
        g, cid = _paired_vector(ft["ar"], ft["as"])
        d, cid_d = _paired_vector(ft["vr"], ft["vs"])
        g30, cid_30 = _paired_vector(EVAL_DIR / f"poscontrol_cmss_{domain}_qwenomni_real.json",
                                     EVAL_DIR / f"poscontrol_cmss_{domain}_qwenomni_silent.json")
        if not (np.array_equal(cid, cid_d) and np.array_equal(cid, cid_30)):
            raise ValueError(f"{domain}: full-track cells are not item-aligned with the grid")
        a, _ = correctness(ft["ar"])
        f, _ = correctness(ft["as"])
        v, _ = correctness(ft["vs"])
        excess = (((a == 1) & (v == 0)).astype(float) - ((f == 1) & (v == 0)).astype(float)) * 100.0
        for name, values in (("Deliv", g), ("DeltaA", d), ("Excess", excess), ("DelivChange", g - g30)):
            m, lo, hi = cluster_bootstrap_mean(values, cid, n_boot=N_BOOT, seed=SEED)
            out += [f"\\newcommand{{\\FT{name}{tag}}}{{{m:+.2f}}}",
                    f"\\newcommand{{\\FT{name}CI{tag}}}{{[{lo:+.2f}, {hi:+.2f}]}}"]
        samples = np.array([int(p.get("audio_samples", 0)) for p in json.loads(ft["ar"].read_text())["predictions"]])
        if samples.min() <= 0:
            raise ValueError(f"{domain}: full-track cell does not record the audio it was fed")
        coverage = np.minimum(samples, 30 * 16000) / samples  # share of the track the 30 s grid saw
        out += [f"\\newcommand{{\\FTSeconds{tag}}}{{{float(np.median(samples)) / 16000:.0f}}}",
                f"\\newcommand{{\\FTCoverageThirty{tag}}}{{{100.0 * float(np.median(coverage)):.0f}}}"]
    return out


def _envelope_macros() -> list[str]:
    """How often Delta_A(v) exceeds G_A, cell by cell and condition by condition.

    Delta_A(v) <= G_A is an empirical envelope on this grid, not a theorem: synergy can break
    it. Every one of the 13 visual conditions x 6 cells is checked as an item-paired contrast
    (Delta_A(v) - G_A per item, video-cluster bootstrap), and the macros count the point
    inversions and how many of them exclude zero.
    """
    conditions = [""] + [f"_vis-{f}-s{sev}" for f in FAMILIES for sev in ("0.33", "0.66", "1")]
    total = inverted = significant = 0
    for _domain, _kind, _model, _frontend, tag in CELLS:
        deliv, cid = _paired_vector(EVAL_DIR / f"poscontrol_{tag}_real.json",
                                    EVAL_DIR / f"poscontrol_{tag}_silent.json")
        for cond in conditions:
            real = EVAL_DIR / f"avqa_{tag}_real{cond}.json"
            silent = EVAL_DIR / f"avqa_{tag}_silent{cond}.json"
            if not (real.exists() and silent.exists()):
                raise ValueError(f"envelope check needs {real.name} and {silent.name}")
            d, cid_v = _paired_vector(real, silent)
            if not np.array_equal(cid, cid_v):
                raise ValueError(f"{tag}{cond}: not item-aligned with its delivery cells")
            mean, lo, _hi = cluster_bootstrap_mean(d - deliv, cid, n_boot=N_BOOT, seed=SEED)
            total += 1
            inverted += int(mean > 0)
            significant += int(lo > 0)
    # Qwen2.5-Omni minus VideoLLaMA2.1-AV delivery on the same music items: the two cells the
    # registered 20 pp threshold splits.
    q, cid_q = _paired_vector(EVAL_DIR / "poscontrol_cmss_music_qwenomni_real.json",
                              EVAL_DIR / "poscontrol_cmss_music_qwenomni_silent.json")
    v, cid_b = _paired_vector(EVAL_DIR / "poscontrol_cmss_music_vl2_real.json",
                              EVAL_DIR / "poscontrol_cmss_music_vl2_silent.json")
    if not np.array_equal(cid_q, cid_b):
        raise ValueError("music delivery cells are not item-aligned across systems")
    m, lo, hi = cluster_bootstrap_mean(q - v, cid_q, n_boot=N_BOOT, seed=SEED)
    # Qwen2.5-Omni minus video-SALMONN 2+ delivery on the same music items: the headline
    # contrast between the two Whisper-family systems, as one paired statistic.
    w, cid_w = _paired_vector(EVAL_DIR / "poscontrol_cmss_music_vsalm2_real.json",
                              EVAL_DIR / "poscontrol_cmss_music_vsalm2_silent.json")
    if not np.array_equal(cid_q, cid_w):
        raise ValueError("music delivery cells are not item-aligned across Whisper-family systems")
    fm, flo, fhi = cluster_bootstrap_mean(q - w, cid_q, n_boot=N_BOOT, seed=SEED)
    return [
        f"\\newcommand{{\\FamilyDiff}}{{{fm:+.2f}}}",
        f"\\newcommand{{\\FamilyDiffCI}}{{[{flo:+.2f}, {fhi:+.2f}]}}",
        f"\\newcommand{{\\EnvelopeChecks}}{{{total}}}",
        f"\\newcommand{{\\EnvelopeInversions}}{{{inverted}}}",
        f"\\newcommand{{\\EnvelopeInversionsSig}}{{{significant}}}",
        f"\\newcommand{{\\GateSplitDiff}}{{{m:+.2f}}}",
        f"\\newcommand{{\\GateSplitDiffCI}}{{[{lo:+.2f}, {hi:+.2f}]}}",
    ]


def delta_curve(tag: str) -> list[tuple[float, float, float]]:
    """Delta_A at each visual severity, pooled over the four corruption families."""
    out: list[tuple[float, float, float]] = []
    for sev in ("clean", "0.33", "0.66", "1"):
        suffixes = [""] if sev == "clean" else [f"_vis-{f}-s{sev}" for f in FAMILIES]
        diffs: list[np.ndarray] = []
        videos: list[str] = []
        for suffix in suffixes:
            real, vids = correctness(EVAL_DIR / f"avqa_{tag}_real{suffix}.json")
            silent, _ = correctness(EVAL_DIR / f"avqa_{tag}_silent{suffix}.json")
            diffs.append((real - silent) * 100.0)
            videos.extend(vids)
        index = {v: i for i, v in enumerate(sorted(set(videos)))}
        cid = np.array([index[v] for v in videos])
        out.append(cluster_bootstrap_mean(np.concatenate(diffs), cid, n_boot=N_BOOT, seed=SEED))
    return out


def figure_surfaces(rows: list[dict[str, object]]) -> None:
    """Delta_A(v) per cell, with the delivery gain drawn as a reference line (an envelope here, not a bound).

    Visual degradation cannot reveal more audio reliance than audio alone supplies, so the
    delivery gain is a horizontal bound on each curve. Drawing it turns the panels into a
    single statement: a curve rises only in the gap between Delta_A(0) and that bound, and
    where the bound is at zero there is nothing to rise into.
    """
    x = [0.0, 0.33, 0.66, 1.0]
    order = [(d, m) for d in ("MUSIC-AVQA", "AVUT")
             for m in ("VideoLLaMA2.1-AV", "video-SALMONN 2+", "Qwen2.5-Omni")]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 4.0), sharex=True, sharey=True)
    for ax, (domain, model) in zip(axes.ravel(), order, strict=True):
        row = next((r for r in rows if r["domain"] == domain and r["model"] == model), None)
        if row is None:
            ax.set_visible(False)
            continue
        colour, marker = style_for(model, str(row["frontend"]))
        curve = delta_curve(str(row["tag"]))
        mean = [c[0] for c in curve]
        lo = [c[1] for c in curve]
        hi = [c[2] for c in curve]
        gain = float(row["deliv"])
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=1)
        ax.axhline(gain, color="0.25", lw=1.0, ls="-.", zorder=2)
        ax.text(0.02, gain + 0.8, f"$G_A$ = {gain:+.2f} (reference)", fontsize=5.8, color="0.25")
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, mean, color=colour, marker=marker, ms=4, lw=1.7, zorder=3)
        ax.set_title(f"{model} / {domain}   SG = {mean[-1] - mean[0]:+.2f} pp", fontsize=7.5)
        ax.set_xticks(x)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("visual degradation severity $v$", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta_A(v)$  (pp)", fontsize=7.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "figures" / "fig3_surfaces.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Compute the cells, write the macros, render both figures."""
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    rows = collect()
    if len(rows) < 2:
        raise SystemExit("not enough cells on disk to draw the crossover")

    macros: list[str] = []
    # Delta_A(v) endpoints and SG, so the surfaces section cites macros like everything else.
    # Fusion loss and headroom are paired item-level quantities, so both are bootstrapped
    # directly rather than differenced from two separate intervals -- a review noted they
    # were previously reported as bare point estimates.
    for row in rows:
        curve = delta_curve(str(row["tag"]))
        row["dA_severe"] = curve[-1][0]
        row["sg"] = curve[-1][0] - curve[0][0]
        tag = str(row["tag"])
        gain_d, gain_c = _paired_vector(EVAL_DIR / f"poscontrol_{tag}_real.json",
                                        EVAL_DIR / f"poscontrol_{tag}_silent.json")
        clean_d, _ = _paired_vector(EVAL_DIR / f"avqa_{tag}_real.json",
                                    EVAL_DIR / f"avqa_{tag}_silent.json")
        loss_m, loss_lo, loss_hi = cluster_bootstrap_mean(gain_d - clean_d, gain_c, n_boot=N_BOOT, seed=SEED)
        row["loss_ci"] = (loss_m, loss_lo, loss_hi)
        sev_d, sev_c = _pooled_severe(tag)
        head_m, head_lo, head_hi = cluster_bootstrap_mean(gain_d - sev_d, gain_c, n_boot=N_BOOT, seed=SEED)
        row["headroom_ci"] = (head_m, head_lo, head_hi)
    print(f"{'front end':10s} {'domain':12s} {'floor':>7s} {'audio':>7s} {'vision':>7s} "
          f"{'both':>7s} {'delivery':>20s} {'Delta_A(0)':>20s} {'loss':>7s}")
    for row in rows:
        tag = macro_suffix(str(row["domain"]), str(row["model"]))
        loss = float(row["deliv"]) - float(row["dA"])
        macros += [
            f"\\newcommand{{\\Deliv{tag}}}{{{float(row['deliv']):+.2f}}}",
            f"\\newcommand{{\\DelivCI{tag}}}{{[{float(row['deliv_lo']):+.2f}, {float(row['deliv_hi']):+.2f}]}}",
            f"\\newcommand{{\\DeltaA{tag}}}{{{float(row['dA']):+.2f}}}",
            f"\\newcommand{{\\DeltaACI{tag}}}{{[{float(row['dA_lo']):+.2f}, {float(row['dA_hi']):+.2f}]}}",
            f"\\newcommand{{\\FusionLoss{tag}}}{{{loss:.2f}}}",
            f"\\newcommand{{\\FusionLossCI{tag}}}{{[{row['loss_ci'][1]:+.2f}, {row['loss_ci'][2]:+.2f}]}}",
            f"\\newcommand{{\\Headroom{tag}}}{{{row['headroom_ci'][0]:+.2f}}}",
            f"\\newcommand{{\\HeadroomCI{tag}}}{{[{row['headroom_ci'][1]:+.2f}, {row['headroom_ci'][2]:+.2f}]}}",
            f"\\newcommand{{\\SG{tag}}}{{{float(row['sg']):+.2f}}}",
            f"\\newcommand{{\\DeltaASevere{tag}}}{{{float(row['dA_severe']):+.2f}}}",
            f"\\newcommand{{\\AudioOnly{tag}}}{{{float(row['audio_only']):.2f}}}",
            f"\\newcommand{{\\VisionOnly{tag}}}{{{float(row['vision_only']):.2f}}}",
            f"\\newcommand{{\\Both{tag}}}{{{float(row['both']):.2f}}}",
            f"\\newcommand{{\\Floor{tag}}}{{{float(row['floor']):.2f}}}",
        ]
        print(f"{row['frontend']:10s} {row['domain']:12s} {float(row['floor']):7.2f} "
              f"{float(row['audio_only']):7.2f} {float(row['vision_only']):7.2f} {float(row['both']):7.2f} "
              f"{float(row['deliv']):+7.2f} [{float(row['deliv_lo']):+6.2f},{float(row['deliv_hi']):+6.2f}] "
              f"{float(row['dA']):+7.2f} [{float(row['dA_lo']):+6.2f},{float(row['dA_hi']):+6.2f}] {loss:7.2f}")

    # The crossover is an interaction claim, so the interaction contrast is computed here
    # rather than quoted. It was briefly hand-entered in the manuscript and the estimate and
    # its interval came from different coverage conditions; deriving both from the same
    # bootstrap makes that class of mismatch impossible.
    macros += _interaction_macros()
    macros += _probe_macros()
    macros += _matched_coverage_macros()
    macros += _shuffled_macros()
    macros += _census_macros()
    macros += _degeneration_macros()
    macros += _access_macros()
    macros += _ceiling_macros()
    macros += _contrast_macros()
    macros += _envelope_macros()
    macros += _fulltrack_macros()

    (OUT_DIR / "numbers_crossover.tex").write_text("\n".join(sorted(macros)) + "\n")
    figure_crossover(rows)
    figure_fusion(rows)
    figure_surfaces(rows)
    print(f"\nwrote {OUT_DIR / 'numbers_crossover.tex'} ({len(macros)} macros)")
    print(f"wrote {OUT_DIR / 'figures' / 'fig1_crossover.pdf'}")
    print(f"wrote {OUT_DIR / 'figures' / 'fig2_fusion.pdf'}")
    print(f"wrote {OUT_DIR / 'figures' / 'fig3_surfaces.pdf'}")


if __name__ == "__main__":
    main()
