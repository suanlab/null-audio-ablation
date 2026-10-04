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
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--", zorder=1)
    ax.axhline(GATE_PP, color="#009E73", lw=1.1, ls=":", zorder=1)
    ax.text(-0.30, GATE_PP + 1.0, "20 pp gate", fontsize=6, color="#009E73", ha="left")

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
        for x, y in zip(xs, ys, strict=True):
            # Labels sit outward from the crossing so neither the gate line nor the
            # opposite series' label is overwritten.
            dx, dy = (-34, -2) if (x == 0 and y > 5) else (0, 9 if y > 5 else -14)
            ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=6.5, color=colour)

    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels([f"{d}\n({'music' if d == 'MUSIC-AVQA' else 'speech'} audio)" for d in domains], fontsize=7)
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylabel("audio delivery gain (pp)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-7.0, 27.0)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=2, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "figures" / "fig1_crossover.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_fusion(rows: list[dict[str, object]]) -> None:
    """Delivery gain against Delta_A(0); distance below the diagonal is fusion loss."""
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    lim = (-8.0, 26.0)
    ax.plot(lim, lim, color="0.55", lw=0.9, ls="--", zorder=1)
    ax.annotate("all delivered audio used", (13.0, 13.0), rotation=34, fontsize=6,
                color="0.4", ha="center", va="bottom")
    ax.axhline(0.0, color="0.8", lw=0.7, zorder=0)
    ax.axvline(0.0, color="0.8", lw=0.7, zorder=0)

    for row in rows:
        colour, marker = style_for(str(row["model"]), str(row["frontend"]))
        x, y = float(row["deliv"]), float(row["dA"])
        if x - y > 5.0:  # audio arrived and was discarded: show the drop
            ax.plot([x, x], [x, y], color=colour, lw=1.0, ls=":", zorder=2)
            ax.annotate(f"fusion loss\n{x - y:.2f} pp", (x, (x + y) / 2), textcoords="offset points",
                        xytext=(8, 0), ha="left", va="center", fontsize=6.5, color=colour)
        ax.errorbar(x, y, xerr=[[x - float(row["deliv_lo"])], [float(row["deliv_hi"]) - x]],
                    yerr=[[y - float(row["dA_lo"])], [float(row["dA_hi"]) - y]],
                    fmt=marker, color=colour, ms=6, capsize=2, elinewidth=0.8, zorder=3)
        # The two non-delivering cells land on the same spot near the origin, so each
        # label is placed by hand rather than by a shared offset.
        # Whisper/MUSIC-AVQA and BEATs/AVUT are the two non-delivering cells and land on
        # the same spot, so their labels go in opposite directions; the rest are placed
        # directly under their own marker rather than offset sideways, which previously
        # left a label sitting closer to a different point than to its own.
        offsets = {
            ("Whisper", "AVUT"): (10, -4, "left"),
            ("BEATs", "MUSIC-AVQA"): (0, -15, "center"),
            ("Whisper", "MUSIC-AVQA"): (0, 15, "center"),
            ("BEATs", "AVUT"): (0, -17, "center"),
        }
        dx, dy, ha = offsets.get((str(row["frontend"]), str(row["domain"])), (7, -2, "left"))
        ax.annotate(f"{row['frontend']} / {row['domain']}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6, color=colour, va="center", ha=ha)

    ax.set_xlim(-11.0, 30.0)
    ax.set_ylim(-11.0, 21.0)
    ax.set_xlabel("audio delivery gain (pp)\nwhat audio alone can contribute", fontsize=7.5)
    ax.set_ylabel(r"$\Delta_A(0)$ (pp)" "\nwhat it adds on top of vision", fontsize=7.5)
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
    rng = np.random.default_rng(SEED)

    def draw(a: Path, b: Path) -> tuple[np.ndarray, float]:
        diff, cid = _paired_vector(a, b)
        n_clusters = int(cid.max()) + 1
        sums = np.zeros(n_clusters)
        counts = np.zeros(n_clusters)
        np.add.at(sums, cid, diff)
        np.add.at(counts, cid, 1)
        pick = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
        return sums[pick].sum(1) / counts[pick].sum(1), float(diff.mean())

    wa, wa_m = draw(EVAL_DIR / "poscontrol_cmss_avut_vsalm2_real.json",
                    EVAL_DIR / "poscontrol_cmss_avut_vsalm2_silent.json")
    wm, wm_m = draw(EVAL_DIR / "poscontrol_cmss_music_vsalm2_real.json",
                    EVAL_DIR / "poscontrol_cmss_music_vsalm2_silent.json")
    ba, ba_m = draw(EVAL_DIR / "full_poscontrol_cmss_avut_vl2_real.json",
                    EVAL_DIR / "full_poscontrol_cmss_avut_vl2_silent.json")
    bm, bm_m = draw(EVAL_DIR / "full_poscontrol_cmss_music_vl2_real.json",
                    EVAL_DIR / "full_poscontrol_cmss_music_vl2_silent.json")
    draws = (wa - wm) - (ba - bm)
    point = (wa_m - wm_m) - (ba_m - bm_m)
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
    """The item-level ceiling on what audio can add, and how much of it is realised.

    ``G_A - Delta_A(0)`` is a difference of two *marginal* accuracies and is silent on
    whether the items audio answers alone are the items vision already answers. Where they
    coincide, a near-zero ``Delta_A(0)`` is benchmark redundancy, not discarded information.
    The quantity that separates the two is the share of items audio alone gets right and
    vision alone gets wrong: no fusion, however good, can add more than that. On the anchor
    cell it is 6.00 pp against a marginal fusion loss of 18.33 pp, so the marginal figure
    attributes to fusion roughly three times what fusion could ever have delivered.
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
        out += [
            f"\\newcommand{{\\Ceiling{tag}}}{{{ceiling:.2f}}}",
            f"\\newcommand{{\\CeilingCI{tag}}}{{[{lo:.2f}, {hi:.2f}]}}",
            f"\\newcommand{{\\CeilingN{tag}}}{{{int(unique.sum())}}}",
            f"\\newcommand{{\\Realised{tag}}}{{{100.0 * realised / ceiling:.0f}}}" if ceiling > 0.5
            else f"\\newcommand{{\\Realised{tag}}}{{n/a}}",
        ]
    return out


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
    """Delta_A(v) per cell, with the delivery gain drawn as the ceiling it imposes.

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
        ax.text(0.02, gain + 0.8, f"delivery ceiling {gain:+.2f}", fontsize=5.8, color="0.25")
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
