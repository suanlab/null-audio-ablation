# What a Null Audio Ablation Measures

Code, preregistration, and per-cell predictions for *"What a Null Audio Ablation Measures:
Separating Delivery, Front-End Capability, and Fusion"* (submitted to IEEE Open Journal of
Signal Processing, ICASSP 2027 track).

## What the paper argues

The standard test of whether an audio-visual LLM uses sound is to replace the audio with
silence and report the accuracy drop, `Δ_A`. A near-zero `Δ_A` is read as the model
ignoring what it hears. At least three different situations produce that number, and the
measurement cannot tell them apart:

1. **Not delivered** — the audio never reached the model in a usable form.
2. **Delivered but not fused** — audio alone carries the answer, and it adds nothing once
   vision is present.
3. **Redundant benchmark** — either channel alone nearly solves the task.

Adding a blank-channel *delivery control* separates the first from the others; pairing it
with `Δ_A` quantifies the second.

## What is here

| Path | Contents |
|---|---|
| `src/videollm/` | Estimands, video-cluster bootstrap, visual degradation, answer matching |
| `scripts/` | Grid runners, aggregation, figure and macro generation, encoder probes |
| `eval_results/` | Per-item predictions for every evaluation cell the paper cites |
| `splits/` | Pilot split item ids and manifests with SHA-256 provenance |
| `preregistration.md` | Frozen preregistration with dated, append-only amendments |
| `tests/` | Regression tests, several of which encode harness bugs found during this work |

`scripts/make_crossover_figures.py` regenerates every figure and every tabulated value in
the manuscript from the files in `eval_results/`, emitting the macro file the paper cites
rather than digits typed by hand. A handful of in-text quantities (answer-diversity
percentages, audio-coverage fractions) are computed outside that script and are not
covered by it. The numbers were produced under `requirements-frozen.txt`; the looser bounds
in `pyproject.toml` are for installation, not reproduction.

The manuscript itself is not included here while it is under review.

## What is deliberately **not** here

**No video or audio media, from any dataset.** All three corpora used or referenced are
YouTube-sourced and redistributing their media is not ours to authorise:

- **MUSIC-AVQA** (Li et al., CVPR 2022) — obtain from the authors' official release.
- **AVUT** (EMNLP 2025) — **licence unresolved**: the dataset card carries no licence field
  and the repository no LICENSE file, as of 2026-08-14. We use it for internal research
  evaluation only and redistribute nothing.
- **VGGSound-derived AVQA** — an anchor we used earlier and withdrew after finding it
  misattributed; it appears here only as an inertness control.

No model weights. All models are third-party checkpoints obtained from their own releases.

`splits/` contains item identifiers and question ids, so the exact evaluation subsets are
reconstructible once you have obtained the media yourself.

## Reproducing

```bash
pip install -e ".[dev]"
pytest tests/                                   # regression suite
python scripts/make_crossover_figures.py        # figures + every quoted number
python scripts/pooled_sg.py --model cmss_music_vsalm2
```

The grid runners require the media and the model checkpoints, and are provided so the
procedure is inspectable rather than because they can be run from this repository alone.

## Status of the claims

The protocol, the estimands and the 20 pp delivery gate were preregistered before the runs.
The crossover, the fusion-loss quantity, the bound, and the probes are **exploratory** —
formulated after the confirmatory grid, prompted by adversarial internal review. Our
preregistered primary claim was **not met**, and the paper says so. `preregistration.md`
records each outcome with its date; its final amendment is explicitly retrospective.

## Citation

Citation details will be added if the paper is accepted.
