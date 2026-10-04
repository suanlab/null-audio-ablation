# What a Null Audio Ablation Measures

Code, preregistration, and per-cell predictions for *"What a Null Audio Ablation Measures:
Separating Delivery, Front-End Capability, and Fusion"* (submitted to IEEE/ACM Transactions
on Audio, Speech, and Language Processing).

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

A third audio-visual LLM, added after the first draft, sharpens the point. It is fronted by
the same encoder family as one of the other two and behaves differently: the two systems
fall on opposite sides of the delivery gate on music audio while being indistinguishable on
speech at every one of thirteen visual conditions. So naming the audio front end does not
tell you whether audio reached the model --- the delivery control has to be run per system.
The same model delivers the most music audio of anything tested here and realises none of
the resulting ceiling with vision intact, which is what makes delivery and fusion separable
rather than two names for one thing.

## What is here

| Path | Contents |
|---|---|
| `src/videollm/` | Estimands, video-cluster bootstrap, visual degradation, answer matching |
| `scripts/` | Grid runners, aggregation, figure and macro generation, encoder probes |
| `eval_results/` | Per-item predictions for every evaluation cell the paper cites |
| `splits/` | Pilot split item ids and manifests with SHA-256 provenance |
| `preregistration.md` | Frozen preregistration with dated, append-only amendments |
| `tests/` | Regression tests, several of which encode harness bugs found during this work |

Four defects in the third model's adapter are worth naming, because **every one produced
well-formed output**: a missing audio block in the conversation, so the chat template
emitted no audio placeholder and the waveform was never referenced by any token position;
a superseded benchmark subset; a video token budget sized against a context four times the
checkpoint's, which overflowed silently because the tokenizer warns once per process; and a
pixel-range assumption that meant the visual-degradation axis had never run. The first is
the instructive one: byte-identical output across audio conditions is exactly what a model
that ignores audio produces under greedy decoding, and only the delivery control told the
two apart. `tests/test_qwen_audio_context_contract.py` pins the first and third so they
cannot return silently; `eval_results/QWENOMNI_STALE.md` marks the cells they produced.

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
python scripts/delivery_gain.py                 # delivery gate G_A for every model
python scripts/pooled_sg.py --model cmss_music_vsalm2
```

The grid runners require the media and the model checkpoints, and are provided so the
procedure is inspectable rather than because they can be run from this repository alone.

## Status of the claims

The protocol, the estimands and the 20 pp delivery gate were preregistered before the runs.
The crossover, the fusion-loss quantity, the bound, and the probes are **exploratory** —
formulated after the confirmatory grid, prompted by adversarial internal review. Our
preregistered primary claim was **not met**, and the paper says so. The third front end was
added by a dated amendment written **before any of its results were observed**, so its
configuration cannot be read as chosen to produce them; it faced the same preregistered
delivery gate as the other two, and everything downstream of that gate is exploratory like
the rest of the cross-system comparison. `preregistration.md` records each outcome with its
date; two of its amendments are explicitly retrospective and say so.

## Citation

Citation details will be added if the paper is accepted.
