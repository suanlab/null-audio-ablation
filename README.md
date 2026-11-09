# Interpreting Null Audio Ablations

Code, preregistration, and per-cell predictions for *"Interpreting Null Audio Ablations:
System-Level Controls and Their Limits"* (submitted to IEEE/ACM Transactions on Audio, Speech,
and Language Processing).

## What the paper argues

The standard test of whether an audio-visual LLM uses sound is to replace the audio with
silence and report the accuracy drop, `Δ_A`. A near-zero `Δ_A` is read as the model
ignoring what it hears. At least three different situations produce that number:

1. **Not delivered** — the audio never reached the model in a usable form.
2. **Delivered but not fused** — audio alone changes the answers, and it adds nothing once
   vision is present.
3. **Redundant benchmark** — either channel alone nearly solves the task.

The paper studies what two controls add: a blank-video *delivery control*, and an item-level
split of the clean-input gain (answers audio alone supplies and vision alone misses, checked
against the same count under silence). They **constrain** which reading applies; they do not
identify the cause, and some cells stay undetermined.

Two systems on the same Whisper encoder family gain −0.33 and +22.33 pp from audio delivery on
the same music items, so the encoder family does not predict whether audio reaches the model.
On speech no difference between them is detected, and the paper does not claim equivalence.
The system that delivers the most music audio shows no detectable audio-unique signal on
music, yet recovers 66 % of its audio-unique speech items.

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
covered by it. The numbers were produced under `requirements-frozen.txt`, an exact
`pip list --format=freeze` of the analysis environment; the looser bounds in `pyproject.toml`
are for installation, not reproduction.

Checks added before submission, all exploratory:

- **Whole-track rerun of the third system.** The Qwen2.5-Omni adapter fed the model the first 30 s
  of each track; an earlier draft said it received the whole track. The four estimand conditions were
  rerun with `QWEN_AUDIO_SECONDS=full` and silence matched to each item's length
  (`scripts/run_qwen_fulltrack.sh`, `eval_results/*_qwenomni_fulltrack_*.json`). The delivery gain is
  unchanged: +0.00 pp [−3.03, +3.03] on music and +1.00 pp [−2.61, +4.45] on speech, and no reading
  changes.
- **Shuffled audio for the third system** (`eval_results/shuf_poscontrol_cmss_*_qwenomni.json`).
  These cells split its music delivery gain into de-degeneration (+2.00 pp) and content
  (+20.33 pp).
- **Encoder weights against Whisper-large-v3** (`scripts/check_encoder_weights.py`).
  video-SALMONN 2+'s encoder is bit-identical in bf16. Qwen2.5-Omni's tower was trained from
  that initialisation (relative difference 0.65).
- **Instrument probe on matched input** (`scripts/run_encoder_probe_matched.py`). All three encoders
  get the same first 30 s, with paired intervals over clips: BEATs 54.6 %, Qwen2.5-Omni tower
  49.4 %, Whisper-large-v3 35.7 %. The ordering holds on the 682 clips whose label comes from an
  audio-visual question.
- **Item-level split** of the clean-input gain (recovered, synergy, lost), with the selection
  headroom checked against its blank-channel floor, computed in `make_crossover_figures.py`.
- **The registered Holm family** (`scripts/primary_family.py`). It holds four MUSIC-AVQA tests, per
  the 2026-08-16 amendment, and an eight-test version with AVUT is a labelled sensitivity analysis.

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
pip install -e ".[dev,analysis]"                # or: pip install -r requirements-frozen.txt
pytest tests/                                   # regression suite
python scripts/make_crossover_figures.py        # figures + every quoted number
python scripts/primary_family.py                # the registered 8-test Holm family
python scripts/delivery_gain.py                 # delivery gate G_A for every model
python scripts/pooled_sg.py --model cmss_music_vsalm2
```

These need no media, checkpoints or path variables. Running a model adapter does: see
`PATHS.md` for the variables, `MODELS.md` for checkpoint and upstream-code revisions, and
`envs/` for each adapter's exact environment.

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
