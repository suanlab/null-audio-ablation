# Preregistration v0 — Causal Modality Substitution Surface (CMSS)

> **Status: FROZEN-ON-COMMIT.** This document fixes estimands, thresholds, the
> primary test family, and kill criteria **before any pilot data is seen**, per
> TOP_PLAN.md §2.3 / §7.1 (W1) and the immediate-action list. Numbers here are
> *decision rules*, not results. Any change after the first pilot run must be
> recorded in a dated amendment block at the bottom, never by editing the frozen
> body — the whole point of preregistration is that thresholds cannot be moved to
> fit observed data.
>
> Authored: 2026-07-12 (Asia/Seoul). Aligns with `TOP_PLAN.md` (2026-07-11).
> Confirmatory scope = C1, C2, C4. Exploratory scope = C3, C5 (labelled as such in
> the paper; no causal language).

---

## 1. Research question & core claim (locked wording)

**Does measuring the substitution response — how an AV-LLM reallocates to one
modality as the *other* modality's reliability degrades continuously — distinguish
genuine progressive fusion from last-moment fallback, in a way that fixed clean-input
accuracy cannot?**

Core claim under test (C1): clean-input accuracy hides conditional modality reliance;
CMSS identifies late-fallback or non-continuous substitution regimes across multiple
AV-LLMs. This is a **behavioral causal-effect** claim about randomized input
interventions, **not** a claim about internal representational mechanisms. Language
like "the model thinks/decides" is prohibited.

---

## 2. Estimands (fixed definitions)

For item `i`, audio state `a`, visual reliability level `v`, let `Y_i(a, v) ∈ {0,1}`
be answer correctness.

- **Conditional audio contribution** at visual level `v`:
  `Δ_A(v) = E_i[ Y_i(real_audio, v) − Y_i(null_audio, v) ]`
  where `null_audio` is the **primary** audio-removal operator = `silent` (zeros).
  `shuffled` is a **secondary** audio-removal operator reported alongside; `silent`
  is primary because it is deterministic and does not inject cross-item content.
- **Substitution gain**: `SG = Δ_A(v_low) − Δ_A(v_clean)`, with `v_low` = the most
  degraded **non-blank** visual level (`severe`, severity 1.0). The fully-blank video
  endpoint is **not** used to compute SG (it is the delivery positive control, §4).
- **Switch threshold**: `τ_switch` = the smallest degradation level at which
  `Δ_A(v)` exceeds `Δ_A(v_clean)` by ≥ **10 pp** (primary margin). Sensitivity also
  reported at **5 pp** and **15 pp**. The 10 pp margin is fixed here, pre-pilot.
- **Fusion area**: area where full-modal log-odds exceed the better unimodal
  (audio-only, video-only) baseline, computed on item-level log-odds and on the hard
  subset to avoid ceiling artifacts.

---

## 3. Intervention grid (fixed)

### 3.1 Visual reliability axis
Implemented by `src/videollm/data/video_degradation.py` (deterministic, seed-fixed).

- **Severity levels (ordinal):** `clean = 0.0`, `mild = 0.33`, `medium = 0.66`,
  `severe = 1.0`. Plus `blank` = full frame removal, handled as the **positive
  control**, not as a degradation severity.
- **Corruption families (all four, pre-registered as equal):** `motion_blur`,
  `occlusion`, `frame_drop`, `downscale`. Primary curve = **per-family**, then
  aggregated only after per-family direction agreement is checked (C4).
- **Natural-corruption subset:** ≥1 subset of naturally low-visual-SNR items, held
  out to confirm the synthetic-artifact conclusion is not an artifact (C4).

### 3.2 Audio axis
`real` (baseline), `silent` (primary null), `shuffled` (secondary null), plus
`noise`, `shifted` retained for continuity with the existing 5-mode protocol.

### 3.3 Positive control (delivery validation)
Blank-video-with-audio and blank-audio-with-video per model per domain. A model that
does **not** pass both delivery controls is excluded from the headline and marked
`unvalidated` (never silently dropped).

---

## 4. Primary confirmatory test family (fixed; Holm-Bonferroni)

The **primary family** for family-wise correction is exactly:

1. `SG > 0` test per (model × domain), audio null = `silent`, family = pooled visual
   (after per-family agreement gate). → the C1 tests.
2. Delivery positive-control `real vs blank-video-silent` per (model × domain). → C2/
   delivery.
3. `τ_switch` existence per (model × domain).

Everything else (per-family SG, per-severity Δ_A, shuffled variants, noise/shifted,
Shapley/PID attributions, low-level-competence regressions, mitigation) is
**secondary/exploratory** and reported without inclusion in the primary Holm family.

Multiplicity: Holm-Bonferroni at **α = 0.05** over the primary family. The existing
`scripts/multiple_comparison_correction.py` (Holm + BH) is the reference
implementation and will be extended to the CMSS family.

---

## 5. Success thresholds (pinned, from TOP_PLAN §2.2)

| Claim | Pass criterion (pre-fixed) |
|---|---|
| **C1** (main) | `SG ≥ 10 pp` (Holm-corrected `p<0.05`) in **≥3/4** model families and **≥2/3** domains; Spearman between clean-accuracy rank and SG rank `|ρ| < 0.5`; changepoint (cliff) model beats linear by **ΔAIC ≥ 10** or significant held-out-likelihood gain. |
| **C2** (support) | ≥2 model pairs with similar endpoint R−S but `τ_switch`/curve-AUC differing, bootstrap 95% CI excluding 0. |
| **C3** (exploratory) | leave-one-model-out `R² ≥ 0.40` or Spearman `ρ ≥ 0.6`, 95% bootstrap CI excludes 0. Else demoted to exploratory correlation. |
| **C4** (support) | SG direction agrees in **≥3/4** corruption families; **≥90%** of 200 human-audited pairs preserve answer semantics; natural-subset effect direction agrees. |
| **C5** (exploratory) | corrupted-region accuracy **+≥5 pp**, fusion area **+≥15%** relative, clean-accuracy drop **≤1 pp**, both model families paired `p<0.05`. |

**Single point of failure:** C1 cross-domain replication (§2.3). If SG is large only
on MUSIC-AVQA, or the curve is flat once the blank endpoint is excluded, the top-tier
main claim collapses and the work descends to Plan B (evaluation/benchmark paper).

---

## 6. Statistical protocol (fixed, from TOP_PLAN §3.5)

- **Resampling unit = video cluster** (not QA row), bootstrap **10,000** resamples,
  **seed = 2027**, to handle within-video question correlation.
- Item-paired interventions: **exact McNemar** or **paired permutation**.
- **Equivalence** ("no difference") uses **TOST**, margin **±3 pp** accuracy;
  curve-AUC margin fixed from pilot-independent practical bound. Non-significance is
  never reported as equivalence.
- **Seeds:** stochastic corruption ≥ **3** seeds; headline severities/models **5**
  seeds; deterministic zero/shift use item bootstrap instead of seed repeats.
  Mitigation training: **3** independent seeds, full ±95% CI + per-seed disclosure.
- **Curve-shape comparison:** linear vs monotone spline vs changepoint, compared by
  held-out video likelihood / AIC. Cliffs are never declared by eye.
- **Data hygiene:** pilot/threshold-selection set and final test set are
  **video-disjoint**.

---

## 7. Kill / pivot gates (fixed, from TOP_PLAN §4.2)

| Gate | Kill/pivot condition | Action |
|---|---|---|
| W2 harness | model fails A/V positive control OR deterministic replay < 99.5% | kill that adapter/model (≤2 days diagnosis) |
| W3 MUSIC pilot | both validated externals `SG < 5 pp` in graded region AND curve adds nothing over binary endpoint | pivot universal-late-fallback claim → ICASSP diagnostic extension |
| W6 cross-domain | 2nd domain effect direction opposite OR 95% CI within ±5 pp equivalence | kill "general law"; reassess as domain-taxonomy paper |
| W9 artifact validity | human semantic preservation < 90% OR ≥2 corruption families disagree in direction | kill causal-reliability wording → "intervention sensitivity" |
| W12 breadth | validated model families < 4 OR domains < 3 | pivot ICML target → NeurIPS 2027 or ICASSP/ACM MM |
| W16 predictor | regime predictor `R² < 0.2` and no architecture-taxonomy explanation | kill C3 from main paper |
| W19 mitigation | both models corrupted-gain < 3 pp OR clean-drop > 2 pp | kill C5 → negative-result appendix |

---

## 8. Models & domains (headline scope)

- **Models (headline requires passing both delivery controls per domain):**
  VideoLLaMA2.1-7B-AV, video-SALMONN 2+, one of {Ola-7B, VITA} [checkpoint TBD],
  Qwen2.5-Omni-7B (headline only after its audio positive control is fixed &
  re-validated). AGTA/No-Bridge/Vision-Only = calibration only, separated from the
  main model table, no general-performance claims.
- **Domains (≥3 for C1):** MUSIC-AVQA (+Hard) [anchor], AVUT [audio-centric], one of
  {AV-Odyssey/DeafTest, AV-speech LRS3-based QA} [3rd domain].

---

## 9. What is explicitly NOT claimed pre-result

- No headline number, no "models exhibit late fallback", no title/abstract claim is
  asserted until the preregistered tests run on the **final test split**.
- AGTA is not a contribution; it is a calibration probe.
- Synthetic corruption alone does not establish "causal reliability" — this needs the
  human-audit + natural-subset gate (C4) to survive.

---

## Amendments (append-only; dated)

### 2026-08-12 — interpretation rule for `SG` under a `Delta_A` ceiling
**No threshold is changed.** `SG >= 10 pp` (C1), the primary Holm family, the TOST
margin, and every kill criterion stand exactly as frozen above.

**Observation prompting this note (pilot, clean cells only):** the two models sit at
opposite ends of the protocol's range at `v = 0` — AGTA `Delta_A(clean) = +42.7 pp`
(real 75.7% vs silent 33.0%) and video-SALMONN 2+ `Delta_A(clean) = -1.3 pp`
(real 93.7% vs silent 95.0%).

**Problem.** `SG = Delta_A(v_severe) - Delta_A(v_clean)` measures the *increase* in audio
reliance. A model already near-maximally reliant at clean input has almost no headroom:
AGTA cannot gain much beyond +42.7 pp because silent accuracy is already near chance.
Its `SG` may therefore be small or negative for a reason that has nothing to do with
substitution. §5 anticipated a ceiling on *accuracy* (hard subset, log-odds); it did not
anticipate a ceiling on `Delta_A` itself.

**Rule (fixed now, before the degraded cells are read):**
1. `SG` is always reported **alongside `Delta_A(clean)`**; it is never interpreted alone.
2. A model with `Delta_A(clean) >= 25 pp` is flagged **headroom-limited**, and a failure
   to reach the `SG` threshold is reported as *uninformative for that model*, not as
   evidence against substitution.
3. The C1 pass criterion (>=3/4 model families, >=2/3 domains) is evaluated over
   **non-headroom-limited** models; headroom-limited models are reported separately and
   never used to argue either for or against C1.
4. This rule is symmetric and was fixed without seeing any degraded-cell result, so it
   cannot be tuned to favour an outcome.

The 25 pp flag threshold is set now and will not be revisited after the degraded cells
are seen.

### 2026-08-12 — damage-normalised `Delta_A` registered as EXPLORATORY
**Confirmatory analysis is unchanged.** `SG` and `tau_switch` remain defined on the raw
severity axis exactly as frozen in §2–§3. Nothing below is used to evaluate C1–C5.

**Observation (pilot, video-SALMONN 2+, two families).** Raw `SG` differs 2.6× between
corruption families — motion blur `+19.3 pp`, occlusion `+7.3 pp` — but the families
destroy very different amounts of visual information at the same nominal severity. Using
**silent accuracy as the vision-only reference**, the drop from its clean value measures
how much visual evidence a corruption actually removed:

| family | `SG` | vision-only loss at sev 1.0 | `SG` per pp of loss |
|---|---|---|---|
| motion_blur | +19.3 pp | 30.3 pp | 0.637 |
| occlusion | +7.3 pp | 13.0 pp | 0.564 |

Normalised this way the two families nearly coincide, and the per-severity ratios fall in
0.33–0.68. This suggests `Delta_A` tracks *how much vision was destroyed* rather than
*which corruption destroyed it* — relevant to C4, since a pure synthetic-artifact story
would predict the two structurally different corruptions to diverge.

It also exposes a confound in the confirmatory metric: occlusion's `tau_switch` is `none`
only because occlusion at severity 1.0 removes less visual evidence (13 pp) than motion
blur does at 0.66 (20 pp). Severity labels are **not equivalent across families**, so
`tau_switch` is not comparable between them on the raw axis.

**Status: exploratory.** This lens was constructed *after* seeing pilot results, so it is
reported separately, never substituted for the preregistered axis, and cannot be used to
rescue a claim that fails on the frozen definitions. If it is to become confirmatory, it
must be re-registered with thresholds fixed before the final test split is touched.

### 2026-08-13 — C1 OUTCOME RECORDED, and a new primary claim registered

**Part 1 — C1 result on the frozen criteria: NOT MET.**
Recorded before any replacement claim, so the swap cannot be read as moving goalposts.

Pilot, video-SALMONN 2+ (n=300, MUSIC-AVQA, the one delivery-validated external model):

| quantity (frozen definition) | value | criterion | outcome |
|---|---|---|---|
| pooled `SG` (§4 primary basis) | **+9.08 pp** [+6.75, +11.58] | ≥ 10 pp | **not met** (CI straddles it) |
| per-family `SG` | +19.3 / +7.3 / +2.3 / +7.3 | — | 3 of 4 below 10 pp |
| `tau_switch` | defined for motion_blur only | — | 1 of 4 families |
| model families | 1 | ≥ 3/4 | not met |
| domains | 1 | ≥ 2/3 | not met |

`SG` is positive with CI excluding zero in **4/4** families, so the W3 kill criterion
(`SG < 5 pp`) is *not* triggered — the effect is real but smaller and more
family-dependent than C1 required.

**Also refuted: the late-fallback hypothesis.** The `Delta_A` curve is smooth, and the
`SG`/vision-loss ratio is near-constant (0.665 ± 0.084) rather than rising with damage.
A genuine late fallback predicts a rising ratio. Note the ratio is
`1 − D_R/D_S` by construction (see the 2026-08-12 amendment), so it is a *description* of
the curve, not independent evidence about it.

**AGTA is withdrawn from all visual-axis analysis.** Its blank-video control equals its
clean condition (75.67% vs 75.67%, −1.00 pp on the silent arm), so the visual
intervention cannot be shown to reach the model. Per §3.3 it is `unvalidated` on the
visual axis. This supersedes the "headroom-limited" framing of the 2026-08-12
amendment, which described the symptom and misattributed the cause.

**Part 2 — new primary claim (M1), registered as EXPLORATORY pending confirmation.**

> **Clean-input modality ablation is not a valid test of modality grounding.** A model
> can measure as audio-independent at clean input while demonstrably exploiting audio
> once the other modality degrades.

Motivating observation (pilot; this claim was formed *after* seeing it, hence
exploratory): video-SALMONN 2+ has `Delta_A(clean) = −1.33 pp` (exact McNemar p = 0.45)
yet under severe motion blur exceeds its *better single channel* by **+17.33 pp**
(82.67% vs max(audio-only 65.33%, vision-only 64.67%)). The same model reads as
"ignores audio" and "fuses competently" depending only on where it is measured.

**Confirmatory design, fixed now, before the final test split is touched:**

1. **Eligibility.** A model enters only if it passes *both* delivery controls: audio
   (`blank-video real` − `blank-video silent` ≥ 20 pp) and visual
   (`clean silent` − `blank-video silent` ≥ 20 pp).
2. **Apparent independence.** `Delta_A(clean)` is TOST-equivalent to zero at ±3 pp,
   *or* its exact-McNemar p > 0.05.
3. **Utilisation gap.** `max_v [ both(v) − max(audio_only, vision_only(v)) ] ≥ 10 pp`,
   with a video-cluster bootstrap CI excluding zero.
4. **M1 is met** if ≥ 2 eligible models satisfy (2) and (3) simultaneously, in ≥ 2
   domains.
5. Holm–Bonferroni over the M1 family (one gap test per model × domain), α = 0.05.

Thresholds (20 pp delivery, ±3 pp equivalence, 10 pp gap, ≥2 models, ≥2 domains) are
fixed as of this entry and will not be revisited after the test split is run.

**Dropped:** C3 (predictor) — leave-one-model-out `R² ≥ 0.40` over ~4 models is not
estimable. C5 (mitigation) — deferred; M1 does not depend on it.

### 2026-08-14 — M1 OUTCOME: NOT MET (1 of 2 required models), and why

Second delivery-validated external model completed (VideoLLaMA2.1-AV, n=300,
MUSIC-AVQA, 28/28 cells). Both delivery controls pass (audio +67.3 pp, visual
+68.0 pp), so the model is eligible.

| M1 condition | video-SALMONN 2+ | VideoLLaMA2.1-AV |
|---|---|---|
| (1) eligibility — both delivery controls | PASS | PASS |
| (2) apparent independence at clean | PASS (`Delta_A` −1.33 pp, p = 0.45) | **FAIL** (+3.00 pp, p = 0.023) |
| (3) utilisation gap ≥ 10 pp, CI > 0 | PASS (+17.33 pp [+12.0, +22.7]) | **FAIL** (max +4.33 pp) |

**M1 is NOT met**: it requires ≥ 2 models satisfying (2) and (3) simultaneously in
≥ 2 domains; the evidence stands at 1 model, 1 domain. Thresholds are not being
relaxed to admit the second model.

**The reason is a property of the benchmark, and it is measurable.** Single-channel
accuracies on the same 300 items (chance 25%):

| model | audio only (video blanked) | vision only (audio silent) | both |
|---|---|---|---|
| video-SALMONN 2+ | 65.33% | 95.00% | 93.67% |
| VideoLLaMA2.1-AV | **93.67%** | 94.33% | 97.33% |

For VideoLLaMA2.1-AV **either channel alone nearly solves MUSIC-AVQA**, so "both"
cannot exceed "best single" by much — condition (3) is bounded above at ~+4 pp by the
benchmark, not by the model. For video-SALMONN 2+ audio alone reaches only 65%, which
is what leaves room for the +17 pp gap.

This supplies a mechanism for the near-zero `Delta_A(clean)` reported here and in prior
work: **the benchmark is massively modality-redundant**, so removing one channel costs
little regardless of whether the model can use it. It is a property of the testbed, not
evidence that models ignore audio.

**Consequence for scope.** M1 as registered cannot be settled on MUSIC-AVQA alone: a
benchmark where each channel is independently near-sufficient has no headroom for the
utilisation gap. Any confirmatory test of M1 requires a domain where single-channel
accuracy is well below joint accuracy. Domain 2 (AVUT — audio-centric,
text-shortcut-filtered) is therefore not optional breadth; it is the precondition for
the claim to be testable at all.

### 2026-08-16 — AVUT domain: VideoLLaMA2.1-AV fails delivery validation (`unvalidated`)

Recorded as an **outcome**, not a change to any registered criterion. Thresholds and the
delivery-gain definitions are exactly those already applied to MUSIC-AVQA.

VideoLLaMA2.1-AV on the AVUT pilot (300 items / 117 videos):

| control | MUSIC-AVQA | AVUT | criterion |
|---|---|---|---|
| audio delivery gain | +67.33 pp | **+0.33 pp** | ≥ +20 pp → **FAIL** |
| visual delivery gain | +68.00 pp | **+15.00 pp** | ≥ +20 pp → **FAIL** |

Per §3.3 the model is **`unvalidated` on AVUT**: its Δ_A(v) ≈ 0 at every visual severity
and `SG` = −1.0 pp [−5.6, +3.5] are **not reported as a null result about the model**,
because a non-delivering channel produces exactly that surface. The AVUT grid is still
completed and archived so the record is not a partial one, but no AVUT cell for this
model enters any confirmatory family.

Three candidate mechanisms were tested and none explains the failure: clip length
(gain +1.4 / +0.0 / +0.0 pp for <30 s / 30–60 s / >60 s), task family requiring speech
(+2.2 pp speech vs −2.5 pp non-speech), and audio stream format (both corpora aac /
44.1 kHz / stereo). Answer parsing is clean (0 empty, 0 non-ABCD of 300). The same clips
yield video-SALMONN 2+ a Δ_A(clean) of +13.3 pp, so the audio is present and usable.

**Registered limitation of §3.3 arising from this.** The delivery control uses task
accuracy as its proxy for delivery, so it cannot distinguish *signal not delivered* from
*signal delivered but unusable for this task family*. It is therefore sufficient to
**withhold** a null (its purpose) but not to **diagnose** one. Future registrations should
pair each domain with a channel-specific probe task the model is independently known to
solve on the same clips. This limitation is to be stated in the paper, not omitted.

**Ordering change adopted for all future sweeps.** Delivery controls run *before* the
graded grid, not after it (`scripts/run_avut_poscontrols.sh`). On the original ordering
this failure would have surfaced only after the final cell, by which point an
uninterpretable null had the shape of a finding.

### 2026-08-16 — AVUT domain: video-SALMONN 2+ also fails delivery; M1 FINAL = NOT MET

Second outcome for the same domain, recorded under the criteria fixed at 2026-08-13 §1.
No threshold was relaxed.

| AVUT | audio delivery gain | 95% CI | vs. ≥ +20 pp |
|---|---|---|---|
| video-SALMONN 2+ | +14.00 pp | [+8.39, +19.68] (excludes 0) | **FAIL** |
| VideoLLaMA2.1-AV | +0.33 pp | [−4.07, +4.70] (includes 0) | **FAIL** |

Both models are `unvalidated` on AVUT, so **no AVUT cell enters any confirmatory family**
and M1 still rests on one model in one domain: **FINAL VERDICT NOT MET** (see
`M1_VERDICT.md`). The AVUT grid is completed and archived regardless, so the record is not
partial.

**The two failures are qualitatively different and the criterion cannot tell them apart.**
video-SALMONN's gain excludes zero — audio is delivered and carries task information, it
merely does not reach an absolute 20 pp (upper CI bound +19.68). VideoLLaMA2.1-AV's
includes zero — indistinguishable from no delivery at all.

**Second registered limitation of §3.3.** The 20 pp bar is an *absolute* threshold on a
quantity the benchmark bounds. It was calibrated on a domain with high single-channel
accuracy, and it is structurally in tension with the property M1 requires: a domain where
neither channel alone suffices must produce small single-channel gains. A domain therefore
cannot easily satisfy both "testable for M1" and "passes delivery validation".

*Recommended replacement, for the next registration only — not applied here:* replace the
absolute bar with (a) a test that the gain excludes zero, plus (b) a channel-specific
probe task the model is independently known to solve from that channel alone on the same
clips. This also fixes the first §3.3 limitation recorded above, since a probe measures
delivery directly rather than through task accuracy.

**Pooled-visual basis is now reproducible.** `scripts/pooled_sg.py` recomputes the
preregistered §4 statistic from the cell JSONs; it reproduces the recorded
`SG = +9.08 pp [+6.75, +11.58]` for video-SALMONN 2+ on MUSIC-AVQA exactly.

### 2026-08-19 — EXPLORATORY: audio access is not the cause of the AVUT null; the front end is

**Not preregistered.** This is a diagnostic follow-up prompted by an internal adversarial
review, run after the confirmatory grid was complete. It is reported as exploratory and
changes no registered criterion or outcome.

**What the review found.** Two undisclosed mechanisms, either individually sufficient to
explain VideoLLaMA2.1-AV's `+0.33 pp` audio delivery on AVUT, both missed by the
covariate checks recorded on 2026-08-16:

1. **Audio access is confounded with domain.** The real condition is assembled by upstream
   `process_audio_from_video` via `ConstantClipsPerVideoSampler(clip_duration=2,
   clips_per_video=8)`: eight 2-second windows spliced end to end and zero-padded to 30 s,
   i.e. 16 s with the time axis broken at every splice, at any clip length. Measured over
   the actual media: the anchor domain (median 10.0 s) gets **100.0%** coverage on 299/299
   clips; AVUT (median 56.2 s) gets **28.5% median, 12.2% worst**, with only 9/117 clips
   fully covered.
2. **Front-end capability mismatch.** VideoLLaMA2.1-AV is fronted by **BEATs**
   (`videollama2/model/encoder.py:10`), an AudioSet event-tagging encoder with no lexical
   readout; video-SALMONN 2+ is fronted by **Whisper**. AVUT is a speech-understanding
   benchmark: 182/300 pilot items are lexical (Character Matching 85, Information
   Extraction 50, OCR Matching 47) over meeting / interview / talkshow / debate / TED
   speech video.

**Controlled test.** `--audio_access contiguous` was added to the adapter (default
behaviour unchanged; the flag is recorded in every output payload). It feeds the real
condition the first 30 s unbroken, through the same fbank tail the null conditions use.
The silent condition is byte-identical on both paths and was reused as the baseline.

| VideoLLaMA2.1-AV, blank-video delivery control | gain | 95% CI |
|---|---|---|
| AVUT, `sampled` (8x2 s spliced, 28.5% coverage) | +0.33 pp | [−4.07, +4.70] |
| AVUT, `contiguous` (first 30 s, ~53% coverage) | **+1.67 pp** | [−2.70, +6.14] |
| anchor, `sampled` | +67.33 pp | — |
| anchor, `contiguous` | **+67.33 pp** | — |

**Outcome.** Roughly doubling audio coverage and restoring the time axis moves the AVUT
delivery gain by **+1.33 pp**, leaving it far below the 20 pp bar with a CI still covering
zero. On the anchor domain, where all clips are under 18 s and the two paths are
near-equivalent by construction, the change is **exactly 0.00 pp** — so the manipulation
did not damage the audio path and behaves as predicted where it should be inert.

**Audio access is therefore a real confound but not the cause.** The binding constraint is
the front end: an event-tagging encoder cannot perform lexical tasks, and no amount of
audio access repairs that. This supersedes the 2026-08-16 statement that no candidate
mechanism explained the null — that statement was wrong, and it was wrong because the
covariate list omitted the encoder architecture and the harness's audio sampler.

**Consequence for §3.3.** This is a concrete instance of the first registered limitation:
the delivery control, scored by task accuracy, cannot separate *not delivered* from
*delivered but unusable by this front end*. Here the separation required an intervention on
the harness, not a split of the existing data. A channel-specific probe would have
diagnosed it directly.

### 2026-09-06 — RETROSPECTIVE: the anchor benchmark was replaced; earlier "MUSIC-AVQA" entries refer to a different corpus

**This entry is written after the data it describes**, and is labelled retrospective for
that reason. It should not be read as a pre-data record. It exists because the amendment
log otherwise stops on 2026-08-19 while every cell in the current manuscript dates from
2026-08-21 to 2026-08-30, which would leave the paper's entire evidentiary basis
unrecorded.

**What happened.** An adversarial internal review established that the corpus this project
had called "MUSIC-AVQA" throughout — `data/instruct/avqa_pilot_300.jsonl` — was not
MUSIC-AVQA (Li et al., CVPR 2022). It was a VGGSound-derived audio-event set: five
templated question stems over VGGSound class labels with three distractors, in which
audio-sufficiency holds by construction. The misattribution originated in this project's
own documentation and was propagated without checking the raw annotations.

**Every amendment dated before 2026-08-21 that says "MUSIC-AVQA" refers to that
misattributed corpus, not to MUSIC-AVQA.** The most consequential instance is the
2026-08-14 entry reporting VideoLLaMA2.1-AV at "93.67 % audio only, video blanked". On the
real benchmark the same measurement is **42.33 %** against a 24.33 % floor. The
benchmark-redundancy claim built on that number is **withdrawn**; it was an artefact of the
misattributed set. This entry supersedes it rather than the earlier text being edited, so
the record of what was believed when stays intact.

**Replacement anchor, and its provenance.**

| field | value |
|---|---|
| dataset | MUSIC-AVQA (Li et al., CVPR 2022), official release |
| annotations | GeWu-Lab/MUSIC-AVQA `data/json_update/avqa-test.json`, 9{,}129 active records |
| annotations sha256 | `76ce24ee02fd85e28038605aedd4f62633e1335ccda96a1fcf7340b5feaf230e` |
| videos | official real-video archive only; the synthetic (`esa*`) subset was never downloaded |
| coverage | 7{,}365 of 9{,}129 test items have a real video (80.7 %) |
| pilot | 300 items / 206 videos, held out by whole video, seed 2027 |
| pilot sha256 | `76fbcaa9067ed1252e663d0fc741f4f56c47adc9ec411dc8299db3cbbcac6093` |
| scoring | closed 41-word vocabulary presented in each item's own prompt; normalised exact match |

**What was fixed before any outcome was seen, and what was not.** The split rule (hold out
whole videos, seed 2027, target 300 items) is the same rule used for the AVUT split and was
not tuned. The scoring decision (closed vocabulary, deterministic exact match rather than
the LLM-judge convention) was made before the grid, and validated on a 60-item set drawn
from **held-out** videos so that format decisions could not be tuned on the pilot: both
models parsed 60/60. The delivery controls were run **first** rather than last, per the
2026-08-16 ordering change.

**Not preregistered.** The crossover between audio front end and benchmark audio type, the
fusion-loss quantity, and the bound $\Delta_A(v) \le G_A$ are all exploratory. They were
formulated after the confirmatory grid, prompted by review. The manuscript labels them so.

**Outcome unchanged.** The preregistered primary claim (M1) remains **NOT MET**. No
threshold was altered at any point in this replacement.
