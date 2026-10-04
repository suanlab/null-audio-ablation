# Stale: qwenomni_*.json and poscontrol_qwenomni_*.json

**Do not aggregate, cite, or reuse these files.** Kept, not deleted, because the paper's
Sec. XII ("Two Front Ends We Could Not Report") describes the failure they came from, and
the record of that episode should stay inspectable.

## What is wrong with them

Two independent problems, either one disqualifying.

1. **The audio never reached the model.** `scripts/eval_avqa_qwenomni.py` built its
   conversation with a video block and a text block but no audio block, so the chat
   template emitted no `<|AUDIO|>` placeholder. The waveform passed to the processor was
   computed into `input_features` and then never referenced by any token position -- an
   inert tensor. Measured on the rendered prompt: 42 input tokens without the audio block,
   144 with it (the placeholder expands to 102 audio tokens). Identical under both
   transformers 4.57.6 and 5.1.0.

   The signature in the data: all six audio modes share one SHA-256 over their
   predictions, and the blank-video delivery control agrees real-vs-silent on 47/47 items.

   Fixed 2026-09-14 by rendering the template from a conversation that declares audio while
   running `process_mm_info` on a video-only copy, so the harness still supplies exactly
   the waveform `mode_wav()` built. A guard now raises if the template emits no
   `<|AUDIO|>`.

2. **Superseded benchmark.** These runs used the 98-item A-D multiple-choice AVQA subset,
   which the 2026-09-06 amendment replaced with the 300-item MUSIC-AVQA pilot scored over
   its 41-word closed vocabulary. Even with the audio bug fixed, numbers from this subset
   are not comparable with any cell in the current grid.

## Why byte-identical output was not, by itself, proof of a bug

A model that ignores audio entirely also produces byte-identical greedy output across
audio conditions. The two hypotheses are separated by the delivery control, not by the
grid: on blank video, an audio-only-answerable item must move if audio arrives. It did not.
That is what identified this as delivery failure inside our harness rather than a property
of Qwen2.5-Omni -- the same distinction the paper draws for the models it does report.
