"""Five-mode audio intervention protocol applied to video-SALMONN 2+ 7B.

External-model validation of the protocol (Strategy A). Non-model logic is
MIRRORED VERBATIM from the contract file:
    $VIDEOLLAMA2_DIR/eval_avqa_5mode.py
Reused unchanged: SR / CLIP_SECONDS, CHOICE_RE answer-letter regex,
AUDIO_MODES, build_shuffled_mapping, extract_answer_letter,
extract_ground_truth, parse_args CLI shape, POS_IDX / POS_BLANK env
handling, the per-record loop, GT/question parsing, `correct` counting,
the progress print, and the exact output JSON schema.

Mode semantics (audio only; video frames are always the real frames):
    real     : original audio
    silent   : zero waveform
    noise    : gaussian noise waveform
    shuffled : audio from a deterministically chosen different eval video
    shifted  : same audio, circularly shifted by `--shift_seconds`

Every mode passes the IDENTICAL Whisper front-end: the contract's per-mode
math STOPS at the 16 kHz mono waveform, then the model's OWN
WhisperFeatureExtractor + 30 s chunking (replicated from the repo's
LazySupervisedDataset.process_audio) is run on that waveform. This mirrors
the contract's `waveform_to_fbank` correctness principle.

================================================================================
STATIC SOURCE CITATIONS (read-only; nothing executed during authoring)
Repo: $WORK_DIR/video-SALMONN-2/video_SALMONN2_plus
--------------------------------------------------------------------------------
(a) Audio key:
    - dataset.py:679  `data_dict["audio_feature"] = audio`            (the
      Whisper input_features tensor; `audio` = torch.cat(audio_inputs))
    - dataset.py:680  `data_dict["audio_lengths"] = audio_lengths`    (list;
      drives the `<|audio_pad|>` token count baked into input_ids via
      generate_id_target / preprocess_qwen_2_visual, dataset.py:160-174)
    - dataset.py:707  in run_test mode `data_dict.pop("audio_lengths", None)`
      -> audio_lengths is NOT returned to the caller; the pad tokens are
      already encoded in input_ids, and the model's audio path consumes
      only `audio_feature` (modeling_qwen2_5_vl.py:2085 get_audio_embeds).
    - modeling_qwen2_5_vl.py:2417-2418  forward(... audio_feature=..., audio_lengths=...)
    - modeling_qwen2_5_vl.py:2493        prepare_inputs_for_generation(audio_feature=None, ...)
(b) Pixel (video) key:
    - dataset.py:673  `data_dict["pixel_values_videos"] = torch.cat(video, dim=0)`
    - dataset.py:674  `data_dict["video_grid_thw"] = torch.cat(...)`
(c) WhisperFeatureExtractor construction:
    - inference.py:41   `from transformers import AutoTokenizer, WhisperFeatureExtractor`
    - inference.py:112-117  data_args.audio_processor = WhisperFeatureExtractor(
          feature_size=data_args.feature_size, sampling_rate=data_args.sampling_rate,
          hop_length=data_args.hop_length, chunk_length=data_args.chunk_length)
    - argument.py:52-55  DataArguments defaults: feature_size=128,
          chunk_length=30, hop_length=160, sampling_rate=16000
    - dataset.py:372-408  process_audio: deepcopy(audio_processor); 30 s chunks
          `audio[k:k+30*16000]`; per chunk
          `processor(a, sampling_rate=16000, return_tensors="pt")["input_features"].squeeze()`;
          torch.stack over chunks; audio_lengths = ceil(len/(30*16000))*60.
(d) generate call:
    - inference.py:160-170  inputs = test_data._get_item(input_dict);
          inputs = prepare_inputs(inputs);
          outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False);
          output_trimmed = outputs[0, len(inputs["input_ids"][0]):];
          tokenizer.decode(output_trimmed, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    - model load: inference.py:127-143 (AutoTokenizer use_fast=False,
          model_max_length=131072; video_SALMONN2_plus.from_pretrained(
          attn_implementation="flash_attention_2",
          torch_dtype=torch.bfloat16, device_map="cpu"); model.cuda())
    - dataset build: inference.py:103-143 prepare_dataset() pins
          video_max_frames=768, video_min_frames=16, base_interval=0.1,
          max_pixels=61250, video_max_frame_pixels=61250, run_test=True,
          model_type="qwen2.5vl"; make_supervised_data_module ->
          data_module["train_dataset"] (empty dataset_use="", so _get_item
          is invoked directly on an in-memory record, dataset.py:516).

RESIDUAL RISK (cannot be exercised statically, conda env still building):
  - audio_lengths is popped before return (dataset.py:707); the audio-pad
    token count is therefore fixed at _get_item time. We rebuild the audio
    waveform-side BEFORE _get_item by overriding the bound `process_audio`
    so that audio_feature AND audio_lengths stay mutually consistent for
    every mode (same chunk count -> same pad count). This is the only safe
    static design; verify token/feature alignment on-machine.
  - `prepare_inputs_for_generation` whitelists `audio_feature`; HF generate
    forwards remaining model_kwargs, so passing `audio_feature` as a tensor
    in the inputs dict is sufficient (audio_lengths intentionally NOT
    re-added, mirroring stock inference exactly).
================================================================================

Usage (run inside the video-SALMONN-2 conda env):
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_avqa_videosalmonn2.py \
        --model_path $MODELS_DIR/video-SALMONN2_plus_7B_full \
        --eval_data $REPO_ROOT/data/instruct/avqa_test_clean.jsonl \
        --video_dir $REPO_ROOT/data/videos \
        --audio_mode real --output out_real.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

# --- Contract constants (mirrored verbatim from eval_avqa_5mode.py) ----------
SR = 16000
CLIP_SECONDS = 30
CHOICE_RE = re.compile(r"[A-D]")

# --- closed-vocabulary answers (MUSIC-AVQA) --------------------------------------
# Loaded by path so both adapters read model output with byte-identical logic; a
# divergence here would show up as a between-model difference that is really a
# between-parser difference.
_AM_PATH = Path("$REPO_ROOT") / "src" / "videollm" / "data" / "answer_match.py"
_AM_SPEC = importlib.util.spec_from_file_location("answer_match", _AM_PATH)
_AM = importlib.util.module_from_spec(_AM_SPEC)
sys.modules["answer_match"] = _AM
_AM_SPEC.loader.exec_module(_AM)

_VOCAB_RE = re.compile(r"one of the following words:\s*\n(.+?)\n", re.DOTALL)


def prompt_vocabulary(prompt: str) -> tuple[str, ...]:
    """Read the candidate list out of the item's own prompt.

    Deriving it from the prompt rather than a side file guarantees the reader scores
    against exactly the words the model was shown, per item.
    """
    m = _VOCAB_RE.search(prompt)
    if not m:
        return ()
    return tuple(w.strip() for w in m.group(1).split(",") if w.strip())

AUDIO_MODES = ("real", "silent", "noise", "shuffled", "shifted")

# CMSS visual axis. Imported by path, not as `videollm.…`, because this adapter
# runs inside the video-SALMONN conda env where that package is absent.
_VD_PATH = Path(__file__).resolve().parent.parent / "src" / "videollm" / "data" / "video_degradation.py"
_VD_SPEC = importlib.util.spec_from_file_location("video_degradation", _VD_PATH)
assert _VD_SPEC is not None and _VD_SPEC.loader is not None
_VD = importlib.util.module_from_spec(_VD_SPEC)
# Register before exec: @dataclass resolves its module via sys.modules and
# raises AttributeError if the module is absent there.
sys.modules["video_degradation"] = _VD
_VD_SPEC.loader.exec_module(_VD)
VISUAL_MODES = _VD.VISUAL_MODES

# Repo paths (Strategy A requirement 1: sys.path insert the repo subdir).
REPO_DIR = Path("$WORK_DIR/video-SALMONN-2/video_SALMONN2_plus")


def build_shuffled_mapping(n: int, seed: int) -> list[int]:
    """Deterministic derangement (no sample maps to itself); matches contract."""
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(100):
        perm = idx[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    return [(i + 1) % n for i in range(n)]


def extract_answer_letter(text: str) -> str:
    m = CHOICE_RE.search(text.upper())
    return m.group(0) if m else ""


def extract_ground_truth(gpt_response: str) -> str:
    m = re.search(r"answer is ([A-D])", gpt_response, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="video-SALMONN2+ 7B five-mode AVQA eval")
    p.add_argument(
        "--model_path",
        default="$MODELS_DIR/video-SALMONN2_plus_7B_full",
    )
    p.add_argument("--eval_data", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--audio_mode", default="real", choices=list(AUDIO_MODES))
    p.add_argument("--shift_seconds", type=float, default=3.0)
    p.add_argument("--visual_mode", default="clean", choices=list(VISUAL_MODES),
                   help="Visual degradation family (CMSS visual axis)")
    p.add_argument("--visual_severity", type=float, default=1.0,
                   help="Visual degradation severity in [0, 1]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--answer_format",
        default="letter",
        choices=("letter", "vocab"),
        help=(
            "How to read the model's answer. 'letter' (default) extracts an A-D choice, for "
            "4-way multiple-choice sets such as AVUT. 'vocab' matches the reply against the "
            "closed answer vocabulary listed in the item's own prompt, for MUSIC-AVQA, whose "
            "41 answers are words rather than letters."
        ),
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


# --- Per-mode waveform math (mirrors contract get_mode_fbank, but STOPS at the
#     16 kHz mono waveform; the model's own Whisper front-end runs on it) -----
def _safe_load_wav(video_path: str) -> np.ndarray:
    """Decode a video's audio track to a 16 kHz mono float32 waveform.

    Uses the same AudioDecoder backend the repo's dataset.process_audio uses,
    so the waveform we feed back is byte-for-byte what the real path would see.
    """
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(video_path, sample_rate=SR, num_channels=1)
        audio = decoder.get_all_samples()
        wav = audio.data.numpy().squeeze(0)
        return np.asarray(wav, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - eval robustness, mirror upstream
        print(f"[warn] audio load failed for {video_path}: {exc}")
        return np.zeros(CLIP_SECONDS * SR, dtype=np.float32)


def get_mode_waveform(
    mode: str,
    video_path: str,
    donor_path: str | None,
    shift_seconds: float,
) -> np.ndarray:
    """Return the per-mode 16 kHz mono waveform (contract math, waveform-level).

    Mirrors eval_avqa_5mode.get_mode_fbank exactly, including the
    np.roll(wav, int(shift_seconds*SR)) shift and the build_shuffled_mapping
    donor selection (donor path resolved by the caller).

    ``silent`` and ``noise`` match the **real audio's length** rather than using a fixed
    ``CLIP_SECONDS`` window. The model chunks audio into 30 s segments and emits one
    ``<|audio_pad|>`` block per chunk, so a fixed 30 s null on a longer clip supplies
    fewer blocks than the prompt expects and the model then generates an empty string.
    That failure is invisible on MUSIC-AVQA (every clip is under 30 s) and hit 223 of 300
    AVUT items, where clips run to 131 s.
    """
    if mode in ("silent", "noise"):
        n = len(_safe_load_wav(video_path))  # match the real recording's length
        if n <= 0:
            n = CLIP_SECONDS * SR
        if mode == "silent":
            return np.zeros(n, dtype=np.float32)
        return np.random.randn(n).astype(np.float32)
    if mode == "shuffled":
        # The donor is a different clip, so its length differs from this item's. The model
        # chunks audio into 30 s segments and emits one <|audio_pad|> block per chunk, so a
        # donor of the wrong length changes the block count and the forward pass fails to
        # broadcast (17 AVUT items failed this way). Match the item's own recording length,
        # exactly as the silent and noise nulls above do: the intervention is the audio
        # content, not its duration.
        donor = _safe_load_wav(donor_path or video_path)
        n = len(_safe_load_wav(video_path))
        if n <= 0:
            n = CLIP_SECONDS * SR
        if len(donor) >= n:
            return donor[:n].astype(np.float32)
        reps = int(np.ceil(n / max(len(donor), 1)))
        return np.tile(donor, reps)[:n].astype(np.float32)
    if mode == "shifted":
        wav = _safe_load_wav(video_path)
        return np.roll(wav, int(shift_seconds * SR)).astype(np.float32)
    raise ValueError(f"get_mode_waveform should not be called for mode={mode!r}")


def _make_process_video_frames_override(original, degradation):
    """Build a replacement for the dataset's bound `process_video_frames`.

    The visual analogue of `_make_process_audio_override`: the intervention is applied
    to the DECODED frames, then the repo's own preprocessing (`process_video_frames`,
    dataset.py:473) runs unchanged, so patch counts and `video_grid_thw` stay
    consistent. A `clean` config returns the original bound method untouched, keeping
    behaviour byte-identical to stock inference.
    """
    if degradation is None or degradation.mode == "clean" or degradation.severity == 0.0:
        return original

    def process_video_frames(video, frame_idx, video_length):
        # Upstream hands us uint8 (T, C, H, W); the degradation contract is float [0, 1].
        frames = torch.from_numpy(np.ascontiguousarray(video)).float() / 255.0  # (T, C, H, W)
        degraded = _VD.apply_visual_degradation(frames, degradation)  # (T, C, H, W)
        restored = (degraded * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        return original(restored, frame_idx, video_length)

    return process_video_frames


def _make_process_audio_override(test_data, mode_waveform: np.ndarray | None):
    """Build a replacement for the dataset's bound `process_audio`.

    Replicates LazySupervisedDataset.process_audio (dataset.py:372-408)
    exactly EXCEPT the decode step: when `mode_waveform` is provided (any
    non-`real` mode) the supplied waveform is used instead of decoding the
    file, so the per-mode audio passes the model's OWN WhisperFeatureExtractor
    and the 30 s chunking. audio_feature AND audio_lengths are produced by the
    identical math, keeping the `<|audio_pad|>` token count consistent.

    For `mode == "real"` (mode_waveform is None) this falls back to the repo's
    original bound method so behaviour is byte-identical to stock inference.
    """
    if mode_waveform is None:
        return test_data.process_audio  # real: use the repo's exact path

    data_args = test_data.data_args

    def process_audio(audio_file):  # noqa: ARG001 - signature must match repo
        try:
            audio_kwargs = {
                "sampling_rate": 16000,
                "padding": "max_length",
                "return_attention_mask": False,
            }
            processor = copy.deepcopy(data_args.audio_processor)
            audio_data = [np.asarray(mode_waveform, dtype=np.float32)]
            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                if audio_data[idx].shape[0] < audio_kwargs["sampling_rate"]:
                    padding = audio_kwargs["sampling_rate"] - audio_data[idx].shape[0]
                    audio_data[idx] = np.pad(
                        audio_data[idx], (0, padding), mode="constant", constant_values=0
                    )
                audio_lst = [
                    audio_data[idx][k : k + 30 * audio_kwargs["sampling_rate"]]
                    for k in range(0, len(audio_data[idx]), 30 * audio_kwargs["sampling_rate"])
                ]
                spectrogram_lst = [
                    processor(a, sampling_rate=audio_kwargs["sampling_rate"], return_tensors="pt")[
                        "input_features"
                    ].squeeze()
                    for a in audio_lst
                ]
                audio_inputs.append(torch.stack(spectrogram_lst, dim=0))
                audio_lengths.append(
                    math.ceil(len(audio_data[idx]) / (30 * audio_kwargs["sampling_rate"])) * 60
                )
            return audio_inputs, audio_lengths
        except Exception as exc:  # noqa: BLE001 - mirror upstream bare-except contract
            print(f"[warn] process_audio override failed: {exc}")
            return None, None

    return process_audio


def custom_prepare_inputs(inputs: dict) -> dict:
    """Like inference.py:prepare_inputs but does NOT drop the audio tensor key.

    Stock prepare_inputs (inference.py:91-100) pops "audio" and other test-only
    scaffolding then keeps only tensors. The Whisper features live under
    `audio_feature` (dataset.py:679), which stock code keeps because it is a
    tensor; we keep that behaviour and additionally strip the test-only
    scaffolding keys explicitly. All tensors are moved to the current CUDA
    device (cuda:0 under CUDA_VISIBLE_DEVICES=2 - never hardcode cuda:2).
    """
    inputs.pop("video", None)
    inputs.pop("image", None)
    inputs.pop("prompt", None)
    inputs.pop("ref", None)
    inputs.pop("audio", None)  # raw audio path scaffolding (NOT audio_feature)
    inputs.pop("use_audio", None)
    inputs.pop("should_use", None)
    dev = f"cuda:{torch.cuda.current_device()}"
    return {k: v.to(dev) for k, v in inputs.items() if isinstance(v, torch.Tensor)}


def main() -> None:
    args = parse_args()
    # Deterministic seeding (CHECK.md B.6) for reproducible noise mode.
    import random as _r
    _r.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Strategy A req 1: sys.path insert the repo subdir BEFORE repo imports.
    sys.path.insert(0, str(REPO_DIR))

    import transformers  # noqa: F401  (ensures repo's transformers is importable)
    from qwenvl.data.dataset import make_supervised_data_module
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
    from qwenvl.train.argument import DataArguments
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    # Strategy A req 6: pin dataset args exactly as inference.py:prepare_dataset.
    data_args = DataArguments()
    data_args.video_max_frames = 768
    data_args.video_min_frames = 16
    data_args.base_interval = 0.1
    data_args.max_pixels = 61250
    data_args.video_max_frame_pixels = 61250
    data_args.run_test = True
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(args.model_path)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=131072,
        padding_side="right",
        use_fast=False,
    )

    model = video_SALMONN2_plus.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.cuda()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    test_data = data_module["train_dataset"]

    # --- Contract: load records, video dir, shuffled map, POS_* env ----------
    records = [json.loads(line) for line in Path(args.eval_data).read_text().splitlines()]
    video_dir = Path(args.video_dir)
    shuffled_map = build_shuffled_mapping(len(records), args.seed)
    # Capture the pristine bound method once; re-reading it after an override
    # would wrap an already-wrapped function.
    _orig_process_video_frames = test_data.process_video_frames

    _pos_idx_path = os.environ.get("POS_IDX")
    _pos_idx = set(json.loads(Path(_pos_idx_path).read_text())) if _pos_idx_path else None
    _pos_blank = os.environ.get("POS_BLANK") == "1"

    preds: list[dict[str, str]] = []
    correct = 0
    for i, rec in enumerate(records):
        if _pos_idx is not None and i not in _pos_idx:
            continue
        video = str(rec["video"])
        vpath = str(video_dir / video)
        human = rec["conversations"][0]["value"]
        gt = extract_ground_truth(rec["conversations"][1]["value"])
        question = human.replace("<video>", "").replace("<audio>", "").strip()

        try:
            # Strategy A req 3: per-mode audio is computed waveform-level, then
            # routed through the model's OWN Whisper front-end by overriding the
            # dataset's bound process_audio for this record.
            if args.audio_mode == "real":
                mode_wav = None
            else:
                donor = None
                if args.audio_mode == "shuffled":
                    donor = str(video_dir / str(records[shuffled_map[i]]["video"]))
                mode_wav = get_mode_waveform(
                    args.audio_mode, vpath, donor, args.shift_seconds
                )
            test_data.process_audio = _make_process_audio_override(test_data, mode_wav)
            # Visual axis: per-item deterministic seed, same contract as eval_avqa.py.
            test_data.process_video_frames = _make_process_video_frames_override(
                _orig_process_video_frames,
                _VD.DegradationConfig(
                    mode=args.visual_mode, severity=args.visual_severity, seed=args.seed + i
                ),
            )

            # Strategy A req 2: use_audio=True ALWAYS (we always supply audio,
            # even for `silent`); repo chat format with `<video>\n` prefix and
            # an empty gpt turn; question already stripped of <video>/<audio>.
            input_dict = {
                "video": vpath,
                "use_audio": True,
                "conversations": [
                    {"from": "human", "value": "<video>\n" + question},
                    {"from": "gpt", "value": ""},
                ],
            }

            inputs = test_data._get_item(input_dict)

            # Strategy A req 5: POS_BLANK -> zero the pixel tensor (analogue of
            # the contract's av["video"]=torch.zeros_like(...)), keep audio on.
            if _pos_blank and inputs.get("pixel_values_videos") is not None:
                inputs["pixel_values_videos"] = torch.zeros_like(
                    inputs["pixel_values_videos"]
                )

            # Strategy A req 4: custom prepare_inputs that keeps audio_feature.
            inputs = custom_prepare_inputs(inputs)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            output_trimmed = outputs[0, len(inputs["input_ids"][0]) :]
            out = tokenizer.decode(
                output_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:  # noqa: BLE001 - keep eval going, log failure
            print(f"[warn] inference failed sample {i} ({video}): {exc}")
            out = ""

        pred = extract_answer_letter(out)
        if args.answer_format == "vocab":
            vocab = prompt_vocabulary(human)
            pred = _AM.match_answer(out, vocab)
            gt = _AM.match_answer(rec["conversations"][1]["value"], vocab)
        is_ok = pred == gt and gt != ""
        correct += int(is_ok)
        preds.append(
            {
                "index": str(i),
                "video": video,
                "gt": gt,
                "pred": pred,
                "correct": str(is_ok),
                "output": out,
            }
        )
        if (i + 1) % 10 == 0:
            print(
                f"[{i+1}/{len(records)}] acc={100*correct/(i+1):.1f}% "
                f"mode={args.audio_mode}"
            )

    total = len(preds)  # processed count (== len(records) unless POS_IDX subset)
    result = {
        "accuracy": 100.0 * correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "audio_mode": args.audio_mode,
        "answer_format": args.answer_format,
        "shift_seconds": args.shift_seconds if args.audio_mode == "shifted" else None,
        "visual_mode": args.visual_mode,
        "visual_severity": args.visual_severity if args.visual_mode != "clean" else 0.0,
        "model": args.model_path,
        "predictions": preds,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(
        f"[done] {args.audio_mode}: {result['accuracy']:.2f}% "
        f"({correct}/{total}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
