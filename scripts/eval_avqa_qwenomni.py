"""B4 (2nd external model): five-mode audio protocol on Qwen2.5-Omni-7B.

Cross-family external validation. Same matched 98-sample AVQA subset, same
mode semantics, same deterministic shuffled mapping, same output schema as
scripts/eval_avqa.py and VideoLLaMA2/eval_avqa_5mode.py, so results drop into
scripts/aggregate_matched_98.py as a 5th model (prefix ``qwenomni``).

Audio-only intervention; video frames are always the real frames. We supply
our own controlled 16 kHz waveform and set use_audio_in_video=False so the
model consumes exactly the audio we pass. The conversation must declare an
audio block or the template emits no <|AUDIO|> placeholder and the waveform is
silently ignored -- see the guard in run().

Usage (videollm venv, transformers 5.1 has Qwen2.5-Omni):
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_avqa_qwenomni.py \
        --eval_data data/instruct/avqa_test_clean.jsonl \
        --video_dir data/videos --audio_mode real \
        --output eval_results/qwenomni_real.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

SR = 16000
# Audio fed to the model. Every cell cited before 2026-11-09 was run at the default: the first
# 30 s of each track, zero-padded to 30 s, with silence and noise also 30 s long. "full" passes
# the whole track up to the processor's 300 s limit, and sizes silence and noise to that item's
# real length so that the null differs from real audio in content only, not in length.
AUDIO_SECONDS = os.environ.get("QWEN_AUDIO_SECONDS", "30")
FULL_TRACK = AUDIO_SECONDS == "full"
CLIP_SECONDS = 300 if FULL_TRACK else int(AUDIO_SECONDS)
CHOICE_RE = re.compile(r"[A-D]")
AUDIO_MODES = ("real", "silent", "noise", "shuffled", "shifted")

# qwen_omni_utils budgets a video's pixels from MODEL_SEQ_LEN, which defaults to 128000 --
# nearly 4x this checkpoint's real context of 32768. Left at the default, high-resolution
# clips render to ~43.5k tokens and overflow silently: the tokenizer warns only once per
# process, and the adapter records no per-sample failure, so roughly a third of MUSIC-AVQA
# items would decode past the end of the position embeddings. Set before qwen_omni_utils
# is imported, since the module reads it at import time. 28000 leaves room for the audio
# placeholder (~1.5k tokens for a 60 s clip) and the 64 generated tokens; it only bites on
# clips that were over budget -- items already under it render unchanged.
MODEL_SEQ_LEN = int(os.environ.setdefault("MODEL_SEQ_LEN", "28000"))
CONTEXT_LIMIT = 32768

# CMSS visual axis. Imported by path rather than as `videollm.…` so this adapter
# also works in environments where that package is not installed.
_VD_PATH = Path(__file__).resolve().parent.parent / "src" / "videollm" / "data" / "video_degradation.py"
_VD_SPEC = importlib.util.spec_from_file_location("video_degradation", _VD_PATH)
assert _VD_SPEC is not None and _VD_SPEC.loader is not None
_VD = importlib.util.module_from_spec(_VD_SPEC)
# Register before exec: @dataclass resolves its module via sys.modules.
sys.modules["video_degradation"] = _VD
_VD_SPEC.loader.exec_module(_VD)
VISUAL_MODES = _VD.VISUAL_MODES

# Closed-vocabulary reader (MUSIC-AVQA). Loaded by path so all three adapters read model
# output with byte-identical logic; a divergence here would surface as a between-model
# difference that is really a between-parser difference.
_AM_PATH = Path(__file__).resolve().parent.parent / "src" / "videollm" / "data" / "answer_match.py"
_AM_SPEC = importlib.util.spec_from_file_location("answer_match", _AM_PATH)
assert _AM_SPEC is not None and _AM_SPEC.loader is not None
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


def degrade_videos(vids, degradation):
    """Apply the CMSS visual intervention to qwen_omni_utils video tensors.

    Placed at the same hook point as the POS_BLANK control: on the decoded video
    tensors, before the processor, so downstream grid/patch bookkeeping is unchanged.
    A clean or zero-severity config returns the input untouched.

    Args:
        vids: List of video tensors ``(T, C, H, W)``, or None.
        degradation: The intervention to apply.

    ``qwen_omni_utils.fetch_video`` decodes frames as uint8 and calls ``.float()`` after
    resizing without dividing, so at this hook point a float tensor carries **0-255**
    values, not the ``[0, 1]`` the degradation contract expects. An earlier version of
    this function assumed float meant ``[0, 1]`` and raised on everything the library
    actually produces, which is why the visual axis never ran for this model: every
    sample failed, and a cell of empty predictions is not distinguishable from a cell of
    wrong ones without reading the log. The sibling adapters take uint8 from their
    upstreams and divide by 255 themselves; this one divides the float.

    Args:
        vids: List of video tensors ``(T, C, H, W)``, or None.
        degradation: The intervention to apply.

    Returns:
        Degraded tensors with the input dtype, scale, and shape preserved.

    Raises:
        ValueError: If a tensor is outside ``[0, 255]``. Mean/std-normalised input has
            negative values, and clamping it into the degradation range would corrupt
            the frames rather than degrade them.
    """
    if vids is None or degradation is None or degradation.mode == "clean" or degradation.severity == 0.0:
        return vids
    out = []
    for v in vids:
        lo, hi = float(v.min()), float(v.max())
        if lo < 0.0 or hi > 255.0:
            raise ValueError(
                f"expected uint8 or float in [0, 255], got {v.dtype} range [{lo:.3f}, {hi:.3f}]"
            )
        frames = v.float() / 255.0  # (T, C, H, W) in [0, 1] for the degradation contract
        degraded = _VD.apply_visual_degradation(frames, degradation)  # (T, C, H, W)
        restored = (degraded * 255.0).clamp(0, 255)  # (T, C, H, W) back to the 0-255 scale
        if v.dtype == torch.uint8:
            out.append(restored.round().to(torch.uint8))
        else:
            out.append(restored.to(v.dtype))
    return out
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

SYS = ("You are Qwen, a virtual human able to perceive auditory and visual "
       "inputs, as well as generate text and speech.")


def load_wav16(video_path: str) -> np.ndarray:
    """Decode mono 16 kHz waveform from a video via librosa (ffmpeg backend)."""
    try:
        import librosa

        wav, _ = librosa.load(video_path, sr=SR, mono=True)
        return np.asarray(wav, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - eval robustness
        print(f"[warn] audio load failed {video_path}: {exc}")
        return np.zeros(SR, dtype=np.float32)


def fit_len(w: np.ndarray, t: int | None = None) -> np.ndarray:
    """Truncate or zero-pad to ``t`` samples; in full-track mode only cap at the processor limit."""
    if t is None:
        if FULL_TRACK:
            return w[: CLIP_SECONDS * SR]
        t = CLIP_SECONDS * SR
    if len(w) > t:
        return w[:t]
    if len(w) < t:
        return np.pad(w, (0, t - len(w)))
    return w


def null_len(vpath: str) -> int:
    """Length of the silence/noise null: 30 s by default, the item's own real length in full mode."""
    return len(fit_len(load_wav16(vpath))) if FULL_TRACK else CLIP_SECONDS * SR


def build_shuffled_mapping(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(100):
        perm = idx[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    return [(i + 1) % n for i in range(n)]


def mode_wav(mode: str, vpath: str, donor: str | None, shift_s: float) -> np.ndarray:
    if mode == "real":
        return fit_len(load_wav16(vpath))
    if mode == "silent":
        return np.zeros(null_len(vpath), dtype=np.float32)
    if mode == "noise":
        return np.random.randn(null_len(vpath)).astype(np.float32)
    if mode == "shuffled":
        donor_wav = load_wav16(donor or vpath)
        return fit_len(donor_wav, null_len(vpath)) if FULL_TRACK else fit_len(donor_wav)
    if mode == "shifted":
        return np.roll(fit_len(load_wav16(vpath)), int(shift_s * SR))
    raise ValueError(mode)


def ans_letter(t: str) -> str:
    m = CHOICE_RE.search(t.upper())
    return m.group(0) if m else ""


def gt_letter(s: str) -> str:
    m = re.search(r"answer is ([A-D])", s, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
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


def main() -> None:
    a = parse_args()
    # Deterministic seeding (CHECK.md B.6) for reproducible noise mode.
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    recs = [json.loads(x) for x in Path(a.eval_data).read_text().splitlines()]
    vdir = Path(a.video_dir)

    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="cuda"
    )
    model.eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()  # text-only QA; skip speech head

    import os as _os

    _pos_idx_path = _os.environ.get("POS_IDX")
    _pos_idx = set(json.loads(Path(_pos_idx_path).read_text())) if _pos_idx_path else None
    _pos_blank = _os.environ.get("POS_BLANK") == "1"

    import hashlib

    adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shuf = build_shuffled_mapping(len(recs), a.seed)
    preds, correct = [], 0
    for i, rec in enumerate(recs):
        if _pos_idx is not None and i not in _pos_idx:
            continue
        video = str(rec["video"])
        vpath = str(vdir / video)
        gt = gt_letter(rec["conversations"][1]["value"])
        q = rec["conversations"][0]["value"].replace("<video>", "").replace("<audio>", "").strip()
        wav = np.zeros(0, dtype=np.float32)
        try:
            donor = str(vdir / str(recs[shuf[i]]["video"])) if a.audio_mode == "shuffled" else None
            wav = mode_wav(a.audio_mode, vpath, donor, a.shift_seconds)
            # The chat template only emits <|audio_bos|><|AUDIO|><|audio_eos|> when the
            # conversation carries an audio block. Without it the rendered prompt has no
            # audio position, the waveform we pass rides along as an inert input_features
            # tensor, and every audio mode decodes byte-identically. Render the template
            # from a conversation that declares audio, but run process_mm_info on a
            # video-only copy so it never loads audio off disk -- we still supply exactly
            # the waveform mode_wav() built.
            user_mm = [
                {"type": "video", "video": vpath},
                {"type": "text", "text": q},
            ]
            conv_mm = [
                {"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": user_mm},
            ]
            conv_tmpl = [
                {"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "audio", "audio": vpath}, *user_mm]},
            ]
            chat = proc.apply_chat_template(conv_tmpl, add_generation_prompt=True, tokenize=False)
            if "<|AUDIO|>" not in chat:
                raise RuntimeError(
                    "chat template emitted no <|AUDIO|> placeholder; audio would be ignored"
                )
            from qwen_omni_utils import process_mm_info

            _au, imgs, vids = process_mm_info(conv_mm, use_audio_in_video=False)
            if _pos_blank and vids is not None:
                vids = [torch.zeros_like(v) for v in vids]
            # Visual axis: per-item deterministic seed, same contract as eval_avqa.py.
            vids = degrade_videos(
                vids,
                _VD.DegradationConfig(mode=a.visual_mode, severity=a.visual_severity, seed=a.seed + i),
            )
            inputs = proc(
                text=chat, audio=[wav], images=imgs, videos=vids,
                return_tensors="pt", padding=True, use_audio_in_video=False,
            ).to(model.device)
            n_tok = int(inputs["input_ids"].shape[1])
            if n_tok + 64 > CONTEXT_LIMIT:
                raise RuntimeError(
                    f"sequence {n_tok} + 64 generated exceeds context {CONTEXT_LIMIT}; "
                    f"lower MODEL_SEQ_LEN (currently {MODEL_SEQ_LEN})"
                )
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    return_audio=False, use_audio_in_video=False,
                )
            gen = out_ids[:, inputs["input_ids"].shape[1]:]
            txt = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] sample {i} ({video}) failed: {exc}")
            txt = ""
        pred = ans_letter(txt)
        if a.answer_format == "vocab":
            vocab = prompt_vocabulary(rec["conversations"][0]["value"])
            pred = _AM.match_answer(txt, vocab)
            gt = _AM.match_answer(rec["conversations"][1]["value"], vocab)
        good = pred == gt and gt != ""
        correct += int(good)
        preds.append({"index": str(i), "video": video, "gt": gt,
                       "pred": pred, "correct": str(good), "output": txt,
                       "audio_samples": int(len(wav))})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(recs)}] acc={100*correct/(i+1):.1f}% mode={a.audio_mode}")

    total = len(preds)  # processed count (== len(recs) unless POS_IDX subset)
    Path(a.output).write_text(json.dumps({
        "accuracy": 100.0 * correct / total if total else 0.0,
        "correct": correct, "total": total, "audio_mode": a.audio_mode,
        "answer_format": a.answer_format,
        "model_seq_len": MODEL_SEQ_LEN,
        "audio_seconds": AUDIO_SECONDS, "sample_rate": SR, "blank_video": _pos_blank,
        "seed": a.seed, "eval_data": a.eval_data, "adapter_sha256": adapter_sha256,
        "shift_seconds": a.shift_seconds if a.audio_mode == "shifted" else None,
        "visual_mode": a.visual_mode,
        "visual_severity": a.visual_severity if a.visual_mode != "clean" else 0.0,
        "model": MODEL_ID, "predictions": preds,
    }, indent=2))
    print(f"[done] {a.audio_mode}: {100*correct/total:.2f}% ({correct}/{total}) -> {a.output}")


if __name__ == "__main__":
    main()
