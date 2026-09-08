"""B4 (2nd external model): five-mode audio protocol on Qwen2.5-Omni-7B.

Cross-family external validation. Same matched 98-sample AVQA subset, same
mode semantics, same deterministic shuffled mapping, same output schema as
scripts/eval_avqa.py and VideoLLaMA2/eval_avqa_5mode.py, so results drop into
scripts/aggregate_matched_98.py as a 5th model (prefix ``qwenomni``).

Audio-only intervention; video frames are always the real frames. We supply
our own controlled 16 kHz waveform and set use_audio_in_video=False so the
model consumes exactly the audio we pass.

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
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

SR = 16000
CLIP_SECONDS = 30
CHOICE_RE = re.compile(r"[A-D]")
AUDIO_MODES = ("real", "silent", "noise", "shuffled", "shifted")

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


def degrade_videos(vids, degradation):
    """Apply the CMSS visual intervention to qwen_omni_utils video tensors.

    Placed at the same hook point as the POS_BLANK control: on the decoded video
    tensors, before the processor, so downstream grid/patch bookkeeping is unchanged.
    A clean or zero-severity config returns the input untouched.

    Args:
        vids: List of video tensors ``(T, C, H, W)``, or None.
        degradation: The intervention to apply.

    Returns:
        Degraded tensors with the input dtype and shape preserved.

    Raises:
        ValueError: If a float tensor falls outside ``[0, 1]``; the degradation
            contract clamps to that range, so a normalised tensor would be corrupted.
    """
    if vids is None or degradation is None or degradation.mode == "clean" or degradation.severity == 0.0:
        return vids
    out = []
    for v in vids:
        if v.dtype == torch.uint8:
            frames = v.float() / 255.0  # (T, C, H, W)
        else:
            frames = v.float()  # (T, C, H, W)
            if float(frames.min()) < 0.0 or float(frames.max()) > 1.0:
                raise ValueError(
                    f"expected uint8 or float in [0, 1], got float range "
                    f"[{float(frames.min()):.3f}, {float(frames.max()):.3f}]"
                )
        degraded = _VD.apply_visual_degradation(frames, degradation)  # (T, C, H, W)
        if v.dtype == torch.uint8:
            out.append((degraded * 255.0).round().clamp(0, 255).to(torch.uint8))
        else:
            out.append(degraded.to(v.dtype))
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


def fit_len(w: np.ndarray) -> np.ndarray:
    t = CLIP_SECONDS * SR
    if len(w) > t:
        return w[:t]
    if len(w) < t:
        return np.pad(w, (0, t - len(w)))
    return w


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
        return np.zeros(CLIP_SECONDS * SR, dtype=np.float32)
    if mode == "noise":
        return np.random.randn(CLIP_SECONDS * SR).astype(np.float32)
    if mode == "shuffled":
        return fit_len(load_wav16(donor or vpath))
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

    shuf = build_shuffled_mapping(len(recs), a.seed)
    preds, correct = [], 0
    for i, rec in enumerate(recs):
        if _pos_idx is not None and i not in _pos_idx:
            continue
        video = str(rec["video"])
        vpath = str(vdir / video)
        gt = gt_letter(rec["conversations"][1]["value"])
        q = rec["conversations"][0]["value"].replace("<video>", "").replace("<audio>", "").strip()
        try:
            donor = str(vdir / str(recs[shuf[i]]["video"])) if a.audio_mode == "shuffled" else None
            wav = mode_wav(a.audio_mode, vpath, donor, a.shift_seconds)
            conv = [
                {"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [
                    {"type": "video", "video": vpath},
                    {"type": "text", "text": q},
                ]},
            ]
            chat = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            from qwen_omni_utils import process_mm_info

            _au, imgs, vids = process_mm_info(conv, use_audio_in_video=False)
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
        good = pred == gt and gt != ""
        correct += int(good)
        preds.append({"index": str(i), "video": video, "gt": gt,
                       "pred": pred, "correct": str(good), "output": txt})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(recs)}] acc={100*correct/(i+1):.1f}% mode={a.audio_mode}")

    total = len(preds)  # processed count (== len(recs) unless POS_IDX subset)
    Path(a.output).write_text(json.dumps({
        "accuracy": 100.0 * correct / total if total else 0.0,
        "correct": correct, "total": total, "audio_mode": a.audio_mode,
        "shift_seconds": a.shift_seconds if a.audio_mode == "shifted" else None,
        "visual_mode": a.visual_mode,
        "visual_severity": a.visual_severity if a.visual_mode != "clean" else 0.0,
        "model": MODEL_ID, "predictions": preds,
    }, indent=2))
    print(f"[done] {a.audio_mode}: {100*correct/total:.2f}% ({correct}/{total}) -> {a.output}")


if __name__ == "__main__":
    main()
