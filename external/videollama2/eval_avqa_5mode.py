"""B4: Five-mode audio intervention protocol applied to VideoLLaMA2.1-7B-AV.

External-model validation of the protocol. Uses the SAME matched 98-sample
AVQA subset and the SAME mode semantics as our in-house VideoLLM evaluation
(scripts/eval_avqa.py), so results slot directly into
scripts/aggregate_matched_98.py.

Mode semantics (audio only; video frames are always the real frames):
    real     : original audio
    silent   : zero waveform
    noise    : gaussian noise waveform
    shuffled : audio from a deterministically chosen different eval video
    shifted  : same audio, circularly shifted by `--shift_seconds`

Output schema matches eval_avqa.py exactly:
    {"accuracy", "correct", "total", "predictions":[{index,video,gt,pred,correct,output}]}

Usage (run inside the `videollama2` conda env):
    CUDA_VISIBLE_DEVICES=2 python eval_avqa_5mode.py \
        --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
        --eval_data $REPO_ROOT/data/instruct/avqa_test_clean.jsonl \
        --video_dir $REPO_ROOT/data/videos \
        --audio_mode real --output out_real.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
import os


def _env(name: str) -> str:
    """A path variable from PATHS.md; fail naming the variable rather than with a bad path."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set ${name} first -- see PATHS.md")
    return value

import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi

from videollama2 import mm_infer, model_init
from videollama2.constants import NUM_FRAMES
from videollama2.mm_utils import frame_sample

# CMSS visual axis. Imported by path because this adapter lives outside the VideoLLM
# repo and its package is not installed in this env.
_VD_PATH = Path(_env("REPO_ROOT")) / "src" / "videollm" / "data" / "video_degradation.py"
_VD_SPEC = importlib.util.spec_from_file_location("video_degradation", _VD_PATH)
assert _VD_SPEC is not None and _VD_SPEC.loader is not None
_VD = importlib.util.module_from_spec(_VD_SPEC)
# Register before exec: @dataclass resolves its module through sys.modules.
sys.modules["video_degradation"] = _VD
_VD_SPEC.loader.exec_module(_VD)
VISUAL_MODES = _VD.VISUAL_MODES


def degraded_frames(video_path, degradation):
    """Decode the frames this model would have used, degraded, as uint8 (T, H, W, C).

    Mirrors `process_video`'s own sampling (uniform, NUM_FRAMES) so the intervention
    replaces exactly the frames the model would otherwise see; the returned array is fed
    back through the model's own preprocessing, leaving that path untouched.
    """
    import torch as _t
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    sampled = frame_sample(len(reader), mode="uniform", num_frames=NUM_FRAMES)
    raw = reader.get_batch(sampled).asnumpy()  # (T, H, W, C) uint8
    frames = _t.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    out = _VD.apply_visual_degradation(frames, degradation)  # (T, C, H, W)
    return (out * 255.0).round().clamp(0, 255).to(_t.uint8).permute(0, 2, 3, 1).numpy()
from videollama2.mm_utils import load_audio_from_video

SR = 16000
CLIP_SECONDS = 30
CHOICE_RE = re.compile(r"[A-D]")

# --- closed-vocabulary answers (MUSIC-AVQA) --------------------------------------
# Loaded by path so both adapters read model output with byte-identical logic; a
# divergence here would show up as a between-model difference that is really a
# between-parser difference.
_AM_PATH = Path(_env("REPO_ROOT")) / "src" / "videollm" / "data" / "answer_match.py"
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


def waveform_to_fbank_full(wav: np.ndarray) -> torch.Tensor:
    """Filterbank over the WHOLE track, with no 30 s truncation.

    BEATs carries convolutional and gated-relative position encodings and no length cap,
    and the audio embedding is concatenated after the video features rather than inserted
    at a fixed number of placeholders, so a longer fbank simply yields more audio tokens.
    The 30 s limit elsewhere in this file is a pipeline choice, not an encoder constraint
    -- which is what this option tests. The BEATs-fronted model otherwise receives ~27% of
    a clip while the Whisper-fronted comparison model receives all of it, and that
    asymmetry runs in the same direction as the effect it is used to measure.
    """
    t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    floor = CLIP_SECONDS * SR
    if t.numel() < floor:  # keep very short clips at the length the model expects
        t = torch.nn.functional.pad(t, (0, floor - t.numel()), value=0.0)
    waveform = t.unsqueeze(0) * 2 ** 15
    fbank = ta_kaldi.fbank(
        waveform, num_mel_bins=128, sample_frequency=SR, frame_length=25, frame_shift=10
    ).to(torch.bfloat16)
    return fbank.unsqueeze(0)


def waveform_to_fbank(wav: np.ndarray) -> torch.Tensor:
    """Replicate VideoLLaMA2's BEATs fbank tail on a 16kHz mono waveform.

    Mirrors process_audio_from_video's final block so that real / shifted /
    shuffled / synthetic modes all pass through an identical fbank path.
    """
    t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    target = CLIP_SECONDS * SR
    if t.numel() > target:
        t = t[:target]
    elif t.numel() < target:
        t = torch.nn.functional.pad(t, (0, target - t.numel()), value=0.0)
    waveform = t.unsqueeze(0) * 2 ** 15
    fbank = ta_kaldi.fbank(
        waveform, num_mel_bins=128, sample_frequency=SR, frame_length=25, frame_shift=10
    ).to(torch.bfloat16)
    return fbank.unsqueeze(0)


def _safe_load_wav(video_path: str) -> np.ndarray:
    try:
        wav, sr = load_audio_from_video(video_path)
        wav = np.asarray(wav, dtype=np.float32)
        if sr != SR:  # load_audio_from_video already resamples to 16k, defensive only
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR, res_type="fft")
        return wav
    except Exception as exc:  # noqa: BLE001 - eval robustness, mirror upstream
        print(f"[warn] audio load failed for {video_path}: {exc}")
        return np.zeros(CLIP_SECONDS * SR, dtype=np.float32)


def build_shuffled_mapping(n: int, seed: int) -> list[int]:
    """Deterministic derangement (no sample maps to itself); matches eval_avqa.py."""
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(100):
        perm = idx[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    return [(i + 1) % n for i in range(n)]


def get_mode_fbank(
    mode: str,
    video_path: str,
    donor_path: str | None,
    shift_seconds: float,
) -> torch.Tensor:
    if mode == "silent":
        return waveform_to_fbank(np.zeros(CLIP_SECONDS * SR, dtype=np.float32))
    if mode == "noise":
        return waveform_to_fbank(np.random.randn(CLIP_SECONDS * SR).astype(np.float32))
    if mode == "shuffled":
        return waveform_to_fbank(_safe_load_wav(donor_path or video_path))
    if mode == "shifted":
        wav = _safe_load_wav(video_path)
        return waveform_to_fbank(np.roll(wav, int(shift_seconds * SR)))
    raise ValueError(f"get_mode_fbank should not be called for mode={mode!r}")


def extract_answer_letter(text: str) -> str:
    m = CHOICE_RE.search(text.upper())
    return m.group(0) if m else ""


def extract_ground_truth(gpt_response: str) -> str:
    m = re.search(r"answer is ([A-D])", gpt_response, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VideoLLaMA2.1-AV five-mode AVQA eval")
    p.add_argument("--model_path", default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
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
        "--audio_access",
        default="sampled",
        choices=("sampled", "contiguous", "full"),
        help=(
            "How the REAL condition receives audio. 'sampled' (default) keeps upstream's "
            "ConstantClipsPerVideoSampler(2 s x 8): eight 2-second windows spliced end to "
            "end and zero-padded to 30 s, i.e. 16 s of audio with the time axis broken at "
            "every splice, regardless of clip length. 'contiguous' instead feeds the first "
            "30 s of the track unbroken, through the same fbank tail the null conditions "
            "use. On clips under ~18 s the two are equivalent (the windows overlap and "
            "cover everything); on longer clips 'sampled' covers a fraction -- 28.5%% median "
            "on the AVUT pilot -- which confounds any cross-domain comparison of audio "
            "reliance with clip length. 'full' feeds the entire track, matching the "
            "coverage the Whisper-fronted comparison model already receives."
        ),
    )
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
    args = parse_args()
    # Deterministic seeding (CHECK.md B.6) for reproducible noise mode.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    records = [json.loads(line) for line in Path(args.eval_data).read_text().splitlines()]
    video_dir = Path(args.video_dir)

    model, processor, tokenizer = model_init(args.model_path)
    # model_init returns processor as a dict {'image','video','audio'};
    # the AV path uses the 'video' processor with va=True.
    video_processor = processor["video"]

    shuffled_map = build_shuffled_mapping(len(records), args.seed)

    import os as _os

    _pos_idx_path = _os.environ.get("POS_IDX")
    _pos_idx = set(json.loads(open(_pos_idx_path).read())) if _pos_idx_path else None
    _pos_blank = _os.environ.get("POS_BLANK") == "1"

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
            av = video_processor(vpath, va=True)  # {'video':..., 'audio':...} (real)
            # Visual axis: replace the video tensor with one built from degraded frames,
            # keeping the audio produced by the call above. Per-item deterministic seed.
            if args.visual_mode != "clean" and args.visual_severity > 0.0:
                cfg = _VD.DegradationConfig(
                    mode=args.visual_mode, severity=args.visual_severity, seed=args.seed + i
                )
                av["video"] = video_processor(degraded_frames(vpath, cfg), va=False)
            if _pos_blank:
                import torch as _t

                av["video"] = _t.zeros_like(av["video"])
            if args.audio_mode == "real" and args.audio_access == "full":
                av["audio"] = waveform_to_fbank_full(_safe_load_wav(vpath))
            if args.audio_mode == "real" and args.audio_access == "contiguous":
                # Same fbank tail as the null conditions, so real vs. silent differ only in
                # the waveform and not in how it was assembled.
                av["audio"] = waveform_to_fbank(_safe_load_wav(vpath))
            if args.audio_mode != "real":
                donor = None
                if args.audio_mode == "shuffled":
                    donor = str(video_dir / str(records[shuffled_map[i]]["video"]))
                av["audio"] = get_mode_fbank(
                    args.audio_mode, vpath, donor, args.shift_seconds
                )
            out = mm_infer(
                av, question, model=model, tokenizer=tokenizer,
                modal="video", do_sample=False,
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
        preds.append({
            "index": str(i), "video": video, "gt": gt, "pred": pred,
            "correct": str(is_ok), "output": out,
        })
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(records)}] acc={100*correct/(i+1):.1f}% mode={args.audio_mode}")

    total = len(preds)  # processed count (== len(records) unless POS_IDX subset)
    result = {
        "accuracy": 100.0 * correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "audio_mode": args.audio_mode,
        "answer_format": args.answer_format,
        "audio_access": args.audio_access,
        "shift_seconds": args.shift_seconds if args.audio_mode == "shifted" else None,
        "visual_mode": args.visual_mode,
        "visual_severity": args.visual_severity if args.visual_mode != "clean" else 0.0,
        "model": args.model_path,
        "predictions": preds,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"[done] {args.audio_mode}: {result['accuracy']:.2f}% ({correct}/{total}) -> {args.output}")


if __name__ == "__main__":
    main()
