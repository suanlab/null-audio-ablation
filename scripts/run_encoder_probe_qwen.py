"""Linear instrument probe on Qwen2.5-Omni's own audio tower, on the clips the Whisper/BEATs probe used.

The manuscript pairs video-SALMONN 2+ with Qwen2.5-Omni as two systems on one encoder family. A
tensor comparison settles half of that: video-SALMONN 2+'s audio encoder is Whisper-large-v3
exactly (bit-identical after bf16 rounding), while Qwen2.5-Omni's tower was initialised from it
and then trained (global relative weight difference ~0.65). So the frozen-Whisper probe speaks for
video-SALMONN 2+ but not for Qwen. This script probes Qwen's tower the same way, so that the
"system, not encoder" reading can be tested rather than assumed:

* if Qwen's tower decodes instruments no better than Whisper, the delivery gap between the two
  systems is downstream of the encoder;
* if it decodes them much better, part of the gap is in the trained encoder weights.

Features are the final encoder layer's states (1280-d, before the tower's pooling, ln_post and
projection into the LLM), mean-pooled over time -- the same tap as the Whisper probe. Audio is
truncated to the first 30 s, which is what the Whisper feature extractor sees.

Usage::

    CUDA_VISIBLE_DEVICES=1 $VENV/bin/python scripts/run_encoder_probe_qwen.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
QWEN_ID = "Qwen/Qwen2.5-Omni-7B"
PREFIX = "thinker.audio_tower."
SR = 16000
MAX_SECONDS = 30


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Qwen2.5-Omni audio-tower instrument probe")
    p.add_argument("--limit", type=int, default=1319)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "eval_results" / "encoder_probe_instrument_qwen.json"))
    return p.parse_args()


def load_tower(device: str) -> torch.nn.Module:
    """Build the audio tower from the checkpoint's own config and weights, nothing else."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

    index = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/*/model.safetensors.index.json"))[0]
    root = os.path.dirname(index)
    cfg = AutoConfig.from_pretrained(QWEN_ID).thinker_config.audio_config
    tower = Qwen2_5OmniAudioEncoder._from_config(cfg, torch_dtype=torch.bfloat16)
    weight_map = json.loads(Path(index).read_text())["weight_map"]
    state: dict[str, torch.Tensor] = {}
    for shard in sorted({f for k, f in weight_map.items() if k.startswith(PREFIX)}):
        with safe_open(os.path.join(root, shard), "pt") as h:
            for k in h.keys():  # noqa: SIM118 -- safe_open handles are not dicts
                if k.startswith(PREFIX):
                    state[k[len(PREFIX):]] = h.get_tensor(k)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"audio tower weights do not match its config: missing={missing}, unexpected={unexpected}")
    return tower.to(device).eval()


def qwen_features(paths: list[str], device: str) -> np.ndarray:
    """Mean-pooled final-layer states of Qwen2.5-Omni's audio tower, one vector per clip."""
    from run_encoder_probe import load_audio
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(QWEN_ID)
    tower = load_tower(device)
    captured: list[torch.Tensor] = []
    hook = tower.layers[-1].register_forward_hook(lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out))
    out: list[np.ndarray] = []
    for i, path in enumerate(paths):
        wav = load_audio(path)[: MAX_SECONDS * SR]
        feats = extractor(wav, sampling_rate=SR, return_attention_mask=True, return_tensors="pt")
        mask = feats.attention_mask[0].bool()                          # (frames,)
        mel = feats.input_features[0][:, mask]                         # (n_mels, frames)
        feature_lens = mask.sum().view(1)                              # (1,)
        aftercnn_lens, _ = tower._get_feat_extract_output_lengths(feature_lens)
        captured.clear()
        with torch.no_grad():
            tower(mel.to(device, torch.bfloat16), feature_lens=feature_lens.to(device),
                  aftercnn_lens=aftercnn_lens.to(device))
        hidden = captured[-1]                                          # (tokens, 1280)
        out.append(hidden.float().mean(dim=0).cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"  qwen {i + 1}/{len(paths)}", flush=True)
    hook.remove()
    del tower
    torch.cuda.empty_cache()
    return np.stack(out)


def main() -> None:
    """Probe Qwen's tower on the exact clip set and labels of the Whisper/BEATs probe."""
    args = parse_args()
    from encoder_probe import instrument_labels
    from run_encoder_probe import probe

    items = instrument_labels(args.limit)
    paths = [p for p, _ in items]
    names = sorted({y for _, y in items})
    labels = np.array([names.index(y) for _, y in items])
    majority = float(np.bincount(labels).max() / len(labels) * 100)
    print(f"items={len(items)} classes={len(names)} majority={majority:.1f}%", flush=True)
    feats = qwen_features(paths, args.device)
    acc = probe(feats, labels, args.folds)
    result = {
        "items": len(items), "classes": len(names), "majority_baseline": majority,
        "task": "instrument identity (20-way)",
        "encoders": {"Qwen2.5-Omni audio tower": {**acc, "dim": int(feats.shape[1])}},
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"  Qwen2.5-Omni tower: {acc['mean']:.1f}% +/- {acc['std']:.1f}  (dim {feats.shape[1]})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
