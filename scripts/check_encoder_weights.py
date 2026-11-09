"""Compare the deployed audio encoders against Whisper-large-v3, tensor by tensor.

The manuscript calls video-SALMONN 2+ and Qwen2.5-Omni two systems on one encoder family and
reads a frozen Whisper-large-v3 probe as evidence about video-SALMONN 2+'s encoder. Both claims
are about weights, so they are checked on the weights:

* video-SALMONN 2+ stores its encoder in bf16 and Whisper-large-v3 ships in fp16, so equality is
  tested after casting Whisper to bf16 -- bit-identical there means the encoder was not trained;
* Qwen2.5-Omni's tower is reported as the global relative Frobenius difference to Whisper over
  the tensors the two share.

Usage::

    $CONDA_ENVS/qwenomni/bin/python scripts/check_encoder_weights.py
"""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results", "encoder_weight_check.json")

HUB = os.path.expanduser("~/.cache/huggingface/hub")
WHISPER = glob.glob(f"{HUB}/models--openai--whisper-large-v3/snapshots/*/model.safetensors")[0]
VSALM = "$MODELS_DIR/video-SALMONN2_plus_7B_full"
QWEN = os.path.dirname(glob.glob(f"{HUB}/models--Qwen--Qwen2.5-Omni-7B/snapshots/*/model.safetensors.index.json")[0])


def load_prefix(root: str, prefix: str) -> dict[str, torch.Tensor]:
    """Every tensor under ``prefix`` in a sharded checkpoint, keyed without the prefix."""
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out: dict[str, torch.Tensor] = {}
    for shard in sorted({f for k, f in weight_map.items() if k.startswith(prefix)}):
        with safe_open(os.path.join(root, shard), "pt") as h:
            for k in h.keys():  # noqa: SIM118 -- safe_open handles are not dicts
                if k.startswith(prefix):
                    out[k[len(prefix):]] = h.get_tensor(k)
    return out


def main() -> None:
    """Report bf16 identity for video-SALMONN 2+ and relative divergence for Qwen2.5-Omni."""
    with safe_open(WHISPER, "pt") as h:
        whisper = {k[len("model.encoder."):]: h.get_tensor(k) for k in h.keys() if k.startswith("model.encoder.")}  # noqa: SIM118
    report: dict[str, dict[str, float]] = {}
    for name, enc in (("video-SALMONN 2+", load_prefix(VSALM, "audio.")),
                      ("Qwen2.5-Omni", load_prefix(QWEN, "thinker.audio_tower."))):
        shared = [k for k in whisper if k in enc and enc[k].shape == whisper[k].shape]
        identical = sum(torch.equal(whisper[k].to(torch.bfloat16), enc[k].to(torch.bfloat16)) for k in shared)
        num = sum(float((enc[k].float() - whisper[k].float()).norm() ** 2) for k in shared)
        den = sum(float(whisper[k].float().norm() ** 2) for k in shared)
        print(f"{name}: {len(shared)}/{len(whisper)} Whisper encoder tensors shared, "
              f"{identical} bit-identical in bf16, global relative difference {(num / den) ** 0.5:.4f}")
        report[name] = {"shared": len(shared), "whisper_tensors": len(whisper),
                        "bf16_identical": identical, "relative_difference": (num / den) ** 0.5}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
