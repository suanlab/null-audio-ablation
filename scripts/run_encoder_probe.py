"""Extract frozen Whisper and BEATs features and fit linear instrument probes.

Settles by measurement a claim the manuscript currently asserts: that a Whisper front end
cannot support instrument identification while a BEATs front end can. The assertion is
inferred from downstream delivery gains, i.e. through a projector, an adapter and an LLM.
Prior work (Whisper-AT) reports that Whisper's encoder carries audio-event information, so
the honest possibilities differ sharply -- either the information is absent, or it is
present and the deployed tap does not carry it -- and only a probe on the frozen features
separates them.

Both encoders see identical audio from identical clips. Features are mean-pooled over time
and a multinomial logistic regression is fitted with stratified cross-validation, so the
number reported is what is *linearly decodable*: the adapters downstream are shallow, so
information that is not linearly available is unlikely to survive them.

Usage::

    python scripts/run_encoder_probe.py --limit 1319
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
BEATS_REPO = Path("$VIDEOLLAMA2_DIR")
BEATS_CKPT = Path(
    "$VIDEOSALMONN_V1_DIR/ckpt/pretrained_ckpt/"
    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
WHISPER_ID = "openai/whisper-large-v3"
SR = 16000
SEED = 2027


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Frozen-encoder instrument probe")
    p.add_argument("--limit", type=int, default=1319)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "eval_results" / "encoder_probe_instrument.json"))
    return p.parse_args()


def load_audio(path: str) -> np.ndarray:
    """Decode a clip's audio to 16 kHz mono float32."""
    import librosa

    wav, _ = librosa.load(path, sr=SR, mono=True)
    return np.asarray(wav, dtype=np.float32)


def whisper_features(paths: list[str], device: str) -> np.ndarray:
    """Mean-pooled final-layer Whisper encoder states, one vector per clip."""
    from transformers import WhisperFeatureExtractor, WhisperModel

    extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    model = WhisperModel.from_pretrained(WHISPER_ID, torch_dtype=torch.float16).encoder
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    for i, path in enumerate(paths):
        feats = extractor(load_audio(path), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            hidden = model(feats.input_features.to(device, torch.float16)).last_hidden_state
        out.append(hidden.mean(dim=1).squeeze(0).float().cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"  whisper {i + 1}/{len(paths)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.stack(out)


def beats_features(paths: list[str], device: str) -> np.ndarray:
    """Mean-pooled BEATs encoder states, one vector per clip, same audio as Whisper."""
    # Import BEATs as a standalone package: videollama2/__init__ pulls in a projector that
    # imports TRANSFORMERS_CACHE, removed in transformers 5.x, and the encoder needs none
    # of that chain.
    sys.path.insert(0, str(BEATS_REPO / "videollama2" / "model"))
    import torchaudio.compliance.kaldi as ta_kaldi
    from beats.BEATs import BEATs, BEATsConfig

    ckpt = torch.load(BEATS_CKPT, map_location="cpu")
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    for i, path in enumerate(paths):
        wav = torch.from_numpy(load_audio(path)).unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(wav, num_mel_bins=128, sample_frequency=SR, frame_length=25, frame_shift=10)
        with torch.no_grad():
            emb, _, _ = model.extract_features(fbank.unsqueeze(0).to(device), padding_mask=None, feature_only=True)
        out.append(emb.mean(dim=1).squeeze(0).float().cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"  beats {i + 1}/{len(paths)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.stack(out)


def probe(features: np.ndarray, labels: np.ndarray, folds: int) -> dict[str, float]:
    """Stratified cross-validated multinomial logistic regression accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, features, labels, cv=cv, n_jobs=folds)
    return {"mean": float(scores.mean() * 100), "std": float(scores.std() * 100)}


def main() -> None:
    """Extract both encoders' features on identical clips and report probe accuracy."""
    args = parse_args()
    from encoder_probe import instrument_labels

    items = instrument_labels(args.limit)
    paths = [p for p, _ in items]
    names = sorted({y for _, y in items})
    labels = np.array([names.index(y) for _, y in items])
    majority = float(np.bincount(labels).max() / len(labels) * 100)
    print(f"items={len(items)} classes={len(names)} majority={majority:.1f}%", flush=True)

    results: dict[str, object] = {
        "items": len(items), "classes": len(names), "majority_baseline": majority,
        "task": "instrument identity (20-way)", "encoders": {},
    }
    for name, fn in (("BEATs", beats_features), ("Whisper", whisper_features)):
        print(f"extracting {name} ...", flush=True)
        feats = fn(paths, args.device)
        acc = probe(feats, labels, args.folds)
        results["encoders"][name] = {**acc, "dim": int(feats.shape[1])}  # type: ignore[index]
        print(f"  {name}: {acc['mean']:.1f}% +/- {acc['std']:.1f}  (dim {feats.shape[1]})", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
