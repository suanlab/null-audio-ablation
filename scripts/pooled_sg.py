"""Compute the preregistered **pooled-visual** substitution gain for one model/domain.

`preregistration.md` §4 fixes the primary confirmatory family as "`SG > 0` test per
(model × domain), audio null = `silent`, **family = pooled visual**". Per-family `SG`
spans 8x within a single model, so reporting a per-family maximum instead of the pooled
basis is a live cherry-picking risk -- this script exists so the pooled number is
reproducible from the cell JSONs rather than recomputed ad hoc each time.

Pooling is at the item level: for each corruption family the severity-1.0 cell supplies
one paired (real, silent) outcome per item, and the four families' paired differences are
concatenated before the bootstrap. Resampling stays clustered on the video, so the four
appearances of a video's items move together and the correlation induced by pooling is
carried into the interval rather than ignored.

    SG_pooled = mean_pooled(Delta_A at severity 1.0) - Delta_A(clean)

Usage::

    python scripts/pooled_sg.py --model cmss_vsalm2
    python scripts/pooled_sg.py --model cmss_avut_vsalm2 --families motion_blur occlusion frame_drop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from videollm.cmss import cluster_bootstrap_mean

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAMILIES = ("motion_blur", "occlusion", "frame_drop", "downscale")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Preregistered pooled-visual SG")
    parser.add_argument("--model", required=True, help="Model tag used in the cell filenames")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--severity_tag", default="s1", help="Severe-endpoint filename tag")
    parser.add_argument("--eval_dir", default=str(REPO_ROOT / "eval_results"))
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def load_pair(eval_dir: Path, model: str, suffix: str) -> tuple[np.ndarray, list[str]]:
    """Return the paired real−silent correctness difference (pp) and each item's video."""
    out: list[np.ndarray] = []
    videos: list[str] | None = None
    for audio in ("real", "silent"):
        path = eval_dir / f"avqa_{model}_{audio}{suffix}.json"
        payload = json.loads(path.read_text())
        preds = payload["predictions"]
        out.append(np.array([1.0 if str(p["correct"]) == "True" else 0.0 for p in preds]))
        vids = [str(p["video"]) for p in preds]
        if videos is None:
            videos = vids
        elif videos != vids:
            raise ValueError(f"item order differs between real and silent for {suffix or 'clean'}")
    if videos is None:
        raise ValueError("no cells loaded")
    return (out[0] - out[1]) * 100.0, videos


def main() -> None:
    """Report Delta_A(clean), pooled Delta_A(severe) and their difference."""
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    clean_diff, clean_videos = load_pair(eval_dir, args.model, "")
    severe_diffs: list[np.ndarray] = []
    severe_videos: list[str] = []
    for family in args.families:
        diff, videos = load_pair(eval_dir, args.model, f"_vis-{family}-{args.severity_tag}")
        if videos != clean_videos:
            raise ValueError(f"{family}: item order differs from the clean cell")
        severe_diffs.append(diff)
        severe_videos.extend(videos)

    pooled = np.concatenate(severe_diffs)  # (n_items * n_families,)
    index = {v: i for i, v in enumerate(sorted(set(clean_videos)))}
    clean_cid = np.array([index[v] for v in clean_videos])
    pooled_cid = np.array([index[v] for v in severe_videos])

    c_m, c_lo, c_hi = cluster_bootstrap_mean(clean_diff, clean_cid, n_boot=args.n_boot, seed=args.seed)
    s_m, s_lo, s_hi = cluster_bootstrap_mean(pooled, pooled_cid, n_boot=args.n_boot, seed=args.seed)

    # SG is bootstrapped as a single paired statistic so its interval reflects the
    # correlation between the two endpoints instead of differencing two independent CIs.
    rng_clean = clean_diff
    per_family = np.stack(severe_diffs)  # (n_families, n_items)
    sg_item = per_family.mean(axis=0) - rng_clean  # (n_items,)
    g_m, g_lo, g_hi = cluster_bootstrap_mean(sg_item, clean_cid, n_boot=args.n_boot, seed=args.seed)

    print(f"model={args.model}  families={' '.join(args.families)}  n_items={len(clean_diff)}")
    print(f"  Delta_A(clean)          = {c_m:+6.2f} pp  [{c_lo:+6.2f}, {c_hi:+6.2f}]")
    print(f"  Delta_A(severe, pooled) = {s_m:+6.2f} pp  [{s_lo:+6.2f}, {s_hi:+6.2f}]")
    print(f"  SG (pooled visual)      = {g_m:+6.2f} pp  [{g_lo:+6.2f}, {g_hi:+6.2f}]  ", end="")
    print("excludes 0" if g_lo > 0 or g_hi < 0 else "includes 0")


if __name__ == "__main__":
    main()
