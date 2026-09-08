"""Aggregate CMSS surfaces from per-mode AVQA eval JSONs.

Thin I/O layer over :mod:`videollm.cmss`. Loads the ``real``/``silent`` eval runs at
each visual severity for one model and corruption family, pairs them item-by-item,
clusters by video, and reports the pre-registered estimands (``Delta_A(v)``, ``SG``,
``tau_switch``) with video-cluster bootstrap CIs.

Expected inputs follow the naming produced by ``scripts/eval_avqa.py``::

    eval_results/avqa_{model}_{audio_mode}.json                      # clean (severity 0)
    eval_results/avqa_{model}_{audio_mode}_vis-{family}-s{sev:g}.json  # degraded

Outputs::

    eval_results/cmss/{model}_{family}.json
    eval_results/cmss/{model}_{family}.md

Usage::

    python scripts/aggregate_cmss.py \
        --model full_v6 --family motion_blur --severities 0.33 0.66 1.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from videollm.cmss import BOOTSTRAP_SEED, N_BOOT, SWITCH_MARGIN_PP, compute_cmss

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = REPO_ROOT / "eval_results"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace with model, family, severities, and output settings.
    """
    parser = argparse.ArgumentParser(description="Aggregate a CMSS surface from eval JSONs")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint/model tag used in filenames")
    parser.add_argument("--family", type=str, required=True, help="Visual corruption family (e.g. motion_blur)")
    parser.add_argument(
        "--severities",
        type=float,
        nargs="+",
        default=[0.33, 0.66, 1.0],
        help="Degraded severities to include (clean 0.0 is always added)",
    )
    parser.add_argument("--audio_real", type=str, default="real", help="Audio mode used as the real condition")
    parser.add_argument("--audio_null", type=str, default="silent", help="Audio mode used as the null condition")
    parser.add_argument("--eval_dir", type=str, default=str(DEFAULT_EVAL_DIR), help="Directory with eval JSONs")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <eval_dir>/cmss)")
    parser.add_argument("--margin_pp", type=float, default=SWITCH_MARGIN_PP, help="tau_switch margin in pp")
    parser.add_argument("--n_boot", type=int, default=N_BOOT, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="Bootstrap seed")
    return parser.parse_args()


def eval_path(eval_dir: Path, model: str, audio_mode: str, family: str, severity: float) -> Path:
    """Return the eval-JSON path for one (audio mode, visual severity) cell."""
    if severity == 0.0:
        return eval_dir / f"avqa_{model}_{audio_mode}.json"
    return eval_dir / f"avqa_{model}_{audio_mode}_vis-{family}-s{severity:g}.json"


def load_items(path: Path) -> dict[str, tuple[bool, str]]:
    """Load one eval JSON into ``{item_index: (correct, video)}``.

    Args:
        path: Path to an ``eval_avqa.py`` result JSON.

    Returns:
        Mapping from item index to its correctness flag and source video.

    Raises:
        FileNotFoundError: If the eval JSON is missing.
        ValueError: If the payload has no predictions.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing eval JSON: {path}")
    payload = json.loads(path.read_text())
    predictions = payload.get("predictions", [])
    if not predictions:
        raise ValueError(f"no predictions in {path}")
    items = {str(p["index"]): (_as_bool(p["correct"], path), str(p.get("video", ""))) for p in predictions}

    # Schema drift would silently mark every item wrong; cross-check against the
    # accuracy the run reported for itself.
    reported = payload.get("accuracy")
    if reported is not None and items:
        recomputed = 100.0 * sum(ok for ok, _ in items.values()) / len(items)
        if abs(recomputed - float(reported)) > 0.5:
            raise ValueError(
                f"{path}: parsed accuracy {recomputed:.2f}% disagrees with reported "
                f"{float(reported):.2f}% -- correctness field may have changed schema"
            )
    return items


def _as_bool(value: object, path: Path) -> bool:
    """Interpret a correctness field that may be a bool, an int, or a string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{path}: uninterpretable correctness value {value!r}")


def unparsed_items(path: Path) -> set[str]:
    """Return indices whose prediction is empty in one eval JSON.

    An empty prediction means no answer letter was recovered — an undecodable video,
    a failed inference, or an unparseable generation. Every such item is stored as
    ``correct: False``, so it silently enters ``Delta_A`` as a zero-difference pair and
    pulls the estimate toward zero. Surfacing the count keeps a harness failure from
    being read as an absence of audio reliance.
    """
    payload = json.loads(path.read_text())
    return {str(p["index"]) for p in payload.get("predictions", []) if not str(p.get("pred", "")).strip()}


def assert_condition(path: Path, audio_mode: str, family: str, severity: float) -> None:
    """Verify a result file records the condition its filename claims.

    Both harnesses store the intervention they applied (nested under ``config`` for
    ``eval_avqa.py``, flat for the external adapters). Older runs predate those fields
    and are accepted with a warning rather than rejected.

    Raises:
        ValueError: If the recorded condition contradicts the requested one.
    """
    payload = json.loads(path.read_text())
    recorded = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    got_audio = recorded.get("audio_mode")
    got_visual = recorded.get("visual_mode")
    if got_audio is not None and got_audio != audio_mode:
        raise ValueError(f"{path}: recorded audio_mode={got_audio!r}, expected {audio_mode!r}")
    if got_visual is None:
        print(f"[warn] {path.name}: no visual_mode recorded (pre-CMSS run?); trusting filename")
        return
    want_visual = "clean" if severity == 0.0 else family
    if got_visual != want_visual:
        raise ValueError(f"{path}: recorded visual_mode={got_visual!r}, expected {want_visual!r}")
    got_sev = recorded.get("visual_severity")
    want_sev = 0.0 if severity == 0.0 else severity
    if got_sev is not None and abs(float(got_sev) - want_sev) > 1e-9:
        raise ValueError(f"{path}: recorded visual_severity={got_sev}, expected {want_sev}")


def build_grid(
    eval_dir: Path,
    model: str,
    family: str,
    severities: list[float],
    audio_real: str,
    audio_null: str,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], np.ndarray, list[str]]:
    """Load every cell and align it to the common item set.

    Returns:
        ``(per_severity, cluster_ids, item_keys)`` where ``per_severity`` maps each
        severity to paired ``(real, null)`` correctness vectors of shape ``(N,)``,
        ``cluster_ids`` holds the per-item video label, and ``item_keys`` are the
        item indices in the order used.

    Raises:
        ValueError: If the runs share no common items.
    """
    levels = sorted({0.0, *severities})
    cells: dict[tuple[float, str], dict[str, tuple[bool, str]]] = {}
    for sev in levels:
        for mode in (audio_real, audio_null):
            path = eval_path(eval_dir, model, mode, family, sev)
            # Trusting the filename alone once let a duplicate run overwrite a cell with
            # a different condition; verify what the run says it actually did.
            assert_condition(path, mode, family, sev)
            cells[(sev, mode)] = load_items(path)

    common: set[str] | None = None
    for items in cells.values():
        common = set(items) if common is None else common & set(items)
    if not common:
        raise ValueError("no common item indices across the requested eval runs")
    item_keys = sorted(common, key=lambda k: (len(k), k))

    # Videos come from the clean real run; every other cell must agree, otherwise the
    # cells are not the same items and the pairing is meaningless.
    reference = cells[(0.0, audio_real)]
    for (sev, mode), items in cells.items():
        mismatched = [k for k in item_keys if items[k][1] != reference[k][1]]
        if mismatched:
            raise ValueError(
                f"cell (severity={sev:g}, audio={mode}) disagrees with the clean baseline on "
                f"{len(mismatched)} item->video mappings (first: {mismatched[0]})"
            )
    cluster_ids = np.array([reference[k][1] for k in item_keys])  # (N,)

    per_severity: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for sev in levels:
        real_vec = np.array([cells[(sev, audio_real)][k][0] for k in item_keys], dtype=np.int64)  # (N,)
        null_vec = np.array([cells[(sev, audio_null)][k][0] for k in item_keys], dtype=np.int64)  # (N,)
        per_severity[sev] = (real_vec, null_vec)
    return per_severity, cluster_ids, item_keys


def render_markdown(model: str, family: str, result: object, n_items: int, n_clusters: int) -> str:
    """Render a paper-ready markdown summary of one CMSS curve."""
    res = result  # typed loosely to keep this layer I/O-only
    lines = [
        f"# CMSS — {model} / {family}",
        "",
        f"Items: {n_items} (video clusters: {n_clusters}). "
        f"Bootstrap: cluster-level, {N_BOOT} resamples, seed {BOOTSTRAP_SEED}.",
        "",
        "## Conditional audio contribution Δ_A(v)",
        "",
        "| Visual severity | Δ_A (pp) | 95% CI |",
        "|---|---:|---|",
    ]
    for d in res.deltas:  # type: ignore[attr-defined]
        lines.append(f"| {d.severity:g} | {d.delta_pp:+.1f} | [{d.ci95_lo:+.1f}, {d.ci95_hi:+.1f}] |")
    lines += [
        "",
        "## Pre-registered summary",
        "",
        f"- **SG** (substitution gain) = **{res.sg_pp:+.1f} pp** "  # type: ignore[attr-defined]
        f"[{res.sg_ci95_lo:+.1f}, {res.sg_ci95_hi:+.1f}]",  # type: ignore[attr-defined]
        f"- **τ_switch** (margin {res.margin_pp:g} pp) = "  # type: ignore[attr-defined]
        f"**{res.tau_switch if res.tau_switch is not None else 'none (flat curve)'}**",  # type: ignore[attr-defined]
        "",
        "Blank-video is excluded here by design; it is the delivery positive control.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Aggregate one CMSS surface and write JSON + markdown."""
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "cmss"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_severity, cluster_ids, item_keys = build_grid(
        eval_dir, args.model, args.family, list(args.severities), args.audio_real, args.audio_null
    )
    result = compute_cmss(per_severity, cluster_ids, margin_pp=args.margin_pp, n_boot=args.n_boot, seed=args.seed)

    # Surface unanswerable items (undecodable video, failed inference, unparseable
    # generation). They are stored as incorrect, so they enter every paired difference
    # as a zero and pull Delta_A toward zero — a harness failure must not be read as
    # an absence of audio reliance.
    unparsed: dict[str, int] = {}
    for sev in sorted(per_severity):
        for mode in (args.audio_real, args.audio_null):
            path = eval_path(eval_dir, args.model, mode, args.family, sev)
            bad = unparsed_items(path) & set(item_keys)
            if bad:
                unparsed[f"sev{sev:g}_{mode}"] = len(bad)
    if unparsed:
        print(f"[warn] cells with unparseable predictions: {unparsed}")

    n_clusters = int(np.unique(cluster_ids).size)
    payload = {
        "model": args.model,
        "family": args.family,
        "audio_real": args.audio_real,
        "audio_null": args.audio_null,
        "n_items": len(item_keys),
        "n_clusters": n_clusters,
        "unparsed_by_cell": unparsed,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "margin_pp": args.margin_pp,
        "deltas": [asdict(d) for d in result.deltas],
        "sg_pp": result.sg_pp,
        "sg_ci95_lo": result.sg_ci95_lo,
        "sg_ci95_hi": result.sg_ci95_hi,
        "sg_severity": result.sg_severity,
        "tau_switch": result.tau_switch,
    }
    json_path = out_dir / f"{args.model}_{args.family}.json"
    md_path = out_dir / f"{args.model}_{args.family}.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_markdown(args.model, args.family, result, len(item_keys), n_clusters))

    print(render_markdown(args.model, args.family, result, len(item_keys), n_clusters))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
