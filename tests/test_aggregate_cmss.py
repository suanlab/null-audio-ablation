"""Tests for the CMSS aggregation I/O layer (scripts/aggregate_cmss.py).

Covers eval-JSON naming, item pairing across conditions, and video clustering —
the parts that silently corrupt a surface if they drift from ``eval_avqa.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "aggregate_cmss.py"

N_ITEMS, PER_VIDEO = 40, 4


def _load_script() -> ModuleType:
    """Import scripts/aggregate_cmss.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("aggregate_cmss", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_cmss"] = module
    spec.loader.exec_module(module)
    return module


agg = _load_script()


def _write_eval(path: Path, correct_for: object) -> None:
    """Write a minimal eval_avqa.py-shaped result JSON."""
    predictions = [
        {
            "index": str(i),
            "video": f"vid{i // PER_VIDEO}.mp4",
            "gt": "A",
            "pred": "A",
            "correct": str(bool(correct_for(i))),  # type: ignore[operator]
            "output": "",
        }
        for i in range(N_ITEMS)
    ]
    n_correct = sum(1 for p in predictions if p["correct"] == "True")
    # Report a truthful accuracy: the loader cross-checks it to catch schema drift.
    path.write_text(
        json.dumps(
            {
                "accuracy": 100.0 * n_correct / N_ITEMS,
                "correct": n_correct,
                "total": N_ITEMS,
                "predictions": predictions,
            }
        )
    )


def _make_late_fallback_fixtures(eval_dir: Path, model: str = "smoke", family: str = "motion_blur") -> None:
    """Audio is irrelevant until the severe level, where it flips half the items."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    always = lambda i: True  # noqa: E731
    _write_eval(eval_dir / f"avqa_{model}_real.json", always)
    _write_eval(eval_dir / f"avqa_{model}_silent.json", always)
    for sev in ("0.33", "0.66"):
        _write_eval(eval_dir / f"avqa_{model}_real_vis-{family}-s{sev}.json", always)
        _write_eval(eval_dir / f"avqa_{model}_silent_vis-{family}-s{sev}.json", always)
    _write_eval(eval_dir / f"avqa_{model}_real_vis-{family}-s1.json", always)
    _write_eval(eval_dir / f"avqa_{model}_silent_vis-{family}-s1.json", lambda i: i >= N_ITEMS // 2)


# --- path naming -------------------------------------------------------------


def test_clean_path_has_no_visual_suffix(tmp_path: Path) -> None:
    p = agg.eval_path(tmp_path, "full_v6", "real", "motion_blur", 0.0)
    assert p.name == "avqa_full_v6_real.json"


def test_degraded_path_matches_eval_avqa_naming(tmp_path: Path) -> None:
    p = agg.eval_path(tmp_path, "full_v6", "silent", "occlusion", 0.33)
    assert p.name == "avqa_full_v6_silent_vis-occlusion-s0.33.json"
    # severity formatting must use %g so 1.0 -> "s1", matching eval_avqa.py
    assert agg.eval_path(tmp_path, "m", "real", "downscale", 1.0).name == "avqa_m_real_vis-downscale-s1.json"


# --- loading -----------------------------------------------------------------


def test_load_items_parses_correctness_and_video(tmp_path: Path) -> None:
    path = tmp_path / "avqa_m_real.json"
    _write_eval(path, lambda i: i % 2 == 0)
    items = agg.load_items(path)
    assert len(items) == N_ITEMS
    assert items["0"] == (True, "vid0.mp4")
    assert items["1"] == (False, "vid0.mp4")


def test_load_items_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        agg.load_items(tmp_path / "nope.json")


def test_load_items_empty_predictions_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"predictions": []}))
    with pytest.raises(ValueError):
        agg.load_items(path)


# --- grid assembly -----------------------------------------------------------


def test_build_grid_aligns_levels_and_clusters(tmp_path: Path) -> None:
    _make_late_fallback_fixtures(tmp_path)
    per_severity, cluster_ids, item_keys = agg.build_grid(
        tmp_path, "smoke", "motion_blur", [0.33, 0.66, 1.0], "real", "silent"
    )
    assert sorted(per_severity) == [0.0, 0.33, 0.66, 1.0]  # clean auto-added
    assert len(item_keys) == N_ITEMS
    assert cluster_ids.shape == (N_ITEMS,)
    assert np.unique(cluster_ids).size == N_ITEMS // PER_VIDEO
    for real_vec, null_vec in per_severity.values():
        assert real_vec.shape == null_vec.shape == (N_ITEMS,)


def test_build_grid_intersects_items_across_runs(tmp_path: Path) -> None:
    """A short run (e.g. a resumed eval) must shrink the grid, not misalign it."""
    _make_late_fallback_fixtures(tmp_path)
    short = json.loads((tmp_path / "avqa_smoke_silent_vis-motion_blur-s1.json").read_text())
    short["predictions"] = short["predictions"][:12]
    # A real truncated/resumed run reports accuracy over the rows it actually wrote.
    n_ok = sum(1 for p in short["predictions"] if p["correct"] == "True")
    short["accuracy"] = 100.0 * n_ok / len(short["predictions"])
    short["correct"], short["total"] = n_ok, len(short["predictions"])
    (tmp_path / "avqa_smoke_silent_vis-motion_blur-s1.json").write_text(json.dumps(short))

    _, cluster_ids, item_keys = agg.build_grid(tmp_path, "smoke", "motion_blur", [0.33, 0.66, 1.0], "real", "silent")
    assert len(item_keys) == 12
    assert cluster_ids.shape == (12,)


def test_build_grid_without_common_items_raises(tmp_path: Path) -> None:
    _make_late_fallback_fixtures(tmp_path)
    disjoint = json.loads((tmp_path / "avqa_smoke_real.json").read_text())
    for p in disjoint["predictions"]:
        p["index"] = f"x{p['index']}"
    (tmp_path / "avqa_smoke_real.json").write_text(json.dumps(disjoint))
    with pytest.raises(ValueError):
        agg.build_grid(tmp_path, "smoke", "motion_blur", [1.0], "real", "silent")


# --- end-to-end --------------------------------------------------------------


def test_end_to_end_recovers_known_late_fallback(tmp_path: Path) -> None:
    from videollm.cmss import compute_cmss

    _make_late_fallback_fixtures(tmp_path)
    per_severity, cluster_ids, _ = agg.build_grid(tmp_path, "smoke", "motion_blur", [0.33, 0.66, 1.0], "real", "silent")
    res = compute_cmss(per_severity, cluster_ids, n_boot=200)
    assert res.deltas[0].delta_pp == pytest.approx(0.0)  # clean: audio unused
    assert res.deltas[-1].delta_pp == pytest.approx(50.0)  # severe: audio carries half
    assert res.sg_pp == pytest.approx(50.0)
    assert res.tau_switch == 1.0


def test_render_markdown_contains_key_estimands(tmp_path: Path) -> None:
    from videollm.cmss import compute_cmss

    _make_late_fallback_fixtures(tmp_path)
    per_severity, cluster_ids, item_keys = agg.build_grid(tmp_path, "smoke", "motion_blur", [1.0], "real", "silent")
    res = compute_cmss(per_severity, cluster_ids, n_boot=200)
    md = agg.render_markdown("smoke", "motion_blur", res, len(item_keys), 10)
    assert "SG" in md and "τ_switch" in md and "Δ_A" in md
    assert "positive control" in md


# --- unparseable-prediction diagnostic ---------------------------------------


def test_unparsed_items_flags_empty_predictions(tmp_path: Path) -> None:
    """Items with no recovered answer letter must be surfaced, not silently counted as
    wrong — they enter every paired difference as a zero and shrink Delta_A."""
    path = tmp_path / "avqa_m_real.json"
    _write_eval(path, lambda i: False)
    payload = json.loads(path.read_text())
    payload["predictions"][3]["pred"] = ""  # decode/inference failure
    payload["predictions"][7]["pred"] = "   "  # whitespace-only
    path.write_text(json.dumps(payload))

    flagged = agg.unparsed_items(path)
    assert flagged == {"3", "7"}


def test_unparsed_items_empty_when_all_parse(tmp_path: Path) -> None:
    path = tmp_path / "avqa_m_real.json"
    _write_eval(path, lambda i: True)
    assert agg.unparsed_items(path) == set()


# --- provenance and integrity guards -----------------------------------------


def test_assert_condition_rejects_wrong_visual_mode(tmp_path: Path) -> None:
    """A cell whose filename and recorded condition disagree must not be aggregated —
    a duplicate run once overwrote a cell with a different condition."""
    path = tmp_path / "avqa_m_real_vis-occlusion-s1.json"
    _write_eval(path, lambda i: True)
    payload = json.loads(path.read_text())
    payload["config"] = {"audio_mode": "real", "visual_mode": "motion_blur", "visual_severity": 1.0}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="visual_mode"):
        agg.assert_condition(path, "real", "occlusion", 1.0)


def test_assert_condition_rejects_wrong_audio_mode(tmp_path: Path) -> None:
    path = tmp_path / "avqa_m_real.json"
    _write_eval(path, lambda i: True)
    payload = json.loads(path.read_text())
    payload["config"] = {"audio_mode": "silent", "visual_mode": "clean", "visual_severity": 0.0}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="audio_mode"):
        agg.assert_condition(path, "real", "motion_blur", 0.0)


def test_assert_condition_accepts_matching_and_legacy(tmp_path: Path) -> None:
    path = tmp_path / "avqa_m_real_vis-occlusion-s0.33.json"
    _write_eval(path, lambda i: True)
    payload = json.loads(path.read_text())
    payload["config"] = {"audio_mode": "real", "visual_mode": "occlusion", "visual_severity": 0.33}
    path.write_text(json.dumps(payload))
    agg.assert_condition(path, "real", "occlusion", 0.33)  # matching

    legacy = tmp_path / "avqa_m_silent.json"
    _write_eval(legacy, lambda i: True)  # no config block at all
    agg.assert_condition(legacy, "silent", "occlusion", 0.0)  # tolerated with a warning


def test_build_grid_rejects_item_to_video_disagreement(tmp_path: Path) -> None:
    """If two cells map the same item index to different videos they are not the same
    items, and pairing them would be meaningless."""
    _make_late_fallback_fixtures(tmp_path)
    bad = tmp_path / "avqa_smoke_silent_vis-motion_blur-s1.json"
    payload = json.loads(bad.read_text())
    payload["predictions"][0]["video"] = "some_other_video.mp4"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagrees with the clean baseline"):
        agg.build_grid(tmp_path, "smoke", "motion_blur", [0.33, 0.66, 1.0], "real", "silent")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (0, False), ("True", True), ("false", False), ("1", True)],
)
def test_as_bool_accepts_schema_variants(value: object, expected: bool) -> None:
    assert agg._as_bool(value, Path("x.json")) is expected


def test_as_bool_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        agg._as_bool("yes-ish", Path("x.json"))


def test_load_items_detects_schema_drift(tmp_path: Path) -> None:
    """Silently mis-parsing correctness would zero out every effect; the reported
    accuracy is the tripwire."""
    path = tmp_path / "avqa_m_real.json"
    _write_eval(path, lambda i: True)
    payload = json.loads(path.read_text())
    payload["accuracy"] = 12.5  # inconsistent with the rows
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        agg.load_items(path)
