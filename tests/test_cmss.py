"""Tests for the CMSS estimands (Delta_A, substitution gain, switch threshold)."""

from __future__ import annotations

import numpy as np
import pytest

from videollm.cmss import (
    CMSSResult,
    cluster_bootstrap_mean,
    compute_cmss,
    delta_a,
)

N_VIDEOS, PER_VIDEO = 20, 5
N = N_VIDEOS * PER_VIDEO
FAST_BOOT = 200  # keep unit tests quick; production runs use the preregistered 10k


def _clusters() -> np.ndarray:
    """Per-item video labels, shape (N,): 20 videos x 5 questions."""
    return np.repeat(np.arange(N_VIDEOS), PER_VIDEO)


def _ones() -> np.ndarray:
    return np.ones(N, dtype=np.int64)


def _zeros() -> np.ndarray:
    return np.zeros(N, dtype=np.int64)


# --- cluster bootstrap -------------------------------------------------------


def test_bootstrap_point_estimate_is_plain_mean() -> None:
    values = np.arange(N, dtype=np.float64)
    point, lo, hi = cluster_bootstrap_mean(values, _clusters(), n_boot=FAST_BOOT)
    assert point == pytest.approx(values.mean())
    assert lo <= point <= hi


def test_bootstrap_constant_series_has_zero_width_ci() -> None:
    values = np.full(N, 0.42)
    point, lo, hi = cluster_bootstrap_mean(values, _clusters(), n_boot=FAST_BOOT)
    assert point == pytest.approx(0.42)
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


def test_bootstrap_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=N)
    a = cluster_bootstrap_mean(values, _clusters(), n_boot=FAST_BOOT, seed=2027)
    b = cluster_bootstrap_mean(values, _clusters(), n_boot=FAST_BOOT, seed=2027)
    assert a == b


def test_bootstrap_clustering_widens_ci_when_effect_is_video_level() -> None:
    """Video-level (fully correlated within video) signal must widen the CI vs. treating
    every item as independent -- this is why the preregistration mandates clustering."""
    rng = np.random.default_rng(7)
    per_video = rng.normal(size=N_VIDEOS)
    values = np.repeat(per_video, PER_VIDEO)  # (N,) perfectly correlated within video
    _, lo_c, hi_c = cluster_bootstrap_mean(values, _clusters(), n_boot=2000, seed=2027)
    # Pretend every item is its own cluster == naive item bootstrap.
    _, lo_i, hi_i = cluster_bootstrap_mean(values, np.arange(N), n_boot=2000, seed=2027)
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_mean(np.ones(N), np.ones(N - 1))
    with pytest.raises(ValueError):
        cluster_bootstrap_mean(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        cluster_bootstrap_mean(np.ones(N), _clusters(), n_boot=0)


# --- Delta_A -----------------------------------------------------------------


def test_delta_a_full_audio_dependence_is_100pp() -> None:
    res = delta_a(_ones(), _zeros(), _clusters(), severity=1.0, n_boot=FAST_BOOT)
    assert res.delta_pp == pytest.approx(100.0)
    assert res.n_items == N
    assert res.n_clusters == N_VIDEOS
    assert res.severity == 1.0


def test_delta_a_no_audio_dependence_is_zero() -> None:
    res = delta_a(_ones(), _ones(), _clusters(), severity=0.0, n_boot=FAST_BOOT)
    assert res.delta_pp == pytest.approx(0.0)
    assert res.ci95_lo == pytest.approx(0.0)
    assert res.ci95_hi == pytest.approx(0.0)


def test_delta_a_can_be_negative() -> None:
    res = delta_a(_zeros(), _ones(), _clusters(), severity=0.5, n_boot=FAST_BOOT)
    assert res.delta_pp == pytest.approx(-100.0)


def test_delta_a_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError):
        delta_a(_ones(), np.ones(N - 1, dtype=np.int64), _clusters(), severity=0.0)


# --- compute_cmss ------------------------------------------------------------


def _late_fallback_grid() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Audio is ignored until vision is severely degraded (the hypothesised regime)."""
    silent_severe = _ones().copy()
    silent_severe[: N // 2] = 0  # audio matters on half the items only at severity 1.0
    return {
        0.0: (_ones(), _ones()),  # Delta_A = 0
        0.33: (_ones(), _ones()),  # Delta_A = 0
        0.66: (_ones(), _ones()),  # Delta_A = 0
        1.0: (_ones(), silent_severe),  # Delta_A = 50pp
    }


def test_compute_cmss_detects_late_fallback() -> None:
    res = compute_cmss(_late_fallback_grid(), _clusters(), n_boot=FAST_BOOT)
    assert isinstance(res, CMSSResult)
    assert [d.severity for d in res.deltas] == [0.0, 0.33, 0.66, 1.0]
    assert res.deltas[0].delta_pp == pytest.approx(0.0)
    assert res.deltas[-1].delta_pp == pytest.approx(50.0)
    assert res.sg_pp == pytest.approx(50.0)  # 50pp - 0pp
    assert res.tau_switch == 1.0  # only the severe level clears the 10pp margin


def test_compute_cmss_flat_curve_has_no_switch() -> None:
    grid = {s: (_ones(), _ones()) for s in (0.0, 0.33, 0.66, 1.0)}
    res = compute_cmss(grid, _clusters(), n_boot=FAST_BOOT)
    assert res.sg_pp == pytest.approx(0.0)
    assert res.tau_switch is None
    assert all(d.delta_pp == pytest.approx(0.0) for d in res.deltas)


def test_tau_switch_picks_smallest_qualifying_severity() -> None:
    silent_mild = _ones().copy()
    silent_mild[: N // 5] = 0  # Delta_A = 20pp at severity 0.33 (clears 10pp)
    silent_severe = _zeros()  # Delta_A = 100pp at severity 1.0
    grid = {
        0.0: (_ones(), _ones()),
        0.33: (_ones(), silent_mild),
        1.0: (_ones(), silent_severe),
    }
    res = compute_cmss(grid, _clusters(), n_boot=FAST_BOOT)
    assert res.tau_switch == 0.33
    assert res.sg_pp == pytest.approx(100.0)


def test_margin_controls_switch_threshold() -> None:
    silent_mild = _ones().copy()
    silent_mild[: N // 5] = 0  # Delta_A = 20pp
    grid = {0.0: (_ones(), _ones()), 0.33: (_ones(), silent_mild)}
    assert compute_cmss(grid, _clusters(), margin_pp=10.0, n_boot=FAST_BOOT).tau_switch == 0.33
    assert compute_cmss(grid, _clusters(), margin_pp=25.0, n_boot=FAST_BOOT).tau_switch is None


def test_sg_is_zero_when_audio_dependence_is_constant() -> None:
    """Uniformly high audio reliance is NOT substitution -- SG must stay at zero."""
    grid = {s: (_ones(), _zeros()) for s in (0.0, 0.5, 1.0)}
    res = compute_cmss(grid, _clusters(), n_boot=FAST_BOOT)
    assert all(d.delta_pp == pytest.approx(100.0) for d in res.deltas)
    assert res.sg_pp == pytest.approx(0.0)
    assert res.tau_switch is None


def test_compute_cmss_requires_clean_and_two_levels() -> None:
    with pytest.raises(ValueError):
        compute_cmss({0.5: (_ones(), _zeros()), 1.0: (_ones(), _zeros())}, _clusters())
    with pytest.raises(ValueError):
        compute_cmss({0.0: (_ones(), _ones())}, _clusters())


def test_sg_severity_is_recorded_not_assumed() -> None:
    """SG is taken at the largest severity present; a truncated severity list changes
    what SG means, so the level must be reported rather than inferred."""
    grid = {s: (_ones(), _zeros()) for s in (0.0, 0.33, 0.66)}
    res = compute_cmss(grid, _clusters(), n_boot=FAST_BOOT)
    assert res.sg_severity == 0.66

    full = {s: (_ones(), _zeros()) for s in (0.0, 0.33, 0.66, 1.0)}
    assert compute_cmss(full, _clusters(), n_boot=FAST_BOOT).sg_severity == 1.0
