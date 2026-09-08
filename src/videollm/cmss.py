"""Causal Modality Substitution Surface (CMSS) estimands.

Pure, side-effect-free computation of the pre-registered CMSS estimands from
per-item correctness vectors (see ``preregistration.md``). File loading and report
rendering live in ``scripts/aggregate_cmss.py``; this module is the unit-tested
numerical core.

Estimands (visual reliability level ``v``; audio state = real vs. the ``silent`` null):

- ``Delta_A(v) = E_i[ Y_i(real, v) - Y_i(silent, v) ]`` -- the conditional audio
  contribution at visual level ``v``, reported in percentage points (pp).
- ``SG = Delta_A(v_severe) - Delta_A(v_clean)`` -- the substitution gain, i.e. how
  much more the model leans on audio once vision is degraded.
- ``tau_switch`` -- the smallest degradation severity at which ``Delta_A`` exceeds
  its clean value by at least ``margin_pp`` (pre-registered 10 pp).

Confidence intervals use the pre-registered **video-cluster** percentile bootstrap
(whole videos are resampled, not QA rows) with seed 2027, so that several questions
sharing one video are not treated as independent evidence.

Caveat: because every call uses the same seed, all cells share one resample matrix.
Point estimates are unaffected, but the Monte-Carlo error is common-mode across
severities and families -- two cells' CIs failing to overlap is therefore NOT evidence
that they are independently distinguishable.

The fully-blanked-video condition is deliberately *not* a severity level here: it is
the delivery positive control and is handled separately by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SEED = 2027
N_BOOT = 10_000
SWITCH_MARGIN_PP = 10.0


@dataclass(frozen=True)
class DeltaResult:
    """Conditional audio contribution ``Delta_A(v)`` at a single visual severity."""

    severity: float
    delta_pp: float
    ci95_lo: float
    ci95_hi: float
    n_items: int
    n_clusters: int


@dataclass(frozen=True)
class CMSSResult:
    """CMSS summary for one (model, corruption-family) curve."""

    deltas: tuple[DeltaResult, ...]  # one per severity, ascending
    sg_severity: float  # the severity SG was evaluated at (max present, NOT always 1.0)
    sg_pp: float
    sg_ci95_lo: float
    sg_ci95_hi: float
    tau_switch: float | None
    margin_pp: float


def _as_int_vector(values: object, name: str) -> np.ndarray:
    """Coerce a per-item correctness sequence to an integer vector, shape ``(N,)``."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D per-item, got shape {arr.shape}")
    return arr.astype(np.int64)


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile CI for ``mean(values)``, resampling whole clusters with replacement.

    Clusters (videos) are resampled; the bootstrap mean is the ratio of resampled
    cluster sums to resampled cluster counts, which is the standard cluster bootstrap
    for a mean under unequal cluster sizes.

    Args:
        values: Per-item numeric series, shape ``(N,)``.
        cluster_ids: Per-item cluster (video) label, shape ``(N,)``.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed (pre-registered 2027).
        alpha: Two-sided error rate.

    Returns:
        ``(point, ci95_lo, ci95_hi)`` in the same unit as ``values``.

    Raises:
        ValueError: If shapes disagree or the input is empty.
    """
    vals = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    if vals.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {vals.shape}")
    if vals.shape != clusters.shape:
        raise ValueError(f"values {vals.shape} and cluster_ids {clusters.shape} must align")
    if vals.size == 0:
        raise ValueError("values must be non-empty")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    _, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    cluster_sums = np.bincount(inverse, weights=vals, minlength=n_clusters)  # (C,)
    cluster_counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)  # (C,)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))  # (B, C)
    boot_means = cluster_sums[idx].sum(axis=1) / cluster_counts[idx].sum(axis=1)  # (B,)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def delta_a(
    real_correct: object,
    silent_correct: object,
    cluster_ids: np.ndarray,
    severity: float,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> DeltaResult:
    """Conditional audio contribution ``Delta_A(v)`` in pp, with a cluster-bootstrap CI.

    ``real_correct`` and ``silent_correct`` must cover the same items in the same
    order (item-paired) at the same visual severity.

    Args:
        real_correct: Per-item correctness under real audio, shape ``(N,)``.
        silent_correct: Per-item correctness under the silent null, shape ``(N,)``.
        cluster_ids: Per-item video labels, shape ``(N,)``.
        severity: The visual severity this pair was evaluated at.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        A :class:`DeltaResult` with the effect and CI in percentage points.

    Raises:
        ValueError: If the three inputs do not share a common shape.
    """
    real = _as_int_vector(real_correct, "real_correct")
    silent = _as_int_vector(silent_correct, "silent_correct")
    clusters = np.asarray(cluster_ids)
    if not (real.shape == silent.shape == clusters.shape):
        raise ValueError(f"real {real.shape}, silent {silent.shape} and cluster_ids {clusters.shape} must be paired")
    diff = (real - silent).astype(np.float64)  # (N,) in {-1, 0, 1}
    point, lo, hi = cluster_bootstrap_mean(diff, clusters, n_boot, seed)
    return DeltaResult(
        severity=float(severity),
        delta_pp=point * 100.0,
        ci95_lo=lo * 100.0,
        ci95_hi=hi * 100.0,
        n_items=int(real.size),
        n_clusters=int(np.unique(clusters).size),
    )


def compute_cmss(
    per_severity: dict[float, tuple[object, object]],
    cluster_ids: np.ndarray,
    margin_pp: float = SWITCH_MARGIN_PP,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> CMSSResult:
    """Compute the CMSS curve, substitution gain, and switch threshold.

    Args:
        per_severity: Maps visual severity (``0.0`` == clean) to an item-paired
            ``(real_correct, silent_correct)`` tuple, all aligned to ``cluster_ids``.
            Must contain severity ``0.0`` plus at least one degraded level. The
            blanked-video positive control must NOT be passed in here.
        cluster_ids: Per-item video labels, shape ``(N,)``, shared across levels.
        margin_pp: Switch-threshold margin in pp (pre-registered 10).
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed (pre-registered 2027).

    Returns:
        A :class:`CMSSResult`. ``SG`` is evaluated at the **largest severity present**,
        which is reported as ``sg_severity`` -- passing a truncated severity list
        silently changes what ``SG`` means, so the value is recorded rather than assumed.
        ``tau_switch`` is ``None`` when no severity reaches the margin, which is itself a
        reportable (flat-curve) outcome.

    Raises:
        ValueError: If the clean level is missing or fewer than two levels are given.
    """
    if 0.0 not in per_severity:
        raise ValueError("per_severity must include the clean baseline at severity 0.0")
    if len(per_severity) < 2:
        raise ValueError("compute_cmss needs the clean level plus at least one degraded level")

    severities = sorted(per_severity)
    deltas = tuple(delta_a(per_severity[s][0], per_severity[s][1], cluster_ids, s, n_boot, seed) for s in severities)
    by_severity = {d.severity: d for d in deltas}
    clean_delta = by_severity[0.0].delta_pp
    most_degraded = severities[-1]

    # SG gets its own cluster bootstrap on the per-item difference-in-differences.
    real_clean, silent_clean = per_severity[0.0]
    real_deg, silent_deg = per_severity[most_degraded]
    d_clean = _as_int_vector(real_clean, "real_correct") - _as_int_vector(silent_clean, "silent_correct")
    d_deg = _as_int_vector(real_deg, "real_correct") - _as_int_vector(silent_deg, "silent_correct")
    sg_series = (d_deg - d_clean).astype(np.float64)  # (N,)
    sg_point, sg_lo, sg_hi = cluster_bootstrap_mean(sg_series, np.asarray(cluster_ids), n_boot, seed)

    tau_switch: float | None = None
    for s in severities:
        if s <= 0.0:
            continue
        if by_severity[s].delta_pp - clean_delta >= margin_pp:
            tau_switch = s
            break

    return CMSSResult(
        deltas=deltas,
        sg_severity=float(most_degraded),
        sg_pp=sg_point * 100.0,
        sg_ci95_lo=sg_lo * 100.0,
        sg_ci95_hi=sg_hi * 100.0,
        tau_switch=tau_switch,
        margin_pp=float(margin_pp),
    )
