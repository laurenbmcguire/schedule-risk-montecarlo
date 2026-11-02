"""Risk analytics on simulation output: percentiles, target probability,
Spearman sensitivity and schedule-margin recommendation."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from riskpath.workdays import WorkCalendar

_EPS = 1e-9


def percentiles(finish: np.ndarray, qs=(50, 80, 95)) -> dict[int, float]:
    vals = np.percentile(np.asarray(finish, dtype=float), list(qs))
    return {int(q): float(v) for q, v in zip(qs, vals)}


def prob_within(finish: np.ndarray, workdays: float) -> float:
    """P(project completes within ``workdays`` working days), whole-day granularity."""
    finish_day = np.ceil(np.asarray(finish, dtype=float) - _EPS)
    return float(np.mean(finish_day <= workdays + _EPS))


def prob_by_date(finish: np.ndarray, calendar: WorkCalendar, target) -> float:
    return prob_within(finish, calendar.workdays_through(target))


def spearman_sensitivity(durations: np.ndarray, finish: np.ndarray) -> np.ndarray:
    """Spearman rank correlation of each task's sampled duration with project finish.

    Constant (zero-variance) tasks get 0.
    """
    d = np.asarray(durations, dtype=float)
    f = np.asarray(finish, dtype=float)
    rf = stats.rankdata(f)
    rf = rf - rf.mean()
    rd = stats.rankdata(d, axis=1)
    rd = rd - rd.mean(axis=1, keepdims=True)
    num = rd @ rf
    den = np.sqrt((rd**2).sum(axis=1) * (rf**2).sum())
    out = np.zeros(d.shape[0])
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def schedule_margin(p80: float, deterministic: float) -> dict[str, float]:
    """Margin to hold so the committed date is the P80 rather than the
    deterministic (most-likely CPM) date. Recommended margin rounds the P80
    up to the whole working day in which it falls."""
    raw = float(p80 - deterministic)
    recommended = max(0, math.ceil(p80 - _EPS) - math.ceil(deterministic - _EPS))
    pct = 100.0 * raw / deterministic if deterministic > 0 else float("nan")
    return {"raw_days": raw, "recommended_days": int(recommended), "pct_of_deterministic": pct}


def merge_bias(finish: np.ndarray, pert_mean_cpm: float) -> float:
    """Mean simulated finish minus the CPM duration computed with PERT-mean task
    durations. With no risk events this gap is pure merge bias: the max of
    uncertain parallel paths is larger on average than the max of their means."""
    return float(np.mean(finish) - pert_mean_cpm)
