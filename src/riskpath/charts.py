"""Charts: S-curve (hero), histogram, tornado, criticality index and Gantt."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Patch

from riskpath import style
from riskpath.cpm import CPMResult
from riskpath.workdays import WorkCalendar


def _fmt(d: date) -> str:
    return d.strftime("%d %b %Y")


def _short(name: str, n: int = 38) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def s_curve(
    finish: np.ndarray,
    cal: WorkCalendar,
    deterministic_days: float,
    p50: float,
    p80: float,
    target: date | None,
    prob_target: float | None,
    path: Path,
    title: str,
) -> Path:
    fig, ax = style.new_figure()
    days = np.sort(np.ceil(np.asarray(finish) - 1e-9))
    uniq, counts = np.unique(days, return_counts=True)
    cdf = np.cumsum(counts) / days.size
    dates = cal.finish_dates(uniq).astype("datetime64[D]").astype(object)
    ax.step(dates, cdf, where="post", color=style.CRIMSON, lw=2.4, label="P(finish on or before date)")
    ax.fill_between(dates, cdf, step="post", color=style.CRIMSON, alpha=0.07)

    det_date = cal.finish_date(deterministic_days)
    p50_date = cal.finish_date(p50)
    p80_date = cal.finish_date(p80)
    ax.axvline(det_date, color=style.MUTED, lw=1.4, ls=":", label=f"Deterministic CPM  {_fmt(det_date)}")
    ax.axvline(p50_date, color=style.GOLD, lw=1.8, label=f"P50  {_fmt(p50_date)}")
    ax.axvline(p80_date, color=style.DEEP_CRIMSON, lw=1.8, label=f"P80  {_fmt(p80_date)}")
    if target is not None:
        lab = f"Target  {_fmt(target)}"
        if prob_target is not None:
            lab += f"  (P = {prob_target:.0%})"
        ax.axvline(target, color=style.INK, lw=1.6, ls="--", label=lab)
        if prob_target is not None:
            ax.scatter([target], [prob_target], color=style.INK, zorder=5, s=28)
    ax.axhline(0.8, color=style.GRID, lw=1.0, zorder=0)
    ax.axhline(0.5, color=style.GRID, lw=1.0, zorder=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Cumulative probability")
    ax.set_xlabel("Project finish date")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    return style.save(fig, path)


def histogram(
    finish: np.ndarray,
    deterministic_days: float,
    p50: float,
    p80: float,
    p95: float,
    target_days: int | None,
    path: Path,
) -> Path:
    fig, ax = style.new_figure()
    f = np.asarray(finish)
    bins = np.arange(np.floor(f.min()), np.ceil(f.max()) + 2, 2)
    ax.hist(f, bins=bins, color=style.SAND, edgecolor=style.GOLD, linewidth=0.6)
    for x, color, ls, lab in [
        (deterministic_days, style.MUTED, ":", f"Deterministic {deterministic_days:.0f} d"),
        (p50, style.GOLD, "-", f"P50 {p50:.1f} d"),
        (p80, style.CRIMSON, "-", f"P80 {p80:.1f} d"),
        (p95, style.DEEP_CRIMSON, "-", f"P95 {p95:.1f} d"),
    ]:
        ax.axvline(x, color=color, lw=1.8, ls=ls, label=lab)
    if target_days is not None:
        ax.axvline(target_days, color=style.INK, lw=1.5, ls="--", label=f"Target {target_days} d")
    ax.set_xlabel("Project duration (working days)")
    ax.set_ylabel("Iterations")
    ax.set_title(f"Distribution of project duration ({f.size:,} iterations)")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    return style.save(fig, path)


def tornado(ids: list[str], names: list[str], rho: np.ndarray, path: Path, top: int = 10) -> Path:
    fig, ax = style.new_figure()
    order = np.argsort(-np.abs(rho))[:top][::-1]
    labels = [f"{ids[i]}  {_short(names[i])}" for i in order]
    vals = rho[order]
    colors = [style.CRIMSON if v >= 0 else style.GOLD for v in vals]
    ax.barh(range(len(order)), vals, color=colors, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=style.INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color=style.MUTED, lw=0.8)
    hi = max(0.1, float(np.abs(vals).max()) * 1.18)
    ax.set_xlim(min(0.0, float(vals.min()) * 1.2), hi)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman rank correlation of task duration with project finish")
    ax.set_title("Sensitivity: which task durations drive the finish date")
    ax.text(0.99, 0.02, "Spearman rho, independent run (rho = 0)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=style.MUTED)
    fig.tight_layout()
    return style.save(fig, path)


def criticality(
    ids: list[str], names: list[str], ci: np.ndarray, det_critical: np.ndarray, path: Path
) -> Path:
    fig, ax = style.new_figure(figsize=(10, 5.625))
    order = np.argsort(ci, kind="stable")
    labels = [f"{ids[i]}  {_short(names[i], 34)}" + ("  ◆" if det_critical[i] else "") for i in order]
    vals = ci[order]
    colors = [style.CRIMSON if v >= 0.5 else style.GOLD if v >= 0.1 else style.SAND for v in vals]
    ax.barh(range(len(order)), vals, color=colors, height=0.7)
    for y, v in enumerate(vals):
        ax.text(v + 0.008, y, f"{v:.0%}", va="center", fontsize=7.5, color=style.INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Criticality index (share of iterations on a critical path)   ◆ = on deterministic critical path")
    ax.set_title("Criticality index by task")
    ax.legend(
        handles=[
            Patch(color=style.CRIMSON, label=">= 50%"),
            Patch(color=style.GOLD, label="10-50%"),
            Patch(color=style.SAND, label="< 10%"),
        ],
        loc="lower right",
        fontsize=9,
    )
    fig.tight_layout()
    return style.save(fig, path)


def gantt(det: CPMResult, cal: WorkCalendar, path: Path, title: str) -> Path:
    fig, ax = style.new_figure()
    n = len(det.ids)
    order = sorted(range(n), key=lambda i: (det.es[i], det.ef[i], i))
    for row, i in enumerate(order):
        y = n - 1 - row
        dur = det.ef[i] - det.es[i]
        color = style.CRIMSON if det.critical[i] else style.GOLD
        ax.barh(y, dur, left=det.es[i], color=color, height=0.62)
        if det.total_float[i] > 0:
            ax.barh(y, det.total_float[i], left=det.ef[i], color=style.SAND, height=0.62, alpha=0.9,
                    hatch="///", edgecolor=style.GOLD, linewidth=0)
    ax.set_yticks([n - 1 - r for r in range(n)])
    ax.set_yticklabels([f"{det.ids[i]}  {_short(det.names[i], 34)}" for i in order], fontsize=7.5)
    # month ticks expressed as working-day offsets
    start = cal.start_date
    end = cal.finish_date(det.duration)
    ticks, labels = [0], [start.strftime("%d %b %Y")]
    y, m = start.year, start.month
    while True:
        d = date(y, m, 1)
        if d > end:
            break
        if d > start and cal.workday_index(d) >= 8:
            ticks.append(cal.workday_index(d))
            labels.append(d.strftime("%b"))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, det.duration * 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Deterministic schedule from {_fmt(start)} (most-likely durations, working-day calendar)")
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(color=style.CRIMSON, label="Critical (zero float)"),
            Patch(color=style.GOLD, label="Non-critical"),
            Patch(facecolor=style.SAND, edgecolor=style.GOLD, hatch="///", label="Total float"),
        ],
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    return style.save(fig, path)
