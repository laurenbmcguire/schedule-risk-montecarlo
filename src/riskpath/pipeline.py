"""End-to-end pipeline: plan -> CPM -> Monte Carlo -> analytics -> charts + report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from riskpath import __version__, analysis, charts
from riskpath.cpm import cpm
from riskpath.distributions import pert_mean
from riskpath.montecarlo import simulate
from riskpath.plan import Plan, load_plan
from riskpath.report import build_markdown
from riskpath.sample import write_sample

DEFAULT_CONFIG = Path("examples/project.yml")

DEFAULTS: dict[str, Any] = {
    "name": None,
    "plan": "plan.csv",
    "start_date": None,  # required: config file, plan settings or --start
    "target_date": None,
    "holidays": [],
    "rho": 0.0,
    "iterations": 10_000,
    "seed": 42,
    "pert_lambda": 4.0,
}

SETTING_KEYS = ("start_date", "target_date", "holidays", "rho", "iterations", "seed", "pert_lambda")


@dataclass
class RunConfig:
    config: Path = DEFAULT_CONFIG
    plan: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    reports_dir: Path = Path("reports")
    img_dir: Path = Path("docs/img")


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def resolve(cfg: RunConfig) -> tuple[Plan, dict[str, Any], Path]:
    """Load (or generate) the config and plan and merge settings.

    Precedence: defaults < config file < settings embedded in the plan < CLI overrides.
    """
    config_path = Path(cfg.config)
    if not config_path.exists() and config_path == DEFAULT_CONFIG and cfg.plan is None:
        write_sample(config_path.parent)
    conf: dict[str, Any] = dict(DEFAULTS)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            conf.update(yaml.safe_load(fh) or {})
    if cfg.plan is not None:
        plan_path = Path(cfg.plan)
    else:
        plan_path = (config_path.parent / str(conf["plan"])).resolve()
        if not plan_path.exists() and config_path.parent == DEFAULT_CONFIG.parent:
            write_sample(config_path.parent)
    plan = load_plan(plan_path)
    settings = {k: conf.get(k) for k in SETTING_KEYS}
    settings.update({k: v for k, v in plan.settings.items() if k in SETTING_KEYS})
    settings.update({k: v for k, v in cfg.overrides.items() if v is not None})
    # an explicitly passed plan file names the project; otherwise the config does
    fallback = (plan.name, conf.get("name")) if cfg.plan is not None else (conf.get("name"), plan.name)
    settings["name"] = cfg.overrides.get("name") or plan.settings.get("name") or fallback[0] or fallback[1]
    if settings["start_date"] is None:
        raise ValueError("a start_date is required (config file, plan settings or --start)")
    settings["start_date"] = _iso(settings["start_date"])
    if settings["target_date"] is not None:
        settings["target_date"] = _iso(settings["target_date"])
    settings["holidays"] = [_iso(h) for h in (settings.get("holidays") or [])]
    settings["rho"] = float(settings["rho"])
    settings["iterations"] = int(settings["iterations"])
    settings["seed"] = int(settings["seed"])
    settings["pert_lambda"] = float(settings["pert_lambda"])
    return plan, settings, plan_path


def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


def analyse(plan: Plan, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all engines. Returns (summary dict, raw objects for charting)."""
    from riskpath.workdays import WorkCalendar

    cal = WorkCalendar(settings["start_date"], settings["holidays"])
    n_iter, seed, rho, lam = settings["iterations"], settings["seed"], settings["rho"], settings["pert_lambda"]

    det = cpm(plan)
    a, m, b = (np.array(plan.column(c)) for c in ("optimistic", "most_likely", "pessimistic"))
    det_mean = cpm(plan, pert_mean(a, m, b, lam))

    sim = simulate(plan, n_iter, seed, rho=rho, lam=lam)
    sim_norisk = simulate(plan, n_iter, seed, rho=rho, lam=lam, include_risks=False)
    sim_indep = simulate(plan, n_iter, seed, rho=0.0, lam=lam) if rho > 0 else sim

    pct = analysis.percentiles(sim.finish, (5, 50, 80, 95))
    p50, p80, p95 = pct[50], pct[80], pct[95]
    margin = analysis.schedule_margin(p80, det.duration)
    # Sensitivity uses the independent (rho = 0) run with common random numbers:
    # under a shared copula factor every task correlates with the finish, which
    # would hide each task's own contribution.
    sens = analysis.spearman_sensitivity(sim_indep.durations, sim_indep.finish)
    sens_corr = analysis.spearman_sensitivity(sim.durations, sim.finish)
    ci = sim.criticality_index

    target_date = settings["target_date"]
    target_block = None
    if target_date:
        tdays = cal.workdays_through(date.fromisoformat(target_date))
        target_block = {
            "date": target_date,
            "workdays": tdays,
            "probability": _r(analysis.prob_within(sim.finish, tdays), 4),
            "probability_independent": _r(analysis.prob_within(sim_indep.finish, tdays), 4),
            "probability_without_risk_events": _r(analysis.prob_within(sim_norisk.finish, tdays), 4),
        }

    pct_ind = analysis.percentiles(sim_indep.finish, (5, 50, 80, 95))
    pct_nr = analysis.percentiles(sim_norisk.finish, (50, 80, 95))

    tasks = []
    for i, t in enumerate(plan.tasks):
        tasks.append(
            {
                "id": t.id,
                "name": t.name,
                "depends_on": list(t.depends_on),
                "optimistic": t.optimistic,
                "most_likely": t.most_likely,
                "pessimistic": t.pessimistic,
                "pert_mean": _r(pert_mean(t.optimistic, t.most_likely, t.pessimistic, lam)),
                "risk_event_prob": t.risk_event_prob,
                "risk_event_days": t.risk_event_days,
                "es": _r(det.es[i]),
                "ef": _r(det.ef[i]),
                "total_float": _r(det.total_float[i]),
                "deterministic_critical": bool(det.critical[i]),
                "start_date": cal.add_workdays(int(det.es[i])).isoformat(),
                "finish_date": cal.finish_date(det.ef[i]).isoformat() if det.ef[i] > 0 else cal.start_date.isoformat(),
                "criticality_index": _r(ci[i], 4),
                "sensitivity_spearman": _r(sens[i], 4),
                "sensitivity_spearman_correlated": _r(sens_corr[i], 4),
                "risk_hit_rate": _r(sim.risk_hit_rate[i], 4),
            }
        )

    by_ci = sorted(tasks, key=lambda r: (-r["criticality_index"], r["id"]))
    by_sens = sorted(tasks, key=lambda r: (-abs(r["sensitivity_spearman"]), r["id"]))
    near_critical = [
        r["id"] for r in tasks if (not r["deterministic_critical"]) and r["criticality_index"] >= 0.2
    ]

    summary: dict[str, Any] = {
        "project": settings["name"],
        "generated_by": f"riskpath {__version__}",
        "n_tasks": len(plan),
        "settings": {
            "start_date": cal.start_date.isoformat(),
            "target_date": target_date,
            "holidays": settings["holidays"],
            "iterations": n_iter,
            "seed": seed,
            "rho": rho,
            "pert_lambda": lam,
            "calendar": "Mon-Fri working days minus holidays",
            "sensitivity_basis": "Spearman rank correlation on the rho = 0 run (common random numbers)",
        },
        "deterministic": {
            "basis": "most-likely durations, risk events excluded",
            "duration_days": _r(det.duration),
            "finish_date": cal.finish_date(det.duration).isoformat(),
            "critical_path": det.critical_path,
            "pert_mean_duration_days": _r(det_mean.duration),
            "probability_of_meeting": _r(analysis.prob_within(sim.finish, det.duration), 4),
        },
        "simulation": {
            "mean_days": _r(np.mean(sim.finish)),
            "std_days": _r(np.std(sim.finish)),
            "min_days": _r(np.min(sim.finish)),
            "max_days": _r(np.max(sim.finish)),
            "p5_days": _r(pct[5]),
            "p50_days": _r(p50),
            "p80_days": _r(p80),
            "p95_days": _r(p95),
            "p50_date": cal.finish_date(p50).isoformat(),
            "p80_date": cal.finish_date(p80).isoformat(),
            "p95_date": cal.finish_date(p95).isoformat(),
        },
        "target": target_block,
        "margin": {
            "basis": "P80 minus deterministic CPM duration",
            "raw_days": _r(margin["raw_days"]),
            "recommended_days": margin["recommended_days"],
            "pct_of_deterministic": _r(margin["pct_of_deterministic"], 1),
            "commit_date": cal.finish_date(p80).isoformat(),
        },
        "merge_bias": {
            "basis": "mean simulated finish (no risk events) minus CPM on PERT-mean durations",
            "days": _r(analysis.merge_bias(sim_norisk.finish, det_mean.duration)),
        },
        "risk_events": {
            "p80_without_risk_events_days": _r(pct_nr[80]),
            "p80_added_by_risk_events_days": _r(p80 - pct_nr[80]),
            "mean_added_by_risk_events_days": _r(np.mean(sim.finish) - np.mean(sim_norisk.finish)),
        },
        "correlation": {
            "rho": rho,
            "p80_independent_days": _r(pct_ind[80]),
            "p80_correlated_days": _r(p80),
            "p5_p95_spread_independent_days": _r(pct_ind[95] - pct_ind[5]),
            "p5_p95_spread_correlated_days": _r(pct[95] - pct[5]),
        },
        "top_criticality": [
            {"id": r["id"], "name": r["name"], "criticality_index": r["criticality_index"]} for r in by_ci[:5]
        ],
        "top_drivers": [
            {"id": r["id"], "name": r["name"], "sensitivity_spearman": r["sensitivity_spearman"]}
            for r in by_sens[:5]
        ],
        "near_critical_tasks": near_critical,
        "tasks": tasks,
    }
    raw = {"cal": cal, "det": det, "sim": sim, "sens": sens}
    return summary, raw


def render_charts(summary: dict[str, Any], raw: dict[str, Any], img_dir: Path) -> dict[str, str]:
    img_dir = Path(img_dir)
    cal, det, sim, sens = raw["cal"], raw["det"], raw["sim"], raw["sens"]
    s = summary["simulation"]
    tgt = summary["target"]
    target = date.fromisoformat(tgt["date"]) if tgt else None
    out = {
        "s_curve": charts.s_curve(
            sim.finish, cal, det.duration, s["p50_days"], s["p80_days"], target,
            tgt["probability"] if tgt else None, img_dir / "s_curve.png",
            f"{summary['project']}: probability of finishing by date",
        ),
        "histogram": charts.histogram(
            sim.finish, det.duration, s["p50_days"], s["p80_days"], s["p95_days"],
            tgt["workdays"] if tgt else None, img_dir / "histogram.png",
        ),
        "tornado": charts.tornado(sim.ids, sim.names, sens, img_dir / "tornado.png"),
        "criticality": charts.criticality(
            sim.ids, sim.names, sim.criticality_index, det.critical, img_dir / "criticality.png"
        ),
        "gantt": charts.gantt(det, cal, img_dir / "gantt.png", "Deterministic schedule with total float"),
    }
    return {k: str(v) for k, v in out.items()}


def run(cfg: RunConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RunConfig()
    plan, settings, plan_path = resolve(cfg)
    summary, raw = analyse(plan, settings)
    try:
        summary["plan_file"] = Path(plan_path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        summary["plan_file"] = str(plan_path)
    chart_paths = render_charts(summary, raw, cfg.img_dir)
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    img_rel = _relpath(Path(cfg.img_dir), reports)
    (reports / "schedule_risk_report.md").write_text(build_markdown(summary, img_rel), encoding="utf-8")
    summary["_charts"] = chart_paths
    return summary


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(Path(target).resolve(), Path(start).resolve()).replace(os.sep, "/")
