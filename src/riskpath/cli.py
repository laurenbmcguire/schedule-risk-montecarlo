"""Command line: ``python -m riskpath run`` and ``python -m riskpath pm``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from riskpath import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="riskpath", description="Schedule risk Monte Carlo for project plans.")
    p.add_argument("--version", action="version", version=f"riskpath {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run CPM + Monte Carlo, write charts and reports")
    r.add_argument("--config", default="examples/project.yml", help="run configuration YAML")
    r.add_argument("--plan", default=None, help="plan file (.csv, .yaml, .json); overrides the config")
    r.add_argument("--start", dest="start_date", default=None, help="project start date YYYY-MM-DD")
    r.add_argument("--target", dest="target_date", default=None, help="target finish date YYYY-MM-DD")
    r.add_argument("--rho", type=float, default=None, help="Gaussian copula correlation in [0, 1)")
    r.add_argument("--iterations", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--lambda", dest="pert_lambda", type=float, default=None, help="PERT shape (default 4)")
    r.add_argument("--reports-dir", default="reports")
    r.add_argument("--img-dir", default="docs/img")

    m = sub.add_parser("pm", help="run the AI project manager and write project/STATUS.md")
    m.add_argument("--project-dir", default="project")
    m.add_argument("--summary", default="reports/summary.json")
    m.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD)")
    m.add_argument("--no-llm", action="store_true", help="skip the optional LLM narrative")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        from riskpath.pipeline import RunConfig, run

        t0 = time.perf_counter()
        cfg = RunConfig(
            config=Path(args.config),
            plan=Path(args.plan) if args.plan else None,
            overrides={
                k: getattr(args, k)
                for k in ("start_date", "target_date", "rho", "iterations", "seed", "pert_lambda")
            },
            reports_dir=Path(args.reports_dir),
            img_dir=Path(args.img_dir),
        )
        s = run(cfg)
        det, sim, mar, tgt = s["deterministic"], s["simulation"], s["margin"], s["target"]
        print(f"{s['project']}: {s['n_tasks']} tasks, {s['settings']['iterations']:,} iterations")
        print(f"  deterministic  {det['duration_days']:.0f} d  -> {det['finish_date']}")
        print(f"  P50 {sim['p50_days']:.1f} d -> {sim['p50_date']}   P80 {sim['p80_days']:.1f} d -> "
              f"{sim['p80_date']}   P95 {sim['p95_days']:.1f} d -> {sim['p95_date']}")
        if tgt:
            print(f"  P(finish by {tgt['date']}) = {tgt['probability']:.1%}")
        print(f"  recommended margin: {mar['recommended_days']} working days")
        print(f"  wrote {args.reports_dir}/summary.json, {args.reports_dir}/schedule_risk_report.md and "
              f"{len(s['_charts'])} charts in {args.img_dir}/ ({time.perf_counter() - t0:.1f}s)")
        return 0
    if args.command == "pm":
        from riskpath.ai_pm import run_pm

        status = run_pm(
            project_dir=Path(args.project_dir),
            summary_path=Path(args.summary),
            today=args.today,
            use_llm=not args.no_llm,
        )
        print(f"AI PM: health {status['health']['rag']} - {status['health']['reason']}")
        print(f"  summary source: {status['executive_summary_source']}")
        print(f"  wrote {args.project_dir}/STATUS.md and {args.project_dir}/status.json")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
