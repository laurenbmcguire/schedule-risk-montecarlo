"""Synthetic example plan: a generic 25-task cloud migration program.

The estimates are illustrative, written by hand for this portfolio project.
They do not come from any real program. ``write_sample`` regenerates every
file under ``examples/`` deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

# id, name, depends_on, optimistic, most_likely, pessimistic, risk_prob, risk_days
SAMPLE_TASKS: list[tuple] = [
    ("T01", "Program mobilization and governance", "", 5, 8, 12, 0.0, 0),
    ("T02", "Application discovery and assessment", "T01", 15, 20, 35, 0.0, 0),
    ("T03", "Infrastructure dependency mapping", "T01", 10, 15, 25, 0.20, 8),
    ("T04", "Migration strategy and wave plan", "T02;T03", 8, 10, 18, 0.0, 0),
    ("T05", "Landing zone design", "T01", 10, 12, 20, 0.0, 0),
    ("T06", "Landing zone build (IaC, guardrails)", "T05", 15, 20, 30, 0.25, 10),
    ("T07", "Identity federation and SSO", "T05", 10, 15, 25, 0.30, 10),
    ("T08", "Privileged access and RBAC model", "T07", 5, 8, 14, 0.0, 0),
    ("T09", "Hub-spoke network and IP plan", "T05", 8, 10, 16, 0.0, 0),
    ("T10", "Hybrid connectivity and DNS", "T09", 18, 25, 45, 0.35, 15),
    ("T11", "Security and logging baseline", "T06;T08", 8, 12, 20, 0.0, 0),
    ("T12", "Migration factory tooling and runbooks", "T04;T06", 8, 10, 15, 0.0, 0),
    ("T13", "Data replication setup", "T04;T10", 8, 12, 20, 0.0, 0),
    ("T14", "Pilot migration (2 apps)", "T10;T11;T12", 8, 10, 18, 0.30, 6),
    ("T15", "Wave 1 application migration", "T13;T14", 15, 20, 30, 0.0, 0),
    ("T16", "Wave 1 data migration and validation", "T13;T14", 10, 14, 24, 0.20, 7),
    ("T17", "Wave 2 application migration", "T15", 20, 25, 40, 0.0, 0),
    ("T18", "Wave 2 data migration and validation", "T15;T16", 12, 16, 28, 0.25, 8),
    ("T19", "Wave 3 application migration (legacy)", "T17", 20, 30, 50, 0.30, 12),
    ("T20", "Wave 3 data migration (large databases)", "T17;T18", 15, 20, 35, 0.30, 10),
    ("T21", "Operations readiness and FinOps", "T11", 10, 15, 25, 0.0, 0),
    ("T22", "Performance and DR testing", "T19;T20", 8, 10, 18, 0.0, 0),
    ("T23", "Cutover rehearsal", "T21;T22", 3, 5, 8, 0.20, 5),
    ("T24", "Production cutover", "T23", 2, 3, 5, 0.0, 0),
    ("T25", "Hypercare and handover", "T24", 10, 10, 15, 0.0, 0),
]

SAMPLE_SETTINGS: dict = {
    "start_date": "2027-01-04",
    "target_date": "2027-09-30",
    "holidays": [
        "2027-01-18",
        "2027-02-15",
        "2027-05-31",
        "2027-07-05",
        "2027-09-06",
        "2027-11-25",
        "2027-12-24",
    ],
    "rho": 0.3,
    "iterations": 10000,
    "seed": 42,
    "pert_lambda": 4.0,
}

# Small hand-computable network used in docs and tests.
MINI_TASKS: list[dict] = [
    {"id": "A", "name": "Design", "depends_on": [], "optimistic": 2, "most_likely": 3, "pessimistic": 6},
    {"id": "B", "name": "Build API", "depends_on": ["A"], "optimistic": 1, "most_likely": 2, "pessimistic": 4},
    {"id": "C", "name": "Build data layer", "depends_on": ["A"], "optimistic": 3, "most_likely": 4, "pessimistic": 7},
    {"id": "D", "name": "Integrate and test", "depends_on": ["B", "C"], "optimistic": 1, "most_likely": 1, "pessimistic": 2},
]

COLUMNS = [
    "id",
    "name",
    "depends_on",
    "optimistic",
    "most_likely",
    "pessimistic",
    "risk_event_prob",
    "risk_event_days",
]


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(SAMPLE_TASKS, columns=COLUMNS)


def _records() -> list[dict]:
    out = []
    for row in SAMPLE_TASKS:
        rec = dict(zip(COLUMNS, row))
        rec["depends_on"] = [d for d in rec["depends_on"].split(";") if d]
        if not rec["risk_event_prob"]:
            rec.pop("risk_event_prob")
            rec.pop("risk_event_days")
        out.append(rec)
    return out


def write_sample(examples_dir: str | Path = "examples") -> dict[str, Path]:
    """Write plan.csv, plan.yaml, mini_plan.json and project.yml."""
    d = Path(examples_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": d / "plan.csv",
        "yaml": d / "plan.yaml",
        "mini": d / "mini_plan.json",
        "config": d / "project.yml",
    }
    sample_frame().to_csv(paths["csv"], index=False)
    with paths["yaml"].open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"name": "Cloud migration program", "settings": SAMPLE_SETTINGS, "tasks": _records()},
            fh,
            sort_keys=False,
        )
    with paths["mini"].open("w", encoding="utf-8") as fh:
        json.dump({"name": "Mini network", "tasks": MINI_TASKS}, fh, indent=2)
        fh.write("\n")
    config = {"name": "Cloud migration program", "plan": "plan.csv", **SAMPLE_SETTINGS}
    with paths["config"].open("w", encoding="utf-8") as fh:
        fh.write("# Run configuration for `python -m riskpath run` (synthetic example).\n")
        yaml.safe_dump(config, fh, sort_keys=False)
    return paths
