import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from riskpath import ai_pm
from riskpath.pipeline import RunConfig, run
from riskpath.sample import write_sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="module")
def pipeline_summary(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    paths = write_sample(d / "examples")
    s = run(
        RunConfig(
            config=paths["config"],
            overrides={"iterations": 3000},
            reports_dir=d / "reports",
            img_dir=d / "docs" / "img",
        )
    )
    return d, s


def test_pipeline_writes_outputs(pipeline_summary):
    d, s = pipeline_summary
    data = json.loads((d / "reports" / "summary.json").read_text())
    assert data["n_tasks"] == 25 and data["settings"]["iterations"] == 3000
    sim = data["simulation"]
    assert data["deterministic"]["duration_days"] <= sim["p50_days"] <= sim["p80_days"] <= sim["p95_days"]
    assert data["margin"]["recommended_days"] >= data["margin"]["raw_days"]
    assert 0.0 <= data["target"]["probability"] <= 1.0
    for name in ("s_curve", "histogram", "tornado", "criticality", "gantt"):
        assert (d / "docs" / "img" / f"{name}.png").stat().st_size > 10_000
    md = (d / "reports" / "schedule_risk_report.md").read_text()
    assert "P80" in md and "../docs/img/s_curve.png" in md


def _roadmap():
    return {
        "milestones": [
            {"id": "M1", "name": "Engine", "due": "2026-01-10", "status": "done"},
            {"id": "M2", "name": "Import", "due": "2026-02-01", "status": "in_progress"},
        ],
        "backlog": [
            {"id": "B1", "title": "core", "priority": "P0", "status": "done", "owner": "Lauren", "depends_on": [], "milestone": "M1"},
            {"id": "B2", "title": "parser", "priority": "P1", "status": "done", "owner": "Lauren", "depends_on": ["B1"], "milestone": "M2"},
            {"id": "B3", "title": "links", "priority": "P2", "status": "todo", "owner": "Lauren", "depends_on": ["B2"], "milestone": "M2"},
            {"id": "B4", "title": "calendars", "priority": "P1", "status": "todo", "owner": "AI PM", "depends_on": ["B3"], "milestone": "M2"},
            {"id": "B5", "title": "docs", "priority": "P2", "status": "todo", "owner": "AI PM", "depends_on": [], "milestone": "M2"},
            {"id": "B6", "title": "ui", "priority": "P3", "status": "todo", "owner": "Lauren", "depends_on": ["B1", "B2"], "milestone": "M2"},
        ],
    }


RISKS = {
    "risks": [
        {"id": "R1", "risk": "a", "probability": 2, "impact": 3, "owner": "x", "mitigation": "m", "status": "open"},
        {"id": "R2", "risk": "b", "probability": 4, "impact": 4, "owner": "x", "mitigation": "m", "status": "open"},
        {"id": "R3", "risk": "c", "probability": 5, "impact": 5, "owner": "x", "mitigation": "m", "status": "closed"},
    ]
}


def test_ai_pm_rules_are_deterministic():
    roadmap = _roadmap()
    # before M2 due: no overdue, no open P0, R2 scores 16 -> AMBER
    s = ai_pm.build_status(roadmap, RISKS, None, date(2026, 1, 20), use_llm=False)
    assert s["health"]["rag"] == "AMBER" and "R2" in s["health"]["reason"]
    assert [r["id"] for r in s["risks"]] == ["R3", "R2", "R1"]
    assert [r["heat"] for r in s["risks"]] == ["Critical", "High", "Low"]
    # Next actions: unblocked todo only (B4 waits on B3), priority then deps satisfied
    assert [a["id"] for a in s["next_actions"]] == ["B3", "B5", "B6"]
    assert any(b["id"] == "B4" for b in s["blocked"])
    m2 = next(m for m in s["milestones"] if m["id"] == "M2")
    assert m2["pct_done"] == 20.0 and not m2["overdue"]
    # M2 three weeks overdue -> RED
    late = ai_pm.build_status(roadmap, RISKS, None, date(2026, 2, 22), use_llm=False)
    assert late["health"]["rag"] == "RED" and "M2" in late["health"]["reason"]
    # same inputs, same output
    again = ai_pm.build_status(roadmap, RISKS, None, date(2026, 1, 20), use_llm=False)
    assert again == s
    assert s["executive_summary_source"] == "deterministic template"


def test_ai_pm_reads_pipeline_summary_and_writes_status(tmp_path, pipeline_summary):
    d, summary = pipeline_summary
    proj = tmp_path / "project"
    shutil.copytree(ROOT / "project", proj, ignore=shutil.ignore_patterns("STATUS.md", "status.json"))
    status = ai_pm.run_pm(proj, d / "reports" / "summary.json", today="2026-11-02", use_llm=True)
    assert status["executive_summary_source"] == "deterministic template"  # no key -> no network
    assert 3 <= len(status["data"]) <= 5
    assert summary["simulation"]["p80_date"] in " ".join(status["data"])
    md = (proj / "STATUS.md").read_text()
    for heading in ("Overall health", "Milestones", "Next 3 actions", "Risk register", "What the data says"):
        assert heading in md
    assert json.loads((proj / "status.json").read_text())["health"]["rag"] in {"GREEN", "AMBER", "RED"}


def test_llm_failure_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(*a, **k):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(ai_pm.urllib.request, "urlopen", boom)
    s = ai_pm.build_status(_roadmap(), RISKS, None, date(2026, 1, 20), use_llm=True)
    assert s["executive_summary_source"] == "deterministic template"
    assert s["executive_summary"].startswith("Overall health is AMBER")
