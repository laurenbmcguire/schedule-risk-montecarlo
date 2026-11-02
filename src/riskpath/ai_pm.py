"""AI Project Manager: manages this repository as a project.

Inputs:  project/ROADMAP.yml, project/RISKS.yml, reports/summary.json
Outputs: project/STATUS.md, project/status.json

The deterministic engine always runs. If ANTHROPIC_API_KEY is set, an
optional executive summary is requested from the Anthropic Messages API,
grounded only in the computed facts; any failure falls back to a template.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
OPEN_ITEM = {"todo", "doing"}
CLOSED_RISK = {"closed", "retired"}
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-5"

# ---------------------------------------------------------------- loading


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_summary(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_date(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


# ------------------------------------------------------------ deterministic


def risk_heat(score: int) -> str:
    if score >= 20:
        return "Critical"
    if score >= 15:
        return "High"
    if score >= 8:
        return "Medium"
    return "Low"


def risk_register(risks: list[dict]) -> list[dict]:
    out = []
    for r in risks:
        p, i = int(r["probability"]), int(r["impact"])
        if not (1 <= p <= 5 and 1 <= i <= 5):
            raise ValueError(f"risk {r.get('id')}: probability and impact must be 1-5")
        score = p * i
        out.append({**r, "score": score, "heat": risk_heat(score)})
    out.sort(key=lambda r: (-r["score"], -int(r["impact"]), str(r["id"])))
    return out


def milestone_table(milestones: list[dict], backlog: list[dict], today: date) -> list[dict]:
    rows = []
    for ms in milestones:
        items = [b for b in backlog if b.get("milestone") == ms["id"]]
        done = sum(1 for b in items if b.get("status") == "done")
        if ms.get("status") == "done":
            pct = 100.0
        elif items:
            pct = round(100.0 * done / len(items), 1)
        else:
            pct = 0.0
        due = _as_date(ms["due"])
        overdue = ms.get("status") != "done" and due < today
        rows.append(
            {
                "id": ms["id"],
                "name": ms["name"],
                "due": due.isoformat(),
                "status": ms.get("status", "planned"),
                "pct_done": pct,
                "items_done": done,
                "items_total": len(items),
                "overdue": overdue,
                "days_overdue": (today - due).days if overdue else 0,
                "done_criteria": list(ms.get("done_criteria") or []),
            }
        )
    return rows


def health(milestones: list[dict], backlog: list[dict], risks: list[dict], summary_present: bool) -> dict:
    """Explicit RAG rules, evaluated in order (first match wins):

    RED   - any milestone more than 14 days overdue, or 2+ open P0 items,
            or any open risk scoring >= 20
    AMBER - any overdue milestone, or any open P0 item, or any open risk
            scoring >= 15, or the pipeline summary is missing
    GREEN - otherwise
    """
    overdue = [m for m in milestones if m["overdue"]]
    late = [m for m in overdue if m["days_overdue"] > 14]
    open_p0 = [b for b in backlog if b.get("priority") == "P0" and b.get("status") in OPEN_ITEM]
    open_risks = [r for r in risks if str(r.get("status", "open")).lower() not in CLOSED_RISK]
    crit = [r for r in open_risks if r["score"] >= 20]
    high = [r for r in open_risks if r["score"] >= 15]

    def ids(xs):
        return ", ".join(str(x["id"]) for x in xs)

    if late:
        return {"rag": "RED", "reason": f"milestone(s) more than 14 days overdue: {ids(late)}"}
    if len(open_p0) >= 2:
        return {"rag": "RED", "reason": f"{len(open_p0)} open P0 items: {ids(open_p0)}"}
    if crit:
        return {"rag": "RED", "reason": f"open risk(s) scoring >= 20: {ids(crit)}"}
    if overdue:
        return {"rag": "AMBER", "reason": f"overdue milestone(s): {ids(overdue)}"}
    if open_p0:
        return {"rag": "AMBER", "reason": f"open P0 item: {ids(open_p0)}"}
    if high:
        detail = ", ".join(f"{r['id']} (score {r['score']})" for r in high)
        return {
            "rag": "AMBER",
            "reason": f"no overdue milestones or open P0 items, but open risk(s) scoring >= 15: {detail}",
        }
    if not summary_present:
        return {"rag": "AMBER", "reason": "pipeline summary (reports/summary.json) is missing"}
    return {"rag": "GREEN", "reason": "no overdue milestones, no open P0 items, no open risk scoring >= 15"}


def next_actions(backlog: list[dict], n: int = 3) -> list[dict]:
    """Unblocked todo items, ranked by priority then by number of satisfied dependencies (desc), then id."""
    status = {b["id"]: b.get("status") for b in backlog}
    cands = []
    for b in backlog:
        if b.get("status") != "todo":
            continue
        deps = list(b.get("depends_on") or [])
        if not all(status.get(d) == "done" for d in deps):
            continue
        cands.append({**b, "deps_satisfied": len(deps)})
    cands.sort(key=lambda b: (PRIORITY_RANK.get(b.get("priority"), 9), -b["deps_satisfied"], str(b["id"])))
    return cands[:n]


def blocked_items(backlog: list[dict]) -> list[dict]:
    status = {b["id"]: b.get("status") for b in backlog}
    out = []
    for b in backlog:
        if b.get("status") not in OPEN_ITEM:
            continue
        waiting = [d for d in (b.get("depends_on") or []) if status.get(d) != "done"]
        if waiting:
            out.append({"id": b["id"], "title": b["title"], "waiting_on": waiting})
    return out


def data_bullets(summary: dict | None) -> list[str]:
    """3-5 plain-language findings from reports/summary.json."""
    if not summary:
        return ["No pipeline results found: run `python -m riskpath run` to produce reports/summary.json."]
    det, sim, mar = summary["deterministic"], summary["simulation"], summary["margin"]
    b = [
        f"Deterministic CPM finish is {det['duration_days']:.0f} working days ({det['finish_date']}), "
        f"but only {det['probability_of_meeting']:.1%} of {summary['settings']['iterations']:,} "
        f"simulated runs finish that early.",
        f"P50 / P80 / P95 finish: {sim['p50_date']} / {sim['p80_date']} / {sim['p95_date']} "
        f"({sim['p50_days']:.1f} / {sim['p80_days']:.1f} / {sim['p95_days']:.1f} working days).",
        f"Recommended schedule margin is {mar['recommended_days']} working days "
        f"({mar['pct_of_deterministic']:.1f}% of the deterministic plan), committing to {mar['commit_date']}.",
    ]
    tgt = summary.get("target")
    if tgt:
        b.append(f"Probability of meeting the {tgt['date']} target is {tgt['probability']:.1%}.")
    if summary.get("top_drivers"):
        d0 = summary["top_drivers"][0]
        line = f"Top duration driver is {d0['id']} {d0['name']} (Spearman {d0['sensitivity_spearman']:.2f})"
        hidden = [
            t for t in summary.get("tasks", [])
            if not t["deterministic_critical"] and t["criticality_index"] >= 0.2
        ]
        if hidden:
            h0 = max(hidden, key=lambda t: (t["criticality_index"], t["id"]))
            line += (
                f"; {len(hidden)} task(s) with float in the plan are critical in >= 20% of runs, "
                f"led by {h0['id']} {h0['name']} ({h0['criticality_index']:.0%})"
            )
        b.append(line + ".")
    return b[:5]


def change_log_entry(today: date, rag: dict, summary: dict | None, actions: list[dict]) -> str:
    parts = [f"{today.isoformat()}: health {rag['rag']}."]
    if summary:
        parts.append(
            f"Pipeline recommends {summary['margin']['recommended_days']} working days of schedule margin "
            f"(P80 {summary['simulation']['p80_date']})."
        )
    if actions:
        parts.append(f"Next focus: {actions[0]['id']} {actions[0]['title']}.")
    return " ".join(parts)


def template_summary(facts: dict) -> str:
    h = facts["health"]
    ms = facts["milestones"]
    done = sum(1 for m in ms if m["status"] == "done")
    risks = facts["risks"]
    top_risk = risks[0] if risks else None
    s = (
        f"Overall health is {h['rag']}: {h['reason']}. "
        f"{done} of {len(ms)} milestones are complete. "
    )
    if facts["data"]:
        s += facts["data"][0] + " "
        if len(facts["data"]) > 2:
            s += facts["data"][2] + " "
    if top_risk:
        s += (
            f"The highest-scored open risk is {top_risk['id']} (score {top_risk['score']}, {top_risk['heat']}); "
            f"mitigation: {str(top_risk['mitigation']).rstrip('.')}. "
        )
    if facts["next_actions"]:
        s += "Next actions: " + "; ".join(f"{a['id']} {a['title']}" for a in facts["next_actions"]) + "."
    return s.strip()


# -------------------------------------------------------------------- LLM


def llm_summary(facts: dict, timeout: float = 30.0) -> str | None:
    """Return an LLM executive summary, or None on any problem (no key, network, parse)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    model = os.environ.get("AI_PM_MODEL", DEFAULT_MODEL)
    prompt = (
        "You are the project manager for a software portfolio repository. Write a 90-120 word "
        "executive summary of project status for a busy stakeholder. Use ONLY the facts in the JSON "
        "below; do not invent numbers, dates, people or events. Plain prose, no headings, no bullets.\n\n"
        f"FACTS:\n{json.dumps(facts, indent=2, default=str)}"
    )
    body = json.dumps(
        {"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = "".join(
            blk.get("text", "") for blk in payload.get("content", []) if blk.get("type") == "text"
        ).strip()
        return text or None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError):
        return None
    except Exception:  # never fail the run because of the LLM
        return None


# ----------------------------------------------------------------- output


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out


def render_status_md(status: dict) -> str:
    L: list[str] = []
    w = L.append
    w(f"# Project status: {status['project']}")
    w("")
    w(f"_Generated by the AI PM on {status['date']}. Source: `project/ROADMAP.yml`, `project/RISKS.yml`, "
      f"`reports/summary.json`._")
    w("")
    h = status["health"]
    w(f"**Overall health: {h['rag']}** - {h['reason']}")
    w("")
    w("## Executive summary")
    w("")
    w(status["executive_summary"])
    w("")
    w(f"_Summary source: {status['executive_summary_source']}._")
    w("")
    w("## Milestones")
    w("")
    L.extend(
        _table(
            ["ID", "Milestone", "Due", "Status", "% done", "Items"],
            [
                [m["id"], m["name"], m["due"] + (" (overdue)" if m["overdue"] else ""), m["status"],
                 f"{m['pct_done']:.0f}%", f"{m['items_done']}/{m['items_total']}"]
                for m in status["milestones"]
            ],
        )
    )
    w("")
    w("## Next 3 actions")
    w("")
    L.extend(
        _table(
            ["#", "ID", "Action", "Priority", "Owner", "Depends on (done)"],
            [
                [k + 1, a["id"], a["title"], a["priority"], a["owner"], ", ".join(a.get("depends_on") or []) or "-"]
                for k, a in enumerate(status["next_actions"])
            ],
        )
    )
    w("")
    if status["blocked"]:
        w("Blocked: " + "; ".join(f"{b['id']} waits on {', '.join(b['waiting_on'])}" for b in status["blocked"]) + ".")
        w("")
    w("## Risk register")
    w("")
    L.extend(
        _table(
            ["ID", "Risk", "P", "I", "Score", "Heat", "Owner", "Mitigation", "Status"],
            [
                [r["id"], r["risk"], r["probability"], r["impact"], r["score"], r["heat"], r["owner"],
                 r["mitigation"], r.get("status", "open")]
                for r in status["risks"]
            ],
        )
    )
    w("")
    w("## What the data says")
    w("")
    for b in status["data"]:
        w(f"- {b}")
    w("")
    w("## Decision / change log")
    w("")
    w(f"- {status['change_log']}")
    w("")
    return "\n".join(L)


def build_status(roadmap: dict, risks_doc: dict, summary: dict | None, today: date, use_llm: bool = True) -> dict:
    backlog = list(roadmap.get("backlog") or [])
    milestones = milestone_table(list(roadmap.get("milestones") or []), backlog, today)
    risks = risk_register(list(risks_doc.get("risks") or []))
    rag = health(milestones, backlog, risks, summary is not None)
    actions = next_actions(backlog)
    bullets = data_bullets(summary)
    facts = {
        "project": roadmap.get("project", "riskpath"),
        "date": today.isoformat(),
        "health": rag,
        "milestones": [
            {k: m[k] for k in ("id", "name", "due", "status", "pct_done", "overdue")} for m in milestones
        ],
        "risks": [
            {k: r[k] for k in ("id", "risk", "score", "heat", "mitigation")}
            for r in risks
            if str(r.get("status", "open")).lower() not in CLOSED_RISK
        ][:3],
        "next_actions": [{k: a[k] for k in ("id", "title", "priority", "owner")} for a in actions],
        "data": bullets,
    }
    narrative = llm_summary(facts) if use_llm else None
    source = f"LLM ({os.environ.get('AI_PM_MODEL', DEFAULT_MODEL)})" if narrative else "deterministic template"
    return {
        "project": facts["project"],
        "date": today.isoformat(),
        "health": rag,
        "executive_summary": narrative or template_summary(facts),
        "executive_summary_source": source,
        "milestones": milestones,
        "next_actions": [
            {k: a.get(k) for k in ("id", "title", "priority", "owner", "depends_on", "milestone")} for a in actions
        ],
        "blocked": blocked_items(backlog),
        "risks": risks,
        "data": bullets,
        "change_log": change_log_entry(today, rag, summary, actions),
    }


def run_pm(
    project_dir: Path = Path("project"),
    summary_path: Path = Path("reports/summary.json"),
    today: str | date | None = None,
    use_llm: bool = True,
) -> dict:
    project_dir = Path(project_dir)
    today_d = _as_date(today) if today else date.today()
    status = build_status(
        load_yaml(project_dir / "ROADMAP.yml"),
        load_yaml(project_dir / "RISKS.yml"),
        load_summary(summary_path),
        today_d,
        use_llm=use_llm,
    )
    (project_dir / "STATUS.md").write_text(render_status_md(status), encoding="utf-8")
    with (project_dir / "status.json").open("w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, default=str)
        fh.write("\n")
    return status
