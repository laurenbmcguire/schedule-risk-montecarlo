"""Plan model and loaders (CSV, YAML, JSON).

A plan is a directed acyclic graph of tasks with three-point duration
estimates in working days and an optional discrete risk event per task.
Tasks are stored in a stable topological order so the scheduling engines
can do a single forward and backward sweep.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


class PlanError(ValueError):
    """Raised when a plan is malformed (bad estimates, unknown deps, cycles)."""


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    depends_on: tuple[str, ...]
    optimistic: float
    most_likely: float
    pessimistic: float
    risk_event_prob: float = 0.0
    risk_event_days: float = 0.0


@dataclass
class Plan:
    tasks: list[Task]
    name: str = "Project plan"
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tasks = _validate_and_sort(self.tasks)
        self._index = {t.id: i for i, t in enumerate(self.tasks)}

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def ids(self) -> list[str]:
        return [t.id for t in self.tasks]

    def index(self, task_id: str) -> int:
        return self._index[task_id]

    def task(self, task_id: str) -> Task:
        return self.tasks[self._index[task_id]]

    @property
    def predecessors(self) -> list[list[int]]:
        return [[self._index[d] for d in t.depends_on] for t in self.tasks]

    @property
    def successors(self) -> list[list[int]]:
        succ: list[list[int]] = [[] for _ in self.tasks]
        for i, preds in enumerate(self.predecessors):
            for p in preds:
                succ[p].append(i)
        return succ

    def column(self, attr: str) -> list[float]:
        return [float(getattr(t, attr)) for t in self.tasks]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "id": t.id,
                    "name": t.name,
                    "depends_on": ";".join(t.depends_on),
                    "optimistic": t.optimistic,
                    "most_likely": t.most_likely,
                    "pessimistic": t.pessimistic,
                    "risk_event_prob": t.risk_event_prob,
                    "risk_event_days": t.risk_event_days,
                }
                for t in self.tasks
            ]
        )


# ---------------------------------------------------------------- validation
def _validate_and_sort(tasks: Iterable[Task]) -> list[Task]:
    tasks = list(tasks)
    if not tasks:
        raise PlanError("plan has no tasks")
    seen: dict[str, Task] = {}
    for t in tasks:
        if not t.id:
            raise PlanError("task with empty id")
        if t.id in seen:
            raise PlanError(f"duplicate task id {t.id!r}")
        seen[t.id] = t
        a, m, b = t.optimistic, t.most_likely, t.pessimistic
        if any(math.isnan(x) for x in (a, m, b)):
            raise PlanError(f"task {t.id}: missing duration estimate")
        if a < 0:
            raise PlanError(f"task {t.id}: optimistic duration must be >= 0")
        if not (a <= m <= b):
            raise PlanError(
                f"task {t.id}: estimates must satisfy optimistic <= most_likely <= pessimistic "
                f"(got {a}, {m}, {b})"
            )
        if not (0.0 <= t.risk_event_prob <= 1.0):
            raise PlanError(f"task {t.id}: risk_event_prob must be in [0, 1]")
        if t.risk_event_days < 0:
            raise PlanError(f"task {t.id}: risk_event_days must be >= 0")
    for t in tasks:
        for d in t.depends_on:
            if d == t.id:
                raise PlanError(f"task {t.id} depends on itself")
            if d not in seen:
                raise PlanError(f"task {t.id} depends on unknown task {d!r}")

    # Kahn's algorithm, stable with respect to input order.
    order_pos = {t.id: i for i, t in enumerate(tasks)}
    indeg = {t.id: len(set(t.depends_on)) for t in tasks}
    children: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for d in set(t.depends_on):
            children[d].append(t.id)
    ready = sorted([tid for tid, k in indeg.items() if k == 0], key=order_pos.get)
    out: list[Task] = []
    while ready:
        tid = ready.pop(0)
        out.append(seen[tid])
        for c in children[tid]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort(key=order_pos.get)
    if len(out) != len(tasks):
        stuck = sorted(tid for tid, k in indeg.items() if k > 0)
        raise PlanError(f"dependency cycle detected among tasks: {', '.join(stuck)}")
    return out


# ------------------------------------------------------------------- loading
def _split_deps(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value]
    else:
        parts = [p.strip() for p in str(value).replace(",", ";").split(";")]
    # de-duplicate while keeping order
    return tuple(dict.fromkeys(p for p in parts if p))


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanError(f"not a number: {value!r}") from exc
    if math.isnan(f):
        return default
    return f


def task_from_record(rec: dict[str, Any]) -> Task:
    try:
        tid = str(rec["id"]).strip()
        a = _num(rec["optimistic"], float("nan"))
        m = _num(rec["most_likely"], float("nan"))
        b = _num(rec["pessimistic"], float("nan"))
    except KeyError as exc:
        raise PlanError(f"task record missing field {exc.args[0]!r}: {rec}") from exc
    name = rec.get("name")
    if name is None or (isinstance(name, float) and math.isnan(name)):
        name = tid
    return Task(
        id=tid,
        name=str(name).strip(),
        depends_on=_split_deps(rec.get("depends_on")),
        optimistic=a,
        most_likely=m,
        pessimistic=b,
        risk_event_prob=_num(rec.get("risk_event_prob"), 0.0),
        risk_event_days=_num(rec.get("risk_event_days"), 0.0),
    )


def plan_from_records(
    records: Iterable[dict[str, Any]], name: str = "Project plan", settings: dict | None = None
) -> Plan:
    return Plan(tasks=[task_from_record(r) for r in records], name=name, settings=dict(settings or {}))


def _plan_from_mapping(data: Any, default_name: str) -> Plan:
    if isinstance(data, list):
        return plan_from_records(data, name=default_name)
    if not isinstance(data, dict) or "tasks" not in data:
        raise PlanError("YAML/JSON plan must be a list of tasks or a mapping with a 'tasks' key")
    return plan_from_records(
        data["tasks"], name=str(data.get("name", default_name)), settings=data.get("settings") or {}
    )


def load_plan(path: str | Path) -> Plan:
    """Load a plan from ``.csv``, ``.yaml``/``.yml`` or ``.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    default_name = path.stem.replace("_", " ")
    if suffix == ".csv":
        df = pd.read_csv(path, dtype={"id": str, "name": str, "depends_on": str})
        df.columns = [c.strip() for c in df.columns]
        return plan_from_records(df.to_dict(orient="records"), name=default_name)
    if suffix in (".yaml", ".yml"):
        with path.open("r", encoding="utf-8") as fh:
            return _plan_from_mapping(yaml.safe_load(fh), default_name)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return _plan_from_mapping(json.load(fh), default_name)
    raise PlanError(f"unsupported plan format {suffix!r} (use .csv, .yaml, .yml or .json)")
