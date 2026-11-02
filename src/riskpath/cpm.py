"""Critical Path Method: forward pass, backward pass and total float.

The core routine works on a 2-D duration matrix of shape (n_tasks, n_scenarios)
so the same code serves the deterministic schedule (one scenario) and the
Monte Carlo engine (10,000 scenarios at once).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskpath.plan import Plan

FLOAT_TOL = 1e-6


@dataclass
class PassResult:
    es: np.ndarray  # (n_tasks, n_scen)
    ef: np.ndarray
    ls: np.ndarray
    lf: np.ndarray
    total_float: np.ndarray
    finish: np.ndarray  # (n_scen,)

    @property
    def critical(self) -> np.ndarray:
        return self.total_float <= FLOAT_TOL


def forward_backward(
    predecessors: list[list[int]], successors: list[list[int]], durations: np.ndarray
) -> PassResult:
    """Vectorised CPM over scenarios. Tasks must be in topological order."""
    d = np.asarray(durations, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n, m = d.shape
    es = np.zeros((n, m))
    ef = np.zeros((n, m))
    for i in range(n):
        preds = predecessors[i]
        if preds:
            es[i] = ef[preds].max(axis=0)
        ef[i] = es[i] + d[i]
    finish = ef.max(axis=0)
    lf = np.empty((n, m))
    ls = np.empty((n, m))
    for i in range(n - 1, -1, -1):
        succ = successors[i]
        lf[i] = ls[succ].min(axis=0) if succ else finish
        ls[i] = lf[i] - d[i]
    tf = ls - es
    # clean floating-point noise so exact-zero float is reported as 0
    tf[np.abs(tf) <= FLOAT_TOL] = 0.0
    return PassResult(es=es, ef=ef, ls=ls, lf=lf, total_float=tf, finish=finish)


@dataclass
class CPMResult:
    ids: list[str]
    names: list[str]
    durations: np.ndarray
    es: np.ndarray
    ef: np.ndarray
    ls: np.ndarray
    lf: np.ndarray
    total_float: np.ndarray
    duration: float

    @property
    def critical(self) -> np.ndarray:
        return self.total_float <= FLOAT_TOL

    @property
    def critical_path(self) -> list[str]:
        """Zero-float tasks, ordered by early start (then topological order)."""
        idx = [i for i in range(len(self.ids)) if self.critical[i]]
        idx.sort(key=lambda i: (self.es[i], i))
        return [self.ids[i] for i in idx]

    def table(self) -> list[dict]:
        return [
            {
                "id": self.ids[i],
                "name": self.names[i],
                "duration": float(self.durations[i]),
                "es": float(self.es[i]),
                "ef": float(self.ef[i]),
                "ls": float(self.ls[i]),
                "lf": float(self.lf[i]),
                "total_float": float(self.total_float[i]),
                "critical": bool(self.critical[i]),
            }
            for i in range(len(self.ids))
        ]


def cpm(plan: Plan, durations: np.ndarray | None = None) -> CPMResult:
    """Deterministic CPM. Defaults to most-likely durations (risk events excluded)."""
    if durations is None:
        durations = np.array(plan.column("most_likely"))
    durations = np.asarray(durations, dtype=float)
    if durations.shape != (len(plan),):
        raise ValueError("durations must have one value per task")
    res = forward_backward(plan.predecessors, plan.successors, durations)
    return CPMResult(
        ids=plan.ids,
        names=[t.name for t in plan.tasks],
        durations=durations,
        es=res.es[:, 0],
        ef=res.ef[:, 0],
        ls=res.ls[:, 0],
        lf=res.lf[:, 0],
        total_float=res.total_float[:, 0],
        duration=float(res.finish[0]),
    )
