"""Vectorised Monte Carlo schedule simulation.

Each iteration draws a PERT-beta duration for every task (optionally
correlated through a Gaussian copula), adds any discrete risk events that
fire, then runs a full CPM forward/backward pass. All iterations are
evaluated at once as (n_tasks, n_iterations) numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskpath.cpm import forward_backward
from riskpath.distributions import sample_pert
from riskpath.plan import Plan


@dataclass
class SimulationResult:
    ids: list[str]
    names: list[str]
    durations: np.ndarray  # (n_tasks, n_iter) sampled durations incl. risk events
    risk_fired: np.ndarray  # (n_tasks, n_iter) bool
    finish: np.ndarray  # (n_iter,) project duration in working days
    critical: np.ndarray  # (n_tasks, n_iter) bool, zero total float in that iteration
    iterations: int
    seed: int
    rho: float
    lam: float

    @property
    def criticality_index(self) -> np.ndarray:
        """Share of iterations in which each task is on a critical path."""
        return self.critical.mean(axis=1)

    @property
    def risk_hit_rate(self) -> np.ndarray:
        return self.risk_fired.mean(axis=1)

    def percentile(self, q) -> np.ndarray | float:
        return np.percentile(self.finish, q)


def simulate(
    plan: Plan,
    iterations: int = 10_000,
    seed: int = 42,
    rho: float = 0.0,
    lam: float = 4.0,
    include_risks: bool = True,
) -> SimulationResult:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    rng = np.random.default_rng(seed)
    a = np.array(plan.column("optimistic"))
    m = np.array(plan.column("most_likely"))
    b = np.array(plan.column("pessimistic"))
    base = sample_pert(a, m, b, iterations, rng, lam=lam, rho=rho)

    p = np.array(plan.column("risk_event_prob"))[:, None]
    days = np.array(plan.column("risk_event_days"))[:, None]
    # draw risk uniforms unconditionally so include_risks=False keeps the same
    # duration stream as include_risks=True (common random numbers)
    u = rng.random(base.shape)
    fired = (u < p) & include_risks
    durations = base + np.where(fired, days, 0.0)

    res = forward_backward(plan.predecessors, plan.successors, durations)
    return SimulationResult(
        ids=plan.ids,
        names=[t.name for t in plan.tasks],
        durations=durations,
        risk_fired=fired,
        finish=res.finish,
        critical=res.critical,
        iterations=iterations,
        seed=seed,
        rho=rho,
        lam=lam,
    )
