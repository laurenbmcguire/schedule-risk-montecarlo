import numpy as np
from scipy import stats

from riskpath import analysis
from riskpath.cpm import cpm
from riskpath.distributions import pert_mean, pert_params, sample_pert
from riskpath.montecarlo import simulate
from riskpath.plan import Plan
from riskpath.sample import SAMPLE_TASKS, COLUMNS
from riskpath.plan import plan_from_records

from conftest import make_task


def sample_plan():
    return plan_from_records([dict(zip(COLUMNS, row)) for row in SAMPLE_TASKS])


def test_pert_beta_moments_and_bounds():
    a, m, b = 10.0, 15.0, 30.0
    alpha, beta = pert_params(a, m, b)
    assert np.isclose(alpha, 1 + 4 * 5 / 20) and np.isclose(beta, 1 + 4 * 15 / 20)
    x = sample_pert([a], [m], [b], 200_000, np.random.default_rng(0))[0]
    assert x.min() >= a and x.max() <= b
    assert abs(x.mean() - pert_mean(a, m, b)) < 0.05  # analytic mean = 16.667
    # analytic median of the scaled beta
    median = a + (b - a) * stats.beta.median(alpha, beta)
    assert abs(np.median(x) - median) < 0.05


def test_zero_variance_simulation_matches_cpm(mini_plan):
    fixed = Plan([make_task(t.id, t.depends_on, t.most_likely) for t in mini_plan.tasks])
    sim = simulate(fixed, iterations=500, seed=1)
    assert np.allclose(sim.finish, cpm(fixed).duration)
    assert list(sim.criticality_index) == [1.0, 0.0, 1.0, 1.0]


def test_criticality_index_on_competing_paths():
    # Two symmetric parallel paths: each should be critical about half the time.
    plan = Plan(
        [
            make_task("S", (), 1),
            make_task("X", ("S",), 5, 10, 15),
            make_task("Y", ("S",), 5, 10, 15),
            make_task("E", ("X", "Y"), 1),
        ]
    )
    ci = simulate(plan, iterations=20_000, seed=3).criticality_index
    assert ci[plan.index("S")] == 1.0 and ci[plan.index("E")] == 1.0
    assert abs(ci[plan.index("X")] - 0.5) < 0.02
    assert abs(ci[plan.index("X")] + ci[plan.index("Y")] - 1.0) < 1e-9


def test_merge_bias_makes_mean_exceed_pert_mean_cpm():
    plan = Plan([make_task(f"P{i}", (), 5, 10, 20) for i in range(4)] + [make_task("E", [f"P{i}" for i in range(4)], 0)])
    sim = simulate(plan, iterations=20_000, seed=5)
    a, m, b = (np.array(plan.column(c)) for c in ("optimistic", "most_likely", "pessimistic"))
    det_mean = cpm(plan, pert_mean(a, m, b)).duration
    assert analysis.merge_bias(sim.finish, det_mean) > 2.0


def test_monte_carlo_percentiles_converge():
    plan = sample_plan()
    p = [analysis.percentiles(simulate(plan, 20_000, seed=s, rho=0.3).finish) for s in (11, 22, 33)]
    for q in (50, 80, 95):
        vals = np.array([r[q] for r in p])
        # independent seeds agree to within 1.5% of the value
        assert (vals.max() - vals.min()) / vals.mean() < 0.015
    # and the estimate stabilises as iterations grow
    big = analysis.percentiles(simulate(plan, 80_000, seed=99, rho=0.3).finish)
    small = analysis.percentiles(simulate(plan, 10_000, seed=7, rho=0.3).finish)
    assert abs(big[80] - small[80]) / big[80] < 0.01


def test_simulation_is_deterministic_for_fixed_seed():
    plan = sample_plan()
    a = simulate(plan, 2_000, seed=42, rho=0.3)
    b = simulate(plan, 2_000, seed=42, rho=0.3)
    assert np.array_equal(a.finish, b.finish)


def test_correlation_widens_the_spread():
    chain = Plan([make_task("T0", (), 8, 10, 16)] + [make_task(f"T{i}", (f"T{i-1}",), 8, 10, 16) for i in range(1, 10)])
    spread = {}
    for rho in (0.0, 0.8):
        f = simulate(chain, 20_000, seed=4, rho=rho).finish
        spread[rho] = np.percentile(f, 95) - np.percentile(f, 5)
    assert spread[0.8] > 2.0 * spread[0.0]
    # marginals unchanged: mean of each task stays at the PERT mean
    d = simulate(chain, 20_000, seed=4, rho=0.8).durations
    assert np.allclose(d.mean(axis=1), pert_mean(8, 10, 16), atol=0.08)


def test_risk_events_fire_at_their_probability():
    plan = Plan([make_task("A", (), 10, p=1.0, days=5), make_task("B", ("A",), 10, p=0.3, days=4), make_task("C", ("B",), 10)])
    sim = simulate(plan, 20_000, seed=8)
    assert np.allclose(sim.durations[0], 15.0)
    assert abs(sim.risk_hit_rate[1] - 0.3) < 0.015
    assert sim.risk_hit_rate[2] == 0.0
    no_risk = simulate(plan, 20_000, seed=8, include_risks=False)
    assert np.allclose(no_risk.finish, 30.0)


def test_sensitivity_ranks_the_volatile_task_first():
    plan = Plan(
        [
            make_task("calm", (), 9, 10, 11),
            make_task("wild", ("calm",), 5, 10, 40),
            make_task("fixed", ("wild",), 3),
        ]
    )
    sim = simulate(plan, 10_000, seed=2)
    rho = analysis.spearman_sensitivity(sim.durations, sim.finish)
    assert np.argmax(rho) == plan.index("wild")
    assert rho[plan.index("wild")] > 0.95
    assert rho[plan.index("fixed")] == 0.0
