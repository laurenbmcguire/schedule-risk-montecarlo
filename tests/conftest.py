import pytest

from riskpath.plan import Task, Plan


def make_task(tid, deps=(), a=1.0, m=None, b=None, p=0.0, days=0.0, name=None):
    m = a if m is None else m
    b = m if b is None else b
    return Task(tid, name or tid, tuple(deps), float(a), float(m), float(b), p, days)


@pytest.fixture
def mini_plan() -> Plan:
    """Hand-computable network.

        A(3) -> B(2) -> D(1)
        A(3) -> C(4) -> D(1)

    ES/EF: A 0-3, B 3-5, C 3-7, D 7-8. Project = 8.
    LS/LF: A 0-3, B 5-7, C 3-7, D 7-8. Float: B = 2, others 0.
    """
    return Plan(
        [
            make_task("A", (), 2, 3, 6),
            make_task("B", ("A",), 1, 2, 4),
            make_task("C", ("A",), 3, 4, 7),
            make_task("D", ("B", "C"), 1, 1, 2),
        ],
        name="mini",
    )
