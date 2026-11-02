import json

import numpy as np
import pytest
import yaml

from riskpath.cpm import cpm
from riskpath.plan import Plan, PlanError, load_plan
from riskpath.sample import write_sample

from conftest import make_task


def test_cpm_hand_computed_network(mini_plan):
    r = cpm(mini_plan)
    assert r.duration == 8
    assert list(r.es) == [0, 3, 3, 7]
    assert list(r.ef) == [3, 5, 7, 8]
    assert list(r.ls) == [0, 5, 3, 7]
    assert list(r.lf) == [3, 7, 7, 8]
    assert list(r.total_float) == [0, 2, 0, 0]
    assert r.critical_path == ["A", "C", "D"]


def test_cpm_custom_durations_shift_critical_path(mini_plan):
    # Make B longer than C: path flips to A-B-D, C gets float.
    r = cpm(mini_plan, np.array([3.0, 6.0, 4.0, 1.0]))
    assert r.duration == 10
    assert r.critical_path == ["A", "B", "D"]
    assert r.total_float[mini_plan.index("C")] == 2


def test_plan_is_topologically_sorted_regardless_of_input_order():
    plan = Plan([make_task("D", ("B", "C")), make_task("C", ("A",)), make_task("B", ("A",)), make_task("A")])
    pos = {tid: i for i, tid in enumerate(plan.ids)}
    for t in plan.tasks:
        for d in t.depends_on:
            assert pos[d] < pos[t.id]


@pytest.mark.parametrize(
    "tasks, message",
    [
        ([make_task("A", ("B",)), make_task("B", ("A",))], "cycle"),
        ([make_task("A", ("Z",))], "unknown"),
        ([make_task("A"), make_task("A")], "duplicate"),
        ([make_task("A", (), 5, 3, 8)], "optimistic <= most_likely"),
        ([make_task("A", (), 1, 2, 3, p=1.5, days=2)], "risk_event_prob"),
    ],
)
def test_plan_validation_errors(tasks, message):
    with pytest.raises(PlanError, match=message):
        Plan(tasks)


def test_csv_yaml_json_loaders_agree(tmp_path):
    paths = write_sample(tmp_path)
    csv_plan = load_plan(paths["csv"])
    yaml_plan = load_plan(paths["yaml"])
    assert csv_plan.tasks == yaml_plan.tasks
    assert len(csv_plan) == 25
    assert yaml_plan.settings["rho"] == 0.3

    # JSON (list form) round-trip of the same records
    recs = csv_plan.to_frame().to_dict(orient="records")
    jpath = tmp_path / "plan.json"
    jpath.write_text(json.dumps(recs))
    assert load_plan(jpath).tasks == csv_plan.tasks

    mini = load_plan(paths["mini"])
    assert cpm(mini).duration == 8


def test_yaml_plan_with_semicolon_string_dependencies(tmp_path):
    p = tmp_path / "p.yml"
    p.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {"id": "A", "optimistic": 1, "most_likely": 2, "pessimistic": 3},
                    {"id": "B", "depends_on": "A", "optimistic": 1, "most_likely": 1, "pessimistic": 1},
                    {"id": "C", "depends_on": "A; B", "optimistic": 2, "most_likely": 2, "pessimistic": 2},
                ]
            }
        )
    )
    plan = load_plan(p)
    assert plan.task("C").depends_on == ("A", "B")
    assert cpm(plan).duration == 5
