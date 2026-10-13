"""The checker accepts the planned day, rejects hand-built infeasible plans, and recomputes cost."""

import dataclasses

import pytest

from recovery.checker import check_all, check_plan, check_saved_plan, main
from recovery.generator import generate_instance
from recovery.instance import (
    AOG,
    Closure,
    RecoveryPlan,
    planned_rotations,
    preprocess,
    save_instance,
    save_plan,
)

from .conftest import SCALES, TOY_PATH


def plan(routes, cancelled, objective=None):
    return RecoveryPlan(routes=routes, cancelled=list(cancelled), objective=objective)


@pytest.mark.parametrize("scale", SCALES)
def test_planned_schedule_passes_before_any_disruption(scale, costs):
    for seed in range(5):
        inst = generate_instance(scale, seed, "none")
        routes = {k: [(f.id, 0) for f in rot] for k, rot in planned_rotations(inst).items()}
        res = check_plan(inst, plan(routes, [], objective=0.0), costs)
        assert res.ok, res.violations
        assert res.cost.total == 0.0


@pytest.mark.parametrize("kind", ["aog", "closure"])
def test_cancel_everything_is_always_feasible(kind, costs):
    inst = generate_instance("small", 1, kind)
    p = preprocess(inst)
    res = check_plan(inst, plan({}, p.F), costs)
    assert res.ok, res.violations
    expected_cancel = sum(costs.cancel(inst.flight[f]) for f in p.F)
    stays = list(p.ready_station.values())
    expected_short = sum(max(0, inst.b[s] - stays.count(s)) for s in inst.S)
    assert res.cost.cancel == pytest.approx(expected_cancel)
    assert res.cost.positioning == pytest.approx(costs.c_pos * expected_short)
    assert res.cost.delay == res.cost.swap == 0.0


def test_toy_feasible_plans_and_their_cost(toy, costs):
    # f1+30 -> f4+30 ends at TRN: b_MXP = 1 is missed.
    res = check_plan(toy, plan({"A": [("f1", 30), ("f4", 30)]}, ["f2", "f3"]), costs)
    assert res.ok, res.violations
    assert res.end_station == {"A": "TRN"} and res.shortfall["MXP"] == 1
    assert (res.cost.delay, res.cost.swap, res.cost.cancel, res.cost.positioning) == (
        6000.0,
        0.0,
        4000.0,
        5000.0,
    )
    assert res.cost.total == 15000.0
    # f1+30 -> f3+60 ends at MXP: no shortfall.
    res = check_plan(toy, plan({"A": [("f1", 30), ("f3", 60)]}, ["f2", "f4"]), costs)
    assert res.ok and res.cost.total == 3000.0 + 6000.0 + 4000.0
    # Staying on the ground: A is already at MXP.
    res = check_plan(toy, plan({}, ["f1", "f2", "f3", "f4"]), costs)
    assert res.ok and res.end_station == {"A": "MXP"} and res.cost.total == 8000.0


def test_swaps_are_charged(costs):
    inst = generate_instance("tiny", 0, "none")
    rot = planned_rotations(inst)
    # Two aircraft that start at the same station can exchange their whole days.
    a, b = next(
        (a, b)
        for a in inst.K
        for b in inst.K
        if a < b and rot[a] and rot[b] and rot[a][0].origin == rot[b][0].origin
    )
    routes = {k: [(f.id, 0) for f in r] for k, r in rot.items()}
    routes[a], routes[b] = routes[b], routes[a]
    res = check_plan(inst, plan(routes, []), costs)
    assert res.ok, res.violations
    assert res.cost.swap == costs.c_swap * (len(rot[a]) + len(rot[b]))


@pytest.mark.parametrize(
    "routes, cancelled, message",
    [
        ({"A": [("f1", 0)]}, ["f2", "f3", "f4"], "before A is ready at 11:25"),
        ({"A": [("f1", 30), ("f3", 30)]}, ["f2", "f4"], "before f1+30 is turned around at 13:20"),
        ({"A": [("f2", 0), ("f4", 30)]}, ["f1", "f3"], "f2 lands at TRN but f4 leaves FCO"),
        ({"A": [("f3", 60)]}, ["f1", "f2", "f4"], "first flight f3 leaves FCO, but A is at MXP"),
        ({"A": [("f2", 45)]}, ["f1", "f3", "f4"], "not on the grid"),
        ({"A": [("f2", 90)]}, ["f1", "f3", "f4"], "not on the grid"),
        ({"A": [("f1", 30)]}, ["f1", "f2", "f3", "f4"], "f1 is both flown and cancelled"),
        ({"A": [("f1", 30)]}, ["f2", "f4"], "f3 is neither flown nor cancelled"),
        ({"A": [("f1", 30), ("f1", 60)]}, ["f2", "f3", "f4"], "f1 is flown 2 times"),
        ({}, ["f1", "f2", "f3", "f4", "f4"], "f4 is cancelled 2 times"),
        ({}, ["f0", "f1", "f2", "f3", "f4"], "f0 departed before t0 = 10:00"),
        ({"Z": []}, ["f1", "f2", "f3", "f4"], "unknown aircraft Z"),
        ({}, ["f1", "f2", "f3", "f4", "f9"], "unknown flight f9"),
    ],
)
def test_toy_infeasible_plans_fail(toy, costs, routes, cancelled, message):
    res = check_plan(toy, plan(routes, cancelled), costs)
    assert not res.ok
    assert any(message in line for line in res.violations), res.violations


def test_closure_window_is_enforced(toy, costs):
    closed = dataclasses.replace(toy, disruption=Closure("TRN", 600, 800))
    bad = check_plan(closed, plan({"A": [("f2", 0)]}, ["f1", "f3", "f4"]), costs)
    assert any("lands at closed TRN at 13:00" in line for line in bad.violations)
    # Landing exactly at tb is allowed.
    edge = dataclasses.replace(toy, disruption=Closure("TRN", 600, 810))
    assert check_plan(edge, plan({"A": [("f2", 30)]}, ["f1", "f3", "f4"]), costs).ok
    shut = dataclasses.replace(toy, disruption=Closure("MXP", 650, 700))  # after f0 lands
    res = check_plan(shut, plan({"A": [("f1", 30)]}, ["f2", "f3", "f4"]), costs)
    assert any("leaves closed MXP at 11:30" in line for line in res.violations)


def test_aog_window_is_enforced(toy, costs):
    aog = dataclasses.replace(toy, disruption=AOG("A", 660, 720))
    res = check_plan(aog, plan({"A": [("f1", 30)]}, ["f2", "f3", "f4"]), costs)
    assert any("before A is ready at 12:00" in line for line in res.violations)
    assert check_plan(aog, plan({"A": [("f1", 60)]}, ["f2", "f3", "f4"]), costs).ok


def test_reported_objective_must_recompute(toy, costs):
    routes, cancelled = {"A": [("f1", 30), ("f4", 30)]}, ["f2", "f3"]
    assert check_plan(toy, plan(routes, cancelled, objective=15000.0), costs).ok
    res = check_plan(toy, plan(routes, cancelled, objective=14999.0), costs)
    assert any("differs from recomputed cost" in line for line in res.violations)


def test_saved_plan_carries_grid_and_costs(toy, costs, tmp_path):
    p = plan({"A": [("f1", 30), ("f4", 30)]}, ["f2", "f3"], objective=15000.0)
    p.meta = {"delay_step": 30, "d_max": 60, "costs": dataclasses.asdict(costs)}
    save_plan(p, tmp_path / "plan.json")
    assert check_saved_plan(TOY_PATH, tmp_path / "plan.json").ok
    main([str(TOY_PATH), str(tmp_path / "plan.json")])  # exits 0
    p.meta["d_max"] = 0  # same plan on a grid without delays: infeasible
    save_plan(p, tmp_path / "bad.json")
    assert not check_saved_plan(TOY_PATH, tmp_path / "bad.json").ok
    with pytest.raises(SystemExit) as exc:
        main([str(TOY_PATH), str(tmp_path / "bad.json")])
    assert exc.value.code == 1


def test_check_all_counts_bad_saved_plans(toy, tmp_path):
    data, plans = tmp_path / "data", tmp_path / "plans"
    data.mkdir()
    plans.mkdir()
    save_instance(toy, data / "toy_network.json")
    good = RecoveryPlan(
        {"A": [("f1", 30), ("f4", 30)]}, ["f2", "f3"], meta={"instance": "toy_network"}
    )
    bad = RecoveryPlan({"A": [("f1", 0)]}, ["f2", "f3", "f4"], meta={"instance": "toy_network"})
    save_plan(good, plans / "good.json")
    assert check_all(plans, data) == 0
    save_plan(bad, plans / "bad.json")
    assert check_all(plans, data) == 1
