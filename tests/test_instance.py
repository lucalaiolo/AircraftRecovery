import dataclasses
import math

import pytest

from recovery.generator import generate_instance
from recovery.instance import (
    AOG,
    Closure,
    Flight,
    NoDisruption,
    RecoveryPlan,
    elementarity_cap,
    instance_from_dict,
    instance_to_dict,
    load_instance,
    load_plan,
    preprocess,
    save_instance,
    save_plan,
)


def test_json_round_trip(toy, tmp_path):
    for inst in (
        toy,
        generate_instance("small", 3, "aog"),
        generate_instance("small", 3, "closure"),
    ):
        path = tmp_path / f"{inst.name}.json"
        save_instance(inst, path)
        assert load_instance(path) == inst
        assert instance_from_dict(instance_to_dict(inst)) == inst


def test_plan_round_trip(tmp_path):
    plan = RecoveryPlan(
        routes={"A": [("f1", 30), ("f4", 30)], "B": []},
        cancelled=["f3", "f2"],
        objective=123.5,
        meta={"solver": "test", "costs": {"c_pos": 1.0}},
    )
    save_plan(plan, tmp_path / "plan.json")
    back = load_plan(tmp_path / "plan.json")
    assert back.routes == plan.routes
    assert sorted(back.cancelled) == sorted(plan.cancelled)
    assert back.objective == plan.objective
    assert back.meta == plan.meta


@pytest.mark.parametrize(
    "change, message",
    [
        ({"b": {"MXP": 2}}, "sum to |K|"),
        ({"b": {"XXX": 1}}, "unknown station"),
        ({"d_max": 45}, "multiple"),
        ({"d_max": 210}, "elementarity cap"),
        ({"TAT": 0}, "TAT"),
        ({"disruption": AOG("A", 600, 700)}, "in the air"),  # f0 is airborne 09:15-10:45
        ({"disruption": AOG("Z", 700, 800)}, "unknown"),
        ({"disruption": Closure("MXP", 630, 700)}, "lands inside the closure"),
        ({"disruption": Closure("MXP", 600, 600)}, "ta < tb"),
    ],
)
def test_invalid_instances_are_rejected(toy, change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(toy, **change)


def test_invalid_flight_is_rejected(toy):
    bad = Flight("f9", "MXP", "MXP", 700, 60, 100, "A")
    with pytest.raises(ValueError, match="origin = destination"):
        dataclasses.replace(toy, flights=toy.flights + (bad,))


def test_elementarity_cap_without_pairs_is_infinite():
    flights = [Flight("a", "X", "Y", 0, 60, 1, "k"), Flight("b", "X", "Z", 0, 60, 1, "k")]
    assert math.isinf(elementarity_cap(flights, 40))


def test_ready_state_cases(toy):
    # No frozen flight: start-of-day station, ready at t0.
    early = preprocess(dataclasses.replace(toy, disruption=NoDisruption(500)))
    assert early.ready_station == {"A": "FCO"} and early.ready_time == {"A": 500}
    assert early.frozen == () and early.F[0] == "f0"
    # Landed less than TAT before t0: on the ground but not yet free.
    turning = preprocess(dataclasses.replace(toy, disruption=NoDisruption(650)))
    assert turning.ready_station == {"A": "MXP"} and turning.ready_time == {"A": 685}
    # Last frozen flight f4 lands TRN 14:10, long before t0 = 15:00: free at t0.
    late = preprocess(dataclasses.replace(toy, disruption=NoDisruption(900)))
    assert late.ready_station == {"A": "TRN"} and late.ready_time == {"A": 900}
    assert late.F == ()
    # AOG on the ground at 11:00 (f0 landed 10:45), fixed at 12:00: ready at t2.
    aog = preprocess(dataclasses.replace(toy, disruption=AOG("A", 660, 720)))
    assert aog.ready_time == {"A": 720} and aog.F == ("f1", "f2", "f3", "f4")
    # Fixed before the turnaround ends: the turnaround still binds.
    short = preprocess(dataclasses.replace(toy, disruption=AOG("A", 650, 670)))
    assert short.ready_time == {"A": 685}
