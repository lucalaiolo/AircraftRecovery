from itertools import pairwise

import pytest

from recovery.generator import _parse_seeds, choose_d_max, generate_instance
from recovery.instance import (
    AOG,
    Closure,
    elementarity_cap,
    instance_to_dict,
    load_config,
    planned_rotations,
    preprocess,
)

from .conftest import SCALES

SEEDS = range(5)


def test_generator_is_deterministic():
    a = generate_instance("small", 7, "aog")
    b = generate_instance("small", 7, "aog")
    assert instance_to_dict(a) == instance_to_dict(b)
    assert instance_to_dict(a) != instance_to_dict(generate_instance("small", 8, "aog"))


def test_disruptions_share_the_schedule_of_a_seed():
    aog = generate_instance("medium", 2, "aog")
    closure = generate_instance("medium", 2, "closure")
    assert aog.flights == closure.flights and aog.aircraft == closure.aircraft
    assert aog.b == closure.b


@pytest.mark.parametrize("scale", SCALES)
def test_sizes_follow_the_scale_ladder(scale):
    spec = load_config()["scales"][scale]
    for seed in SEEDS:
        inst = generate_instance(scale, seed, "aog")
        assert len(inst.K) == spec["aircraft"]
        assert 0.85 * spec["flights"] <= len(inst.flights) <= spec["flights"]
        assert len(inst.S) <= spec["hubs"] + spec["spokes"]
        assert set(inst.hubs) <= set(inst.S)
        assert sum(inst.b.values()) == len(inst.K)


@pytest.mark.parametrize("scale", SCALES)
def test_planned_rotations_end_where_b_says(scale):
    for seed in SEEDS:
        inst = generate_instance(scale, seed, "none")
        ends = {a.id: a.start_station for a in inst.aircraft}
        for k, rot in planned_rotations(inst).items():
            assert all(f.dest == g.origin for f, g in pairwise(rot))
            if rot:
                assert rot[0].origin == next(a for a in inst.aircraft if a.id == k).start_station
                ends[k] = rot[-1].dest
        for s in inst.S:
            assert inst.b[s] == sum(e == s for e in ends.values())


@pytest.mark.parametrize("scale", SCALES)
def test_aog_is_on_the_ground_and_hurts(scale):
    for seed in SEEDS:
        inst = generate_instance(scale, seed, "aog")
        dis = inst.disruption
        assert isinstance(dis, AOG) and dis.t1 < dis.t2 and inst.t0 == dis.t1
        own = planned_rotations(inst)[dis.aircraft]
        assert not any(f.dep < dis.t1 < f.arr for f in own)
        assert any(dis.t1 <= f.dep < dis.t2 for f in own)
        assert preprocess(inst).ready_time[dis.aircraft] >= dis.t2


@pytest.mark.parametrize("scale", SCALES)
def test_closure_has_no_landing_in_progress_and_hurts(scale):
    min_affected = load_config()["generator"]["closure"]["min_affected"]
    for seed in SEEDS:
        inst = generate_instance(scale, seed, "closure")
        dis = inst.disruption
        assert isinstance(dis, Closure) and dis.ta < dis.tb and inst.t0 == dis.ta
        s, ta, tb = dis.station, dis.ta, dis.tb
        assert not any(f.dep < ta and f.dest == s and ta <= f.arr < tb for f in inst.flights)
        hit = [
            f
            for f in inst.flights
            if f.dep >= ta
            and ((f.origin == s and f.dep < tb) or (f.dest == s and ta <= f.arr < tb))
        ]
        assert len(hit) >= min_affected


def test_default_d_max_rule():
    inst = generate_instance("tiny", 0, "aog")
    F = [inst.flight[f] for f in preprocess(inst).F]
    cap = elementarity_cap(F, inst.TAT)
    assert inst.d_max == min(180, max(d for d in range(0, int(cap), 15) if d < cap))
    # Explicit values at or above the cap are rejected; below it they are kept.
    with pytest.raises(ValueError, match="elementarity cap"):
        choose_d_max(F, inst.TAT, 15, d_max=int(cap) // 15 * 15 + 15)
    if cap % 15 == 0:
        with pytest.raises(ValueError, match="elementarity cap"):
            choose_d_max(F, inst.TAT, 15, d_max=int(cap))
    assert choose_d_max(F, inst.TAT, 15, d_max=60) == 60
    with pytest.raises(ValueError, match="multiple"):
        choose_d_max(F, inst.TAT, 15, d_max=50)


def test_default_d_max_lands_strictly_below_a_binding_cap(toy):
    F = [toy.flight[f] for f in preprocess(toy).F]  # cap 210
    assert choose_d_max(F, toy.TAT, 15, ceiling=300) == 195
    assert choose_d_max(F, toy.TAT, 30, ceiling=300) == 180
    assert choose_d_max(F, toy.TAT, 30) == 180
    assert choose_d_max(F, toy.TAT, 70, ceiling=300) == 140
    with pytest.raises(ValueError):
        generate_instance("tiny", 0, "aog", d_max=600)


def test_parse_seeds():
    assert _parse_seeds("3") == [3]
    assert _parse_seeds("0-3,7") == [0, 1, 2, 3, 7]
