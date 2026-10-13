import dataclasses

import numpy as np
import pytest

from recovery.generator import generate_instance
from recovery.instance import Aircraft, Closure, Flight, Instance, NoDisruption
from recovery.network import Network, closure_deletes

from .conftest import SCALES, micro


@pytest.mark.parametrize("scale", SCALES)
def test_no_path_repeats_a_flight_on_generated_networks(scale):
    for seed in range(5):
        for kind in ("aog", "closure", "none"):
            Network(generate_instance(scale, seed, kind)).assert_elementary()


def test_elementarity_check_catches_a_loop_when_the_cap_is_ignored():
    # f A->B and g B->A, 30 min each: cap 30 + 30 + 2*40 = 140. With D_max = 180 an aircraft can
    # fly f+0, g+0 and f+180: the instance is built valid, then D_max is forced past the cap.
    inst = Instance(
        name="loop",
        stations=("A", "B"),
        aircraft=(Aircraft("k", "A"),),
        flights=(Flight("f", "A", "B", 600, 30, 10, "k"), Flight("g", "B", "A", 700, 30, 10, "k")),
        TAT=40,
        b={"A": 1},
        delay_step=30,
        d_max=120,
        disruption=NoDisruption(0),
    )
    assert Network(inst).repeated_flights() == []
    object.__setattr__(inst, "d_max", 180)
    bad = Network(inst).repeated_flights()
    assert ("f", 0) in bad
    with pytest.raises(AssertionError, match="same flight twice"):
        Network(inst).assert_elementary()


def _brute_arcs(net):
    return {
        (u, v)
        for u in range(net.n)
        for v in range(net.n)
        if net.dest[u] == net.origin[v] and net.ready[u] <= net.dep[v]
    }


@pytest.mark.parametrize("kind", ["aog", "closure"])
def test_arcs_follow_the_connection_rule(kind):
    for seed in range(3):
        net = Network(generate_instance("tiny", seed, kind))
        assert set(net.arcs()) == _brute_arcs(net)
        assert net.n_arcs == len(_brute_arcs(net))


def test_reachability_matches_a_search_from_the_source():
    for seed in range(3):
        net = Network(micro(seed, "closure"))
        for ki in range(len(net.K)):
            seen, stack = set(), [int(v) for v in np.flatnonzero(net.start_ok[:, ki])]
            while stack:
                v = stack.pop()
                if v not in seen:
                    seen.add(v)
                    stack.extend(int(w) for w in net.out_arcs[v])
            assert set(np.flatnonzero(net.reachable[:, ki]).tolist()) == seen


def test_closure_deletes_exactly_the_copies_in_x(toy):
    inst = dataclasses.replace(toy, disruption=Closure("TRN", 600, 800))
    net = Network(inst)
    # f2 lands at TRN at 13:00, 13:30, 14:00; f4 at 14:10 and later.
    assert net.deleted == [("f2", 0)]
    assert ("f2", 0) not in net.copy_index and ("f2", 30) in net.copy_index
    for f in net.F:
        for d in net.D:
            assert ((f.id, d) in net.copy_index) != closure_deletes(f, d, inst.disruption)


def test_start_arcs_respect_the_ready_state(toy):
    net = Network(toy)
    starts = {net.label(v) for v in np.flatnonzero(net.start_ok[:, 0])}
    assert starts == {"f1+30", "f1+60", "f2+0", "f2+30", "f2+60"}  # from MXP, at 11:25 or later
