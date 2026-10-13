"""Expected values of the toy network, worked out by hand."""

from recovery.instance import elementarity_cap, hhmm, preprocess
from recovery.network import Network


def test_toy_ready_state_comes_from_preprocessing(toy):
    p = preprocess(toy)
    assert p.ready_station == {"A": "MXP"}
    assert p.ready_time == {"A": 685}
    assert hhmm(p.ready_time["A"]) == "11:25"


def test_toy_flights_and_grid(toy):
    p = preprocess(toy)
    assert p.t0 == 600
    assert p.frozen == ("f0",)
    assert p.F == ("f1", "f2", "f3", "f4")
    assert toy.D == (0, 30, 60)
    assert toy.b == {"MXP": 1, "FCO": 0, "TRN": 0}


def test_toy_elementarity_cap(toy):
    # f3 FCO->MXP followed by f2 MXP->TRN: 70 + 60 + 2*40.
    F = [toy.flight[f] for f in preprocess(toy).F]
    assert elementarity_cap(F, toy.TAT) == 210


def _labels(net, mask):
    return {net.label(v) for v in range(net.n) if mask[v]}


def test_toy_reachable_copies(toy):
    net = Network(toy)
    reach = net.reachable[:, net.k_index["A"]]
    assert _labels(net, reach) == {
        "f1+30",
        "f1+60",
        "f2+0",
        "f2+30",
        "f2+60",
        "f3+60",
        "f4+30",
        "f4+60",
    }
    assert _labels(net, ~reach) == {"f1+0", "f3+0", "f3+30", "f4+0"}


def test_toy_connection_arcs(toy):
    net = Network(toy)
    arcs = [(net.label(u), net.label(v)) for u, v in net.arcs()]
    assert len(arcs) == len(set(arcs)) == 9
    reach = net.reachable[:, net.k_index["A"]]
    assert {(net.label(u), net.label(v)) for u, v in net.arcs() if reach[u]} == {
        ("f1+30", "f3+60"),
        ("f1+30", "f4+30"),
        ("f1+30", "f4+60"),
        ("f1+60", "f4+60"),
    }


def test_toy_has_exactly_ten_routes(toy, costs):
    routes = Network(toy).enumerate_routes("A", costs)
    got = {(tuple(f"{f}+{d}" for f, d in r.copies), r.end) for r in routes}
    assert len(routes) == 10
    assert got == {
        ((), "MXP"),
        (("f1+30",), "FCO"),
        (("f1+60",), "FCO"),
        (("f2+0",), "TRN"),
        (("f2+30",), "TRN"),
        (("f2+60",), "TRN"),
        (("f1+30", "f3+60"), "MXP"),
        (("f1+30", "f4+30"), "TRN"),
        (("f1+30", "f4+60"), "TRN"),
        (("f1+60", "f4+60"), "TRN"),
    }
