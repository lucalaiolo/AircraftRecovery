"""Pricing returns the minimum reduced cost of every aircraft (formulation §7.2)."""

import numpy as np
import pytest

from recovery.generator import generate_instance
from recovery.instance import Costs
from recovery.network import Network
from recovery.pricing import Duals, price, price_on_arcs, reduced_cost

from .conftest import SCALES, micro


def random_duals(net: Network, costs: Costs, rng: np.random.Generator) -> Duals:
    return Duals(
        pi={f.id: float(rng.uniform(-0.5, 1.0) * costs.cancel(f)) for f in net.F},
        sigma={k: float(rng.uniform(-3, 0) * costs.c_pos) for k in net.K},
        mu={s: float(rng.uniform(0, costs.c_pos)) for s in net.S},
    )


def brute_force(net, costs, duals):
    return {k: min(reduced_cost(r, duals) for r in net.enumerate_routes(k, costs)) for k in net.K}


def small_networks(toy):
    yield Network(toy)
    for seed in range(4):
        for kind in ("aog", "closure"):
            yield Network(micro(seed, kind))


def test_pricing_matches_brute_force(toy, costs):
    rng = np.random.default_rng(0)
    for net in small_networks(toy):
        for _ in range(10):
            duals = random_duals(net, costs, rng)
            expected = brute_force(net, costs, duals)
            for result in (price(net, costs, duals), price_on_arcs(net, costs, duals)):
                for k in net.K:
                    assert result.best[k] == pytest.approx(expected[k], abs=1e-6)
                    best = result.routes[k][0]
                    assert reduced_cost(best.route, duals) == pytest.approx(
                        best.reduced_cost, abs=1e-6
                    )
                    assert net.is_path(k, best.copies)


@pytest.mark.parametrize("scale", SCALES)
def test_zero_duals_give_zero_minimum_reduced_cost(scale, costs):
    for kind in ("aog", "closure"):
        net = Network(generate_instance(scale, 0, kind))
        zero = Duals(pi={f.id: 0.0 for f in net.F}, sigma={k: 0.0 for k in net.K}, mu={})
        result = price(net, costs, zero)
        assert all(result.best[k] == 0.0 for k in net.K)
        assert all(result.routes[k][0].route.copies == () for k in net.K)


@pytest.mark.parametrize("scale", ["small", "medium"])
def test_sweep_and_arc_recurrence_agree(scale, costs):
    rng = np.random.default_rng(1)
    for kind in ("aog", "closure"):
        net = Network(generate_instance(scale, 2, kind))
        for _ in range(3):
            duals = random_duals(net, costs, rng)
            a, b = price(net, costs, duals), price_on_arcs(net, costs, duals)
            for k in net.K:
                assert a.best[k] == pytest.approx(b.best[k], abs=1e-6)


def test_several_routes_per_aircraft_are_distinct_and_sorted(toy, costs):
    rng = np.random.default_rng(2)
    net = Network(micro(1, "closure"))
    duals = random_duals(net, costs, rng)
    result = price(net, costs, duals, max_routes=4)
    for k in net.K:
        found = result.routes[k]
        assert len({r.route.key for r in found}) == len(found)
        rcs = [r.reduced_cost for r in found]
        assert rcs == sorted(rcs) and rcs[0] == result.best[k]
        for r in found:
            assert reduced_cost(r.route, duals) == pytest.approx(r.reduced_cost, abs=1e-6)
