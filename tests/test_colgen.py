"""Column generation: bounds, convergence, purging and stopping rules."""

import dataclasses
import logging

import numpy as np
import pytest

from recovery.colgen import CGSettings, build_master, column_generation, network_pricer, seed_routes
from recovery.generator import generate_instance
from recovery.network import Network
from recovery.pricing import PricedRoute, PricingResult

BACKENDS = ["gurobi", "highs"]
CASES = [("tiny", 0, "aog"), ("tiny", 1, "closure"), ("small", 0, "aog"), ("small", 2, "closure")]


def run(case, backend="highs", **overrides):
    from recovery.instance import Costs

    costs = Costs.from_config()
    net = Network(generate_instance(*case))
    master = build_master(net, costs, backend=backend)
    settings = CGSettings.from_config(**overrides)
    return net, costs, column_generation(master, network_pricer(net, costs), settings)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_converges_with_valid_bounds(case, backend):
    net, costs, cg = run(case, backend)
    assert cg.converged and cg.lb == pytest.approx(cg.z_rmp)
    z_lp = cg.z_rmp
    tol = 1e-6 * max(1.0, abs(z_lp))
    for it in cg.log:
        assert it.lagrangian <= z_lp + tol  # every Lagrangian bound is a lower bound
        assert it.lb <= it.z_rmp + tol
    z = [it.z_rmp for it in cg.log]
    assert all(b <= a + tol for a, b in zip(z, z[1:], strict=False))  # purging never hurts z_RMP
    # Dual feasibility of the final pool: every route column prices out nonnegative.
    rc = cg.master.route_reduced_costs()
    assert min(rc.values()) >= -1e-6


@pytest.mark.parametrize("case", CASES[:3])
def test_smoothing_and_solver_do_not_change_z_lp(case):
    values = [
        run(case, backend, smoothing=alpha)[2].z_rmp for backend in BACKENDS for alpha in (0.0, 0.8)
    ]
    assert np.ptp(values) <= 1e-6 * max(values)


def test_pool_cap_and_purging_keep_the_empty_routes():
    case = ("small", 0, "aog")
    _, _, free = run(case)
    net, costs, cg = run(case, pool_cap=60, purge_age=2, purge_threshold=0.0)
    assert cg.converged and cg.z_rmp == pytest.approx(free.z_rmp)
    assert cg.purged > 0
    assert max(it.pool for it in cg.log) <= 60
    empty = {r.aircraft for r in cg.master.route_of.values() if not r.copies}
    assert empty == set(net.K)


def test_early_stop_reports_the_best_lagrangian_bound():
    case = ("small", 0, "aog")
    _, _, full = run(case, smoothing=0.0)
    _, _, early = run(case, smoothing=0.0, gap_tol=0.5)
    assert early.stop == "gap" and not early.converged
    assert early.lb == max(it.lagrangian for it in early.log)
    assert early.lb <= full.z_rmp + 1e-6
    assert (early.z_rmp - early.lb) / early.z_rmp <= 0.5


def test_seeded_rotations_are_routes():
    from recovery.instance import Costs

    costs = Costs.from_config()
    for kind in ("aog", "closure"):
        net = Network(generate_instance("medium", 3, kind))
        for r in seed_routes(net, costs):
            assert net.is_path(r.aircraft, net.copies_of(r))
            assert all(net.F[net.f_index[f]].orig == r.aircraft for f in r.flights)


def test_a_pool_column_priced_negative_counts_as_converged(caplog):
    from recovery.instance import Costs

    costs = Costs.from_config()
    net = Network(generate_instance("tiny", 0, "aog"))
    master = build_master(net, costs, backend="highs")
    stay = {r.aircraft: r for r in master.route_of.values() if not r.copies}

    def stubborn(duals):  # returns a route the pool already has, with a negative reduced cost
        routes = {k: [PricedRoute(stay[k], -1.0, ())] for k in net.K}
        return PricingResult({k: -1.0 for k in net.K}, routes)

    with caplog.at_level(logging.WARNING, logger="recovery.colgen"):
        cg = column_generation(master, stubborn, CGSettings(smoothing=0.0))
    assert cg.stop == "converged" and len(cg.log) == 1
    assert "already in the pool" in caplog.text


def test_time_limit_stops_early():
    _, _, cg = run(("medium", 0, "closure"), time_limit=1e-9)
    assert cg.stop == "time" and len(cg.log) == 1
    assert cg.lb == cg.log[0].lb


def test_settings_from_config_and_overrides():
    s = CGSettings.from_config(eps=1e-5)
    assert s.eps == 1e-5 and dataclasses.replace(s, smoothing=0.0).smoothing == 0.0
