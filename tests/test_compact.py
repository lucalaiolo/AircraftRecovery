"""Compact arc-flow model against column generation and price-and-branch (Phase 4)."""

import pytest

from recovery.checker import check_plan
from recovery.colgen import CGSettings, price_and_branch
from recovery.compact import compact_size, solve_compact
from recovery.generator import generate_instance
from recovery.instance import Costs
from recovery.network import Network

CASES = [("tiny", s, k) for s in range(3) for k in ("aog", "closure")] + [
    ("small", 0, "aog"),
    ("small", 0, "closure"),
]


@pytest.fixture(scope="module")
def costs():
    return Costs.from_config()


@pytest.mark.parametrize("case", CASES)
def test_lp_bound_compact_optimum_and_price_and_branch_are_ordered(case, costs):
    inst = generate_instance(*case)
    net = Network(inst)
    settings = CGSettings.from_config(gap_tol=0.0)  # full convergence, no early stop
    pb = price_and_branch(inst, costs, backend="highs", settings=settings)
    assert pb.cg.converged
    lp = solve_compact(net, costs, relax=True)
    ip = solve_compact(net, costs, mip_gap=0.0)
    assert ip.status == "optimal"
    tol = len(net.K) * settings.eps + 1e-6 * max(1.0, abs(pb.lb))
    # Same network, so the two LP relaxations coincide (formulation §11).
    assert lp.objective == pytest.approx(pb.lb, abs=tol)
    assert pb.lb <= ip.objective + tol
    assert ip.objective <= pb.z_pb + tol
    for plan in (pb.plan, ip.plan):
        result = check_plan(inst, plan, costs)
        assert result.ok, result.violations


def test_compact_size_matches_the_built_model(costs):
    net = Network(generate_instance("tiny", 1, "closure"))
    size = compact_size(net)
    result = solve_compact(net, costs, relax=True)
    assert result.size == size


def test_compact_gurobi_and_highs_agree_on_a_toy_sized_model(costs, toy):
    net = Network(toy)
    assert compact_size(net).variables < 2000
    values = [solve_compact(net, costs, backend=b).objective for b in ("gurobi", "highs")]
    assert values[0] == pytest.approx(values[1])
