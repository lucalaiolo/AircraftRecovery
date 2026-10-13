"""Greedy + simulated annealing baseline: feasible, never worse than greedy, above the bound."""

import pytest

from recovery.checker import check_plan
from recovery.colgen import CGSettings, price_and_branch
from recovery.generator import generate_instance
from recovery.heuristic import solve_heuristic
from recovery.instance import Costs, load_config

CASES = [
    (s, seed, k) for s in ("tiny", "small", "medium") for seed in (0, 1) for k in ("aog", "closure")
]


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    cfg["heuristic"]["sa_iterations"] = 4000
    return cfg, Costs.from_config(cfg)


@pytest.mark.parametrize("case", CASES)
def test_heuristic_plans_are_feasible_and_above_the_lp_bound(case, setup):
    cfg, costs = setup
    inst = generate_instance(*case)
    h = solve_heuristic(inst, costs, cfg=cfg)
    result = check_plan(inst, h.plan, costs)
    assert result.ok, result.violations
    assert h.cost == pytest.approx(result.cost.total)
    assert h.cost <= h.greedy_cost + 1e-9
    lb = price_and_branch(inst, costs, backend="highs", settings=CGSettings.from_config(cfg)).lb
    assert h.cost >= lb - 1e-6 * max(1.0, lb)


def test_heuristic_is_deterministic(setup):
    cfg, costs = setup
    inst = generate_instance("small", 3, "aog")
    a, b = solve_heuristic(inst, costs, cfg=cfg), solve_heuristic(inst, costs, cfg=cfg)
    assert a.cost == b.cost and a.plan.routes == b.plan.routes
