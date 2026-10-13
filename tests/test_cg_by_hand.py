"""examples/cg_by_hand.py reproduces the README table on both solvers."""

import sys
from pathlib import Path

import pytest

from recovery.master import Master
from recovery.pricing import Duals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import cg_by_hand  # noqa: E402

BACKENDS = ["gurobi", "highs"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cg_by_hand_expected_output(backend):
    rows = cg_by_hand.run(backend)
    assert len(rows) == 3
    assert [r["z_rmp"] for r in rows] == pytest.approx([200, 110, 30])
    # The duals of iterations 0 and 1 are unique; those of iteration 2 are not and are not asserted.
    assert rows[0]["pi"] == pytest.approx((100, 100)) and rows[0]["sigma"] == pytest.approx(
        (0, 0, 0)
    )
    assert rows[1]["pi"] == pytest.approx((10, 100)) and rows[1]["sigma"] == pytest.approx(
        (0, 0, 0)
    )
    assert rows[0]["added"] == ["P→f1", "Q→f1"]
    assert rows[1]["added"] == ["P→f2", "Q→f2"]
    assert rows[2]["added"] == []
    assert "none: stop, LP optimum 30" in cg_by_hand.table(rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_marginal_value_of_an_aircraft(backend):
    # Formulation §6.4: without P the LP optimum is 110 ...
    assert cg_by_hand.run(backend, ["A", "Q"])[-1]["z_rmp"] == pytest.approx(110)
    # ... while sigma_P = -5 belongs to a valid set of final duals with value 30.
    master = Master(
        ["f1", "f2"], {"f1": 100.0, "f2": 100.0}, ["A", "P", "Q"], {}, 1.0, backend=backend
    )
    master.add_routes(cg_by_hand.ROUTES)
    duals = Duals(pi={"f1": 15.0, "f2": 25.0}, sigma={"A": 0.0, "P": -5.0, "Q": -5.0}, mu={})
    assert master.solve_lp().objective == pytest.approx(30)
    assert master.dual_objective(duals) == pytest.approx(30)
    for r in cg_by_hand.ROUTES:  # every dual constraint holds: c_r - sum pi - sigma >= 0
        assert r.cost - sum(duals.pi[f] for f in r.flights) - duals.sigma[r.aircraft] >= 0
    assert all(duals.pi[f] <= 100 for f in duals.pi)
