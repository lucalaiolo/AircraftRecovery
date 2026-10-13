"""Solver interface: the dual sign convention and the Gurobi size limit."""

import numpy as np
import pytest

from recovery.colgen import CGSettings, build_master, column_generation, network_pricer
from recovery.generator import generate_instance
from recovery.instance import Costs
from recovery.network import Network
from recovery.solvers import LicenseLimitError, make_model

BACKENDS = ["gurobi", "highs"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduced_costs_are_c_minus_a_transpose_y(backend):
    """Recompute c - A^T y on a restricted master and compare with the solver's reduced costs."""
    costs = Costs.from_config()
    net = Network(generate_instance("small", 1, "closure"))
    master = build_master(net, costs, backend=backend)
    column_generation(
        master, network_pricer(net, costs), CGSettings(smoothing=0.0, max_iter=4)
    )  # stop early: a genuinely restricted master
    master.solve_lp()
    y = master.model.duals()
    ids = list(master.spec)
    recomputed = np.array(
        [
            master.spec[i].cost
            - sum(a * y[r] for r, a in zip(master.spec[i].rows, master.spec[i].coefs, strict=True))
            for i in ids
        ]
    )
    assert np.allclose(recomputed, master.model.reduced_costs(ids), atol=1e-6)
    # Signs: pi_f <= C_f (z_f columns), 0 <= mu_s <= c_pos (y_s columns), sigma_k <= -mu_{s_k}.
    duals = master.solve_lp().duals
    assert all(duals.pi[f.id] <= costs.cancel(f) + 1e-6 for f in net.F)
    assert all(-1e-6 <= m <= costs.c_pos + 1e-6 for m in duals.mu.values())
    for ki, k in enumerate(net.K):
        assert duals.sigma[k] <= -duals.mu[net.S[net.src_station[ki]]] + 1e-6


def test_gurobi_size_limit_is_reported_clearly():
    m = make_model("gurobi")
    row = m.add_row(">=", 1.0)
    for _ in range(2100):
        m.add_col(1.0, [row], [1.0])
    with pytest.raises(LicenseLimitError, match="--solver highs"):
        m.solve()
