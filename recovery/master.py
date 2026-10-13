"""Restricted master problem (formulation §5) over a pool of route columns.

Rows: cover per flight (= 1), one route per aircraft (= 1), positioning per station (>= b_s).
Columns: routes, cancellations z_f and shortfalls y_s.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from recovery.network import Route
from recovery.pricing import Duals
from recovery.solvers import ColumnSpec, make_model

logger = logging.getLogger(__name__)

GUROBI_LIMIT = 2000  # variables and constraints allowed by the pip license


@dataclass(frozen=True)
class LPSolution:
    objective: float  # z_RMP
    duals: Duals
    seconds: float


@dataclass(frozen=True)
class MIPSolution:
    routes: dict[str, Route]  # the route chosen for each aircraft
    cancelled: list[str]
    objective: float  # cost of the plan, recomputed from the chosen columns: z_PB
    solver_objective: float
    bound: float  # the solver's MIP bound over this pool only, not a bound for the problem
    status: str
    seconds: float


class Master:
    def __init__(
        self,
        flights: Sequence[str],
        cancel_cost: Mapping[str, float],
        aircraft: Sequence[str],
        b: Mapping[str, int],
        c_pos: float,
        *,
        backend: str = "gurobi",
        tol: float = 1e-9,
        threads: int = 1,
    ) -> None:
        self.flights, self.aircraft = list(flights), list(aircraft)
        self.b, self.c_pos = dict(b), c_pos
        self.cancel_cost = dict(cancel_cost)
        self.backend = backend
        m = self.model = make_model(backend, tol=tol, threads=threads)
        self.cover_row = {f: m.add_row("=", 1.0) for f in self.flights}
        self.one_row = {k: m.add_row("=", 1.0) for k in self.aircraft}
        self.pos_row = {s: m.add_row(">=", float(v)) for s, v in self.b.items()}
        self.spec: dict[int, ColumnSpec] = {}
        self.upper: dict[int, float] = {}  # current upper bound of every column
        # No explicit upper bounds on z_f and y_s: they hold at every optimum anyway, and without
        # them the RMP duals satisfy pi_f <= C_f and mu_s <= c_pos (formulation §9).
        self.z_col = dict(
            zip(
                self.flights,
                self._add(
                    [ColumnSpec(self.cancel_cost[f], [r], [1.0]) for f, r in self.cover_row.items()]
                ),
                strict=True,
            )
        )
        self.y_col = dict(
            zip(
                self.pos_row,
                self._add([ColumnSpec(c_pos, [r], [1.0]) for r in self.pos_row.values()]),
                strict=True,
            )
        )
        self.route_of: dict[int, Route] = {}
        self.col_of: dict[tuple, int] = {}
        self.protected: set[int] = set()
        self.nonbasic_for: dict[int, int] = {}  # consecutive LP solves a column stayed non-basic

    def _add(self, specs: list[ColumnSpec]) -> list[int]:
        ids = self.model.add_cols(specs)
        self.spec.update(zip(ids, specs, strict=True))
        self.upper.update((i, s.ub) for i, s in zip(ids, specs, strict=True))
        return ids

    def __contains__(self, key: tuple) -> bool:
        return key in self.col_of

    @property
    def n_routes(self) -> int:
        return len(self.route_of)

    @property
    def n_vars(self) -> int:
        return self.model.n_cols

    @property
    def n_rows(self) -> int:
        return self.model.n_rows

    def gurobi_route_cap(self, margin: int = 50) -> int:
        """Route columns that fit the size-limited Gurobi license next to the z_f and y_s."""
        return GUROBI_LIMIT - len(self.flights) - len(self.pos_row) - margin

    def add_routes(self, routes: Sequence[Route], *, protected: bool = False) -> list[int]:
        specs = []
        for r in routes:
            if r.key in self.col_of:
                raise ValueError(f"route already in the pool: {r.label()}")
            rows = [self.cover_row[f] for f in r.flights] + [self.one_row[r.aircraft]]
            if r.end in self.pos_row:
                rows.append(self.pos_row[r.end])
            specs.append(ColumnSpec(r.cost, rows, [1.0] * len(rows)))
        ids = self._add(specs)
        for i, r in zip(ids, routes, strict=True):
            self.route_of[i] = r
            self.col_of[r.key] = i
            self.nonbasic_for[i] = 0
            if protected:
                self.protected.add(i)
        return ids

    def remove_routes(self, ids: Sequence[int]) -> list[Route]:
        if any(i in self.protected for i in ids):
            raise ValueError("protected columns (the empty routes) are never removed")
        self.model.remove_cols(ids)
        gone = []
        for i in ids:
            r = self.route_of.pop(i)
            del self.col_of[r.key], self.nonbasic_for[i], self.spec[i], self.upper[i]
            gone.append(r)
        return gone

    def solve_lp(self) -> LPSolution:
        res = self.model.solve()
        if res.status != "optimal":
            raise RuntimeError(f"restricted master LP not solved to optimality: {res.status}")
        y = self.model.duals()
        duals = Duals(
            pi={f: float(y[r]) for f, r in self.cover_row.items()},
            sigma={k: float(y[r]) for k, r in self.one_row.items()},
            mu={s: float(y[r]) for s, r in self.pos_row.items()},
        )
        ids = list(self.route_of)
        for i, is_basic in zip(ids, self.model.basic(ids), strict=True):
            self.nonbasic_for[i] = 0 if is_basic else self.nonbasic_for[i] + 1
        return LPSolution(res.objective, duals, res.seconds)

    def dual_objective(self, duals: Duals) -> float:
        """sum_f pi_f + sum_k sigma_k + sum_s b_s mu_s (equals z_RMP at the RMP's own duals)."""
        return (
            sum(duals.pi.values())
            + sum(duals.sigma.values())
            + sum(self.b[s] * duals.mu[s] for s in self.pos_row)
        )

    def lagrangian_bound(self, duals: Duals, best_rc: Mapping[str, float]) -> float:
        """L(pi, mu) of formulation §9, given the exact minimum reduced cost of every aircraft."""
        return (
            self.dual_objective(duals)
            + sum(best_rc.values())
            + sum(min(0.0, self.cancel_cost[f] - duals.pi[f]) for f in self.flights)
            + sum(self.b[s] * min(0.0, self.c_pos - duals.mu[s]) for s in self.pos_row)
        )

    def route_values(self) -> dict[int, float]:
        ids = list(self.route_of)
        return dict(zip(ids, self.model.values(ids).tolist(), strict=True))

    def route_reduced_costs(self) -> dict[int, float]:
        ids = list(self.route_of)
        return dict(zip(ids, self.model.reduced_costs(ids).tolist(), strict=True))

    def purge(
        self, *, threshold: float, min_age: int, room: int = 0, cap: int | None = None
    ) -> list[Route]:
        """Drop route columns non-basic for `min_age` LP solves with reduced cost above
        `threshold`, then more non-basic ones (largest reduced cost first) if `room` new columns
        would overflow `cap`. Empty routes are never dropped.
        """
        rc = self.route_reduced_costs()
        loose = [
            i
            for i in self.route_of
            if i not in self.protected and self.nonbasic_for[i] > 0  # non-basic in the last LP
        ]
        old = [i for i in loose if self.nonbasic_for[i] >= min_age and rc[i] > threshold]
        drop = set(old)
        if cap is not None and self.n_routes - len(drop) + room > cap:
            rest = sorted((i for i in loose if i not in drop), key=lambda i: -rc[i])
            need = self.n_routes - len(drop) + room - cap
            drop.update(rest[:need])
        if not drop:
            return []
        gone = self.remove_routes(sorted(drop))
        logger.info(
            "purged %d route columns (%d stale, %d for the pool cap)",
            len(gone),
            len(old),
            len(gone) - len(old),
        )
        return gone

    def is_fractional(self, tol: float = 1e-6) -> bool:
        """Does the current LP solution have a route column strictly between 0 and 1?"""
        v = np.array(list(self.route_values().values()))
        return bool(np.any((v > tol) & (v < 1 - tol)))

    def solve_mip(
        self, *, time_limit: float | None = None, mip_gap: float | None = None
    ) -> MIPSolution:
        """Price-and-branch: lambda and z binary over the current pool (formulation §10)."""
        binaries = list(self.route_of) + list(self.z_col.values())
        upper = [self.upper[i] for i in binaries]
        self.model.set_bounds(binaries, [0.0] * len(binaries), [min(u, 1.0) for u in upper])
        self.model.set_integer(binaries, True)
        try:
            res = self.model.solve(time_limit=time_limit, mip_gap=mip_gap)
            if not res.has_solution:
                raise RuntimeError(f"price-and-branch MIP found no solution: {res.status}")
            lam = self.route_values()
            z = self.model.values(list(self.z_col.values()))
        finally:
            self.model.set_integer(binaries, False)
            self.model.set_bounds(binaries, [0.0] * len(binaries), upper)
        chosen = {self.route_of[i].aircraft: self.route_of[i] for i, x in lam.items() if x > 0.5}
        cancelled = [f for f, x in zip(self.z_col, z, strict=True) if x > 0.5]
        return MIPSolution(
            routes=chosen,
            cancelled=cancelled,
            objective=self.plan_cost(chosen, cancelled),
            solver_objective=res.objective,
            bound=res.bound,
            status=res.status,
            seconds=res.seconds,
        )

    def plan_cost(self, routes: Mapping[str, Route], cancelled: Sequence[str]) -> float:
        ends = Counter(r.end for r in routes.values())
        shortfall = sum(max(0, b - ends[s]) for s, b in self.b.items())
        return (
            sum(r.cost for r in routes.values())
            + sum(self.cancel_cost[f] for f in cancelled)
            + self.c_pos * shortfall
        )
