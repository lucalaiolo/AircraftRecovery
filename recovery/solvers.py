"""Gurobi and HiGHS behind one small interface.

Columns are added to the existing model and re-solved warm. Both backends return row duals y with
reduced costs c - A^T y (tests/test_solvers.py). Gurobi's size-limit error 10010 is re-raised as
LicenseLimitError.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

INF = math.inf
LICENSE_ERRNO = 10010
BACKENDS = ("gurobi", "highs")


class LicenseLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class SolveResult:
    status: str  # "optimal", "time_limit", "infeasible", "unbounded" or "other"
    objective: float  # inf when no solution is available
    bound: float  # MIP dual bound (the objective for an LP)
    seconds: float

    @property
    def has_solution(self) -> bool:
        return math.isfinite(self.objective)


@dataclass(frozen=True)
class ColumnSpec:
    cost: float
    rows: Sequence[int]
    coefs: Sequence[float]
    lb: float = 0.0
    ub: float = INF


class Model(ABC):
    """A minimization model with stable integer column ids."""

    def __init__(self) -> None:
        self.n_rows = 0
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    @abstractmethod
    def add_row(self, sense: str, rhs: float) -> int:
        """An empty row with `sense` in {"=", ">=", "<="}; columns fill it later."""

    @abstractmethod
    def add_cols(self, cols: Sequence[ColumnSpec]) -> list[int]: ...

    def add_col(self, cost: float, rows: Sequence[int], coefs: Sequence[float], lb=0.0, ub=INF):
        return self.add_cols([ColumnSpec(cost, rows, coefs, lb, ub)])[0]

    def add_rows(self, senses: Sequence[str], rhs: Sequence[float]) -> list[int]:
        return [self.add_row(sense, b) for sense, b in zip(senses, rhs, strict=True)]

    def add_cols_csc(
        self,
        cost: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        start: np.ndarray,
        index: np.ndarray,
        value: np.ndarray,
    ) -> list[int]:
        """Columns in compressed sparse column form: column j has entries
        index[start[j]:start[j + 1]] with coefficients value[...]; `start` has n + 1 entries."""
        return self.add_cols(
            [
                ColumnSpec(
                    float(cost[j]),
                    index[start[j] : start[j + 1]].tolist(),
                    value[start[j] : start[j + 1]].tolist(),
                    float(lb[j]),
                    float(ub[j]),
                )
                for j in range(len(cost))
            ]
        )

    @abstractmethod
    def remove_cols(self, ids: Iterable[int]) -> None: ...

    @abstractmethod
    def set_bounds(self, ids: Sequence[int], lb: Sequence[float], ub: Sequence[float]) -> None: ...

    @abstractmethod
    def set_integer(self, ids: Sequence[int], integer: bool) -> None: ...

    @abstractmethod
    def solve(
        self, *, time_limit: float | None = None, mip_gap: float | None = None
    ) -> SolveResult:
        """Solve; the limits apply to this call only."""

    @abstractmethod
    def values(self, ids: Sequence[int]) -> np.ndarray: ...

    @abstractmethod
    def reduced_costs(self, ids: Sequence[int]) -> np.ndarray: ...

    @abstractmethod
    def duals(self) -> np.ndarray:
        """Row duals in row order."""

    @abstractmethod
    def basic(self, ids: Sequence[int]) -> np.ndarray: ...

    @property
    @abstractmethod
    def n_cols(self) -> int: ...


@lru_cache(maxsize=1)
def _gurobi_env():
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


class GurobiModel(Model):
    def __init__(self, *, tol: float = 1e-9, threads: int = 1) -> None:
        import gurobipy as gp

        super().__init__()
        self._gp = gp
        self.m = gp.Model(env=_gurobi_env())
        self.m.Params.OptimalityTol = tol
        self.m.Params.FeasibilityTol = tol
        self.m.Params.Threads = threads
        # Primal simplex: adding columns keeps the basis primal feasible, so the re-solve starts
        # from it. Gurobi's automatic choice tails off badly in column generation here.
        self.m.Params.Method = 0
        self._rows: list = []
        self._vars: dict[int, object] = {}

    def add_row(self, sense: str, rhs: float) -> int:
        gp = self._gp
        s = {"=": gp.GRB.EQUAL, ">=": gp.GRB.GREATER_EQUAL, "<=": gp.GRB.LESS_EQUAL}[sense]
        self._rows.append(self.m.addLConstr(gp.LinExpr(), s, rhs))
        self.n_rows += 1
        return self.n_rows - 1

    def add_cols(self, cols: Sequence[ColumnSpec]) -> list[int]:
        gp = self._gp
        ids = []
        for c in cols:
            column = gp.Column(list(c.coefs), [self._rows[r] for r in c.rows])
            i = self._new_id()
            self._vars[i] = self.m.addVar(lb=c.lb, ub=c.ub, obj=c.cost, column=column)
            ids.append(i)
        return ids

    def remove_cols(self, ids: Iterable[int]) -> None:
        for i in ids:
            self.m.remove(self._vars.pop(i))

    def set_bounds(self, ids, lb, ub) -> None:
        vs = [self._vars[i] for i in ids]
        self.m.setAttr("LB", vs, list(lb))
        self.m.setAttr("UB", vs, list(ub))

    def set_integer(self, ids, integer: bool) -> None:
        vs = [self._vars[i] for i in ids]
        vtype = self._gp.GRB.INTEGER if integer else self._gp.GRB.CONTINUOUS
        self.m.setAttr("VType", vs, [vtype] * len(vs))

    def solve(self, *, time_limit=None, mip_gap=None) -> SolveResult:
        gp = self._gp
        self.m.Params.TimeLimit = INF if time_limit is None else max(time_limit, 0.0)
        if mip_gap is not None:
            self.m.Params.MIPGap = mip_gap
        start = time.perf_counter()
        try:
            self.m.optimize()
        except gp.GurobiError as err:
            if err.errno == LICENSE_ERRNO:
                self.m.update()
                raise LicenseLimitError(
                    f"Gurobi: model too large for the size-limited license ({self.m.NumVars} "
                    f"variables, {self.m.NumConstrs} constraints; the limit is 2000 each). "
                    "Rerun with --solver highs."
                ) from err
            raise
        seconds = time.perf_counter() - start
        st = self.m.Status
        status = {
            gp.GRB.OPTIMAL: "optimal",
            gp.GRB.TIME_LIMIT: "time_limit",
            gp.GRB.INFEASIBLE: "infeasible",
            gp.GRB.UNBOUNDED: "unbounded",
            gp.GRB.INF_OR_UNBD: "infeasible",
        }.get(st, "other")
        objective = self.m.ObjVal if self.m.SolCount > 0 else INF
        bound = self.m.ObjBound if self.m.IsMIP else objective
        return SolveResult(status, objective, bound, seconds)

    def values(self, ids):
        return np.array(self.m.getAttr("X", [self._vars[i] for i in ids]), dtype=float)

    def reduced_costs(self, ids):
        return np.array(self.m.getAttr("RC", [self._vars[i] for i in ids]), dtype=float)

    def duals(self):
        return np.array(self.m.getAttr("Pi", self._rows), dtype=float)

    def basic(self, ids):
        return np.array(self.m.getAttr("VBasis", [self._vars[i] for i in ids])) == 0

    @property
    def n_cols(self) -> int:
        return len(self._vars)


class HighsModel(Model):
    def __init__(self, *, tol: float = 1e-9, threads: int = 1) -> None:
        import highspy

        super().__init__()
        self._hs = highspy
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("primal_feasibility_tolerance", tol)
        self.h.setOptionValue("dual_feasibility_tolerance", tol)
        self.h.setOptionValue("threads", threads)
        self._order: list[int] = []  # column ids in HiGHS column order
        self._pos: dict[int, int] = {}

    def add_row(self, sense: str, rhs: float) -> int:
        inf = self._hs.kHighsInf
        lo, up = {"=": (rhs, rhs), ">=": (rhs, inf), "<=": (-inf, rhs)}[sense]
        self.h.addRow(lo, up, 0, np.zeros(0, dtype=np.int32), np.zeros(0))
        self.n_rows += 1
        return self.n_rows - 1

    def add_cols(self, cols: Sequence[ColumnSpec]) -> list[int]:
        if not cols:
            return []
        sizes = [len(c.rows) for c in cols]
        return self.add_cols_csc(
            np.array([c.cost for c in cols], dtype=float),
            np.array([c.lb for c in cols], dtype=float),
            np.array([c.ub for c in cols], dtype=float),
            np.concatenate([[0], np.cumsum(sizes)]),
            np.array([r for c in cols for r in c.rows], dtype=np.int64),
            np.array([a for c in cols for a in c.coefs], dtype=float),
        )

    def add_rows(self, senses: Sequence[str], rhs: Sequence[float]) -> list[int]:
        inf = self._hs.kHighsInf
        rhs = np.asarray(rhs, dtype=float)
        sense = np.asarray(senses)
        lower = np.where(sense == "<=", -inf, rhs)
        upper = np.where(sense == ">=", inf, rhs)
        n = len(rhs)
        self.h.addRows(
            n,
            lower,
            upper,
            0,
            np.zeros(n, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0),
        )
        self.n_rows += n
        return list(range(self.n_rows - n, self.n_rows))

    def add_cols_csc(self, cost, lb, ub, start, index, value) -> list[int]:
        n = len(cost)
        if n == 0:
            return []
        inf = self._hs.kHighsInf
        self.h.addCols(
            n,
            np.asarray(cost, dtype=float),
            np.maximum(np.asarray(lb, dtype=float), -inf),
            np.minimum(np.asarray(ub, dtype=float), inf),
            int(start[-1]),
            np.asarray(start[:-1], dtype=np.int32),
            np.asarray(index, dtype=np.int32),
            np.asarray(value, dtype=float),
        )
        ids = []
        for _ in range(n):
            i = self._new_id()
            self._pos[i] = len(self._order)
            self._order.append(i)
            ids.append(i)
        return ids

    def remove_cols(self, ids: Iterable[int]) -> None:
        gone = set(ids)
        if not gone:
            return
        idx = np.array(sorted(self._pos[i] for i in gone), dtype=np.int32)
        self.h.deleteCols(len(idx), idx)
        self._order = [i for i in self._order if i not in gone]
        self._pos = {i: p for p, i in enumerate(self._order)}

    def _idx(self, ids) -> np.ndarray:
        return np.array([self._pos[i] for i in ids], dtype=np.int32)

    def set_bounds(self, ids, lb, ub) -> None:
        if len(ids):
            inf = self._hs.kHighsInf
            lo = np.maximum(np.asarray(lb, dtype=float), -inf)
            up = np.minimum(np.asarray(ub, dtype=float), inf)
            self.h.changeColsBounds(len(ids), self._idx(ids), lo, up)

    def set_integer(self, ids, integer: bool) -> None:
        if len(ids):
            kind = self._hs.HighsVarType.kInteger if integer else self._hs.HighsVarType.kContinuous
            self.h.changeColsIntegrality(len(ids), self._idx(ids), np.array([kind] * len(ids)))

    def solve(self, *, time_limit=None, mip_gap=None) -> SolveResult:
        h, hs = self.h, self._hs
        h.setOptionValue("time_limit", INF if time_limit is None else max(time_limit, 0.0))
        if mip_gap is not None:
            h.setOptionValue("mip_rel_gap", mip_gap)
        start = time.perf_counter()
        h.run()
        seconds = time.perf_counter() - start
        st = h.getModelStatus()
        status = {
            hs.HighsModelStatus.kOptimal: "optimal",
            hs.HighsModelStatus.kTimeLimit: "time_limit",
            hs.HighsModelStatus.kInfeasible: "infeasible",
            hs.HighsModelStatus.kUnbounded: "unbounded",
            hs.HighsModelStatus.kUnboundedOrInfeasible: "infeasible",
        }.get(st, "other")
        info = h.getInfo()
        is_mip = h.getLp().integrality_ and any(
            t != hs.HighsVarType.kContinuous for t in h.getLp().integrality_
        )
        has = info.primal_solution_status == 2  # kSolutionStatusFeasible
        objective = info.objective_function_value if has else INF
        bound = info.mip_dual_bound if is_mip else objective
        return SolveResult(status, objective, bound, seconds)

    def values(self, ids):
        return np.asarray(self.h.getSolution().col_value, dtype=float)[self._idx(ids)]

    def reduced_costs(self, ids):
        return np.asarray(self.h.getSolution().col_dual, dtype=float)[self._idx(ids)]

    def duals(self):
        return np.asarray(self.h.getSolution().row_dual, dtype=float)

    def basic(self, ids):
        status = self.h.getBasis().col_status
        kb = self._hs.HighsBasisStatus.kBasic
        return np.array([status[p] == kb for p in self._idx(ids)], dtype=bool)

    @property
    def n_cols(self) -> int:
        return len(self._order)


def make_model(backend: str, *, tol: float = 1e-9, threads: int = 1) -> Model:
    if backend == "gurobi":
        return GurobiModel(tol=tol, threads=threads)
    if backend == "highs":
        return HighsModel(tol=tol, threads=threads)
    raise ValueError(f"unknown solver backend {backend!r}, expected one of {BACKENDS}")
