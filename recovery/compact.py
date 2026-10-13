"""Compact arc-flow MILP (formulation §11): one binary x_{k,a} per aircraft k and arc a of G_k.

Only arcs with a tail reachable from src_k are built; x is zero on the others anyway. The model is
far too large for the size-limited Gurobi license, so it runs on HiGHS.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from recovery.instance import Costs, RecoveryPlan
from recovery.network import Network
from recovery.solvers import make_model

START, CONN, SINK, EMPTY = 0, 1, 2, 3


@dataclass(frozen=True)
class CompactSize:
    variables: int
    rows: int
    nonzeros: int


@dataclass
class CompactResult:
    status: str
    objective: float  # inf without a solution; for a MIP, the cost of the extracted plan
    bound: float  # the solver's dual bound (the objective for the LP relaxation)
    plan: RecoveryPlan | None
    relaxed: bool
    size: CompactSize
    build_s: float
    solve_s: float


def _arc_pairs(net: Network) -> tuple[np.ndarray, np.ndarray]:
    """(arc, aircraft) pairs of copy-to-copy arcs with a reachable tail."""
    return np.nonzero(net.reachable[net.in_tail])


def compact_size(net: Network) -> CompactSize:
    reach = net.reachable
    K, F, S = len(net.K), len(net.F), len(net.S)
    n_start = int((net.start_ok & reach).sum())
    n_conn = int(reach[net.in_tail].sum()) if net.n_arcs else 0
    n_sink = int(reach.sum())
    return CompactSize(
        variables=n_start + n_conn + n_sink + K + F + S,
        rows=K + n_sink + F + S,
        nonzeros=3 * (n_start + n_conn) + 2 * (n_sink + K) + F + S,
    )


def solve_compact(
    net: Network,
    costs: Costs,
    *,
    backend: str = "highs",
    relax: bool = False,
    time_limit: float | None = None,
    mip_gap: float = 0.0,
    threads: int = 1,
) -> CompactResult:
    t_build = time.perf_counter()
    inst = net.instance
    K, F, S = len(net.K), len(net.F), len(net.S)
    reach = net.reachable
    arc_cost = net.copy_costs(costs)  # cost of the arc into copy v for aircraft k

    # Rows: source per aircraft, conservation per reachable (copy, aircraft), cover, positioning.
    n_cons = int(reach.sum())
    cons_row = np.full(reach.shape, -1, dtype=np.int64)
    cons_row[reach] = K + np.arange(n_cons)
    cover_row = K + n_cons + net.flight_of  # per copy
    pos_base = K + n_cons + F
    senses = ["="] * (K + n_cons + F) + [">="] * S
    rhs = np.concatenate(
        [np.ones(K), np.zeros(n_cons), np.ones(F), [float(inst.b[s]) for s in net.S]]
    )

    # Variables of each kind, with their rows and coefficients (fixed count per kind).
    sv, sk = np.nonzero(net.start_ok & reach)
    a, ck = _arc_pairs(net)
    cu, cv = net.in_tail[a], net.arc_head[a]
    kv, kk = np.nonzero(reach)
    blocks = [
        # start arc src_k -> v: source row, conservation of v (in), cover of v's flight
        (np.stack([sk, cons_row[sv, sk], cover_row[sv]], 1), [1.0, 1.0, 1.0], arc_cost[sv, sk]),
        # connection arc u -> v: conservation of u (out), of v (in), cover of v's flight
        (
            np.stack([cons_row[cu, ck], cons_row[cv, ck], cover_row[cv]], 1),
            [-1.0, 1.0, 1.0],
            arc_cost[cv, ck],
        ),
        # sink arc v -> snk of dest(v): conservation of v (out), positioning of dest(v)
        (np.stack([cons_row[kv, kk], pos_base + net.dest[kv]], 1), [-1.0, 1.0], np.zeros(len(kv))),
        # empty route src_k -> snk of s_k: source row, positioning of s_k
        (np.stack([np.arange(K), pos_base + net.src_station], 1), [1.0, 1.0], np.zeros(K)),
    ]
    kinds = [
        (np.full(len(sv), START), sk, -np.ones(len(sv), dtype=np.int64), sv),
        (np.full(len(a), CONN), ck, cu, cv),
        (np.full(len(kv), SINK), kk, kv, -np.ones(len(kv), dtype=np.int64)),
        (np.full(K, EMPTY), np.arange(K), -np.ones(K, dtype=np.int64), -np.ones(K, dtype=np.int64)),
    ]
    index = np.concatenate([rows.ravel() for rows, _, _ in blocks])
    value = np.concatenate([np.tile(c, len(rows)) for rows, c, _ in blocks])
    counts = np.concatenate([np.full(len(rows), len(c)) for rows, c, _ in blocks])
    cost = np.concatenate([cst for _, _, cst in blocks])
    n_x = len(cost)
    # z_f and y_s: one entry each.
    index = np.concatenate([index, K + n_cons + np.arange(F), pos_base + np.arange(S)])
    value = np.concatenate([value, np.ones(F + S)])
    counts = np.concatenate([counts, np.ones(F + S, dtype=np.int64)])
    cost = np.concatenate([cost, [costs.cancel(f) for f in net.F], np.full(S, costs.c_pos)]).astype(
        float
    )
    start = np.concatenate([[0], np.cumsum(counts)])
    upper = np.concatenate([np.ones(n_x + F), np.full(S, np.inf)])
    size = CompactSize(len(cost), len(rhs), int(start[-1]))

    model = make_model(backend, threads=threads)
    model.add_rows(senses, rhs)
    ids = model.add_cols_csc(cost, np.zeros(len(cost)), upper, start, index, value)
    if not relax:
        model.set_integer(ids[: n_x + F], True)
    build_s = time.perf_counter() - t_build

    res = model.solve(time_limit=time_limit, mip_gap=None if relax else mip_gap)
    if not res.has_solution:
        return CompactResult(res.status, np.inf, res.bound, None, relax, size, build_s, res.seconds)
    if relax:
        return CompactResult(
            res.status, res.objective, res.objective, None, relax, size, build_s, res.seconds
        )

    x = model.values(ids[:n_x]) > 0.5
    kind, k_of, tail, head = (np.concatenate(parts) for parts in zip(*kinds, strict=True))
    nxt: dict[tuple[int, int], int] = {}  # (aircraft, copy) -> next copy, from chosen arcs
    first: dict[int, int] = {}
    for j in np.flatnonzero(x):
        if kind[j] == START:
            first[int(k_of[j])] = int(head[j])
        elif kind[j] == CONN:
            nxt[int(k_of[j]), int(tail[j])] = int(head[j])
    routes, chosen = {}, {}
    for ki, k in enumerate(net.K):
        path, v = [], first.get(ki)
        while v is not None:
            path.append(v)
            v = nxt.get((ki, v))
        chosen[k] = net.route(k, path, costs)
        routes[k] = list(chosen[k].copies)
    z = model.values(ids[n_x : n_x + F])
    cancelled = sorted(f.id for f, zf in zip(net.F, z, strict=True) if zf > 0.5)
    ends = Counter(r.end for r in chosen.values())
    shortfall = sum(max(0, inst.b[s] - ends[s]) for s in net.S)
    objective = (
        sum(r.cost for r in chosen.values())
        + sum(costs.cancel(net.F[net.f_index[f]]) for f in cancelled)
        + costs.c_pos * shortfall
    )
    plan = RecoveryPlan(
        routes=routes,
        cancelled=cancelled,
        objective=objective,
        meta={
            "instance": inst.name,
            "method": "compact",
            "solver": backend,
            "delay_step": inst.delay_step,
            "d_max": inst.d_max,
            "costs": dataclasses.asdict(costs),
        },
    )
    return CompactResult(res.status, objective, res.bound, plan, relax, size, build_s, res.seconds)
