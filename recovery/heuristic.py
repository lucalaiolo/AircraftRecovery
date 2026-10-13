"""Greedy recovery, then simulated annealing over swap and cancel moves.

Each aircraft flies its sequence as early as the delay grid, its ready time and closures allow,
which is the cheapest choice for a fixed sequence.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from recovery.instance import (
    Closure,
    Costs,
    Flight,
    Instance,
    RecoveryPlan,
    load_config,
    preprocess,
)
from recovery.network import closure_deletes


@dataclass
class HeuristicResult:
    plan: RecoveryPlan
    cost: float
    greedy_cost: float
    seconds: float
    moves_accepted: int


class _Evaluator:
    """Feasibility and cost of aircraft sequences (formulation §1 costs)."""

    def __init__(self, inst: Instance, costs: Costs) -> None:
        prep = preprocess(inst)
        self.inst, self.costs = inst, costs
        self.F = [inst.flight[f] for f in prep.F]
        self.station0, self.ready0 = prep.ready_station, prep.ready_time
        self.D = inst.D
        self.closure = inst.disruption if isinstance(inst.disruption, Closure) else None

    def earliest_delay(self, f: Flight, ready: int) -> int | None:
        """Smallest grid delay that departs at or after `ready` on a copy the closure keeps."""
        for d in self.D:
            if f.dep + d >= ready and not closure_deletes(f, d, self.closure):
                return d
        return None

    def route(self, k: str, seq: list[Flight]) -> tuple[list[tuple[str, int]], float, str] | None:
        """The delays, cost and end station of aircraft k flying `seq`; None if infeasible."""
        station, ready = self.station0[k], self.ready0[k]
        out, cost = [], 0.0
        for f in seq:
            if f.origin != station:
                return None
            d = self.earliest_delay(f, ready)
            if d is None:
                return None
            out.append((f.id, d))
            cost += self.costs.delay(f, d) + self.costs.c_swap * (f.orig != k)
            station, ready = f.dest, f.dep + d + f.blk + self.inst.TAT
        return out, cost, station

    def total(self, routes: dict[str, tuple], cancelled: set[str]) -> float:
        ends = Counter(r[2] for r in routes.values())
        shortfall = sum(max(0, b - ends[s]) for s, b in self.inst.b.items())
        return (
            sum(r[1] for r in routes.values())
            + sum(self.costs.cancel(self.inst.flight[f]) for f in cancelled)
            + self.costs.c_pos * shortfall
        )


def greedy(ev: _Evaluator) -> dict[str, list[Flight]]:
    """Flights in scheduled order: the cheapest aircraft at the origin flies it (the original one on
    ties); cancel when nobody can fly it within D_max or flying costs more than cancelling.
    """
    station, ready = dict(ev.station0), dict(ev.ready0)
    seq: dict[str, list[Flight]] = {k: [] for k in ev.inst.K}
    for f in sorted(ev.F, key=lambda f: (f.dep, f.id)):
        best = None
        for k in ev.inst.K:
            if station[k] != f.origin:
                continue
            d = ev.earliest_delay(f, ready[k])
            if d is None:
                continue
            key = (ev.costs.delay(f, d) + ev.costs.c_swap * (f.orig != k), f.orig != k, d, k)
            if best is None or key < best:
                best = key
        if best is None or best[0] > ev.costs.cancel(f):
            continue  # cancelled
        k, d = best[3], best[2]
        seq[k].append(f)
        station[k], ready[k] = f.dest, f.dep + d + f.blk + ev.inst.TAT
    return seq


def _stations_before(ev: _Evaluator, k: str, seq: list[Flight]) -> list[str]:
    """Station of k before each position of seq, plus the end station."""
    out = [ev.station0[k]]
    out += [f.dest for f in seq]
    return out


def anneal(
    ev: _Evaluator,
    seq: dict[str, list[Flight]],
    *,
    iterations: int,
    t0: float,
    t_end: float,
    seed: int,
) -> tuple[dict[str, list[Flight]], int]:
    rng = np.random.default_rng(seed)
    K = list(ev.inst.K)
    by_id = {f.id: f for f in ev.F}
    routes = {k: ev.route(k, seq[k]) for k in K}
    cancelled = set(by_id) - {f.id for s in seq.values() for f in s}
    current = ev.total(routes, cancelled)
    best = (current, {k: list(s) for k, s in seq.items()})
    alpha = (t_end / t0) ** (1.0 / max(iterations, 1))
    temp, accepted = t0, 0

    def cycles(s: list[Flight], stations: list[str]) -> list[tuple[int, int]]:
        """Slices s[i:j] that start and end at the same station (a round trip or longer)."""
        return [
            (i, j)
            for i in range(len(s))
            for j in range(i + 2, min(len(s), i + 4) + 1)
            if stations[i] == stations[j]
        ]

    for _ in range(iterations):
        temp *= alpha
        move = rng.integers(4)
        new_seq: dict[str, list[Flight]] = {}
        new_cancel = set(cancelled)
        if move == 0:  # exchange the rest of two aircraft's days at a common station
            k1, k2 = rng.choice(K, 2, replace=False)
            st1, st2 = _stations_before(ev, k1, seq[k1]), _stations_before(ev, k2, seq[k2])
            i = int(rng.integers(len(st1)))
            js = [j for j, s in enumerate(st2) if s == st1[i]]
            if not js:
                continue
            j = js[int(rng.integers(len(js)))]
            new_seq[k1] = seq[k1][:i] + seq[k2][j:]
            new_seq[k2] = seq[k2][:j] + seq[k1][i:]
        elif move == 1:  # cancel a round trip
            k = K[int(rng.integers(len(K)))]
            cyc = cycles(seq[k], _stations_before(ev, k, seq[k]))
            if not cyc:
                continue
            i, j = cyc[int(rng.integers(len(cyc)))]
            new_seq[k] = seq[k][:i] + seq[k][j:]
            new_cancel |= {f.id for f in seq[k][i:j]}
        elif move == 2:  # reinstate a cancelled round trip f, g on an aircraft at f's origin
            pool = sorted(cancelled)
            if not pool:
                continue
            f = by_id[pool[int(rng.integers(len(pool)))]]
            backs = [
                by_id[g]
                for g in pool
                if by_id[g].origin == f.dest and by_id[g].dest == f.origin and by_id[g].dep > f.dep
            ]
            if not backs:
                continue
            g = min(backs, key=lambda x: x.dep)
            k = K[int(rng.integers(len(K)))]
            stations = _stations_before(ev, k, seq[k])
            spots = [p for p, s in enumerate(stations) if s == f.origin]
            if not spots:
                continue
            p = spots[int(rng.integers(len(spots)))]
            new_seq[k] = seq[k][:p] + [f, g] + seq[k][p:]
            new_cancel -= {f.id, g.id}
        else:  # move a round trip to another aircraft standing at its start station
            k1, k2 = rng.choice(K, 2, replace=False)
            cyc = cycles(seq[k1], _stations_before(ev, k1, seq[k1]))
            if not cyc:
                continue
            i, j = cyc[int(rng.integers(len(cyc)))]
            trip = seq[k1][i:j]
            stations = _stations_before(ev, k2, seq[k2])
            spots = [p for p, s in enumerate(stations) if s == trip[0].origin]
            if not spots:
                continue
            p = spots[int(rng.integers(len(spots)))]
            new_seq[k1] = seq[k1][:i] + seq[k1][j:]
            new_seq[k2] = seq[k2][:p] + trip + seq[k2][p:]

        new_routes = dict(routes)
        for k, s in new_seq.items():
            r = ev.route(k, s)
            if r is None:
                break
            new_routes[k] = r
        else:
            cost = ev.total(new_routes, new_cancel)
            delta = cost - current
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                seq = {**seq, **new_seq}
                routes, cancelled, current = new_routes, new_cancel, cost
                accepted += 1
                if current < best[0] - 1e-9:
                    best = (current, {k: list(s) for k, s in seq.items()})
    return best[1], accepted


def solve_heuristic(
    inst: Instance, costs: Costs, *, cfg: dict[str, Any] | None = None
) -> HeuristicResult:
    cfg = load_config() if cfg is None else cfg
    h = cfg["heuristic"]
    start = time.perf_counter()
    ev = _Evaluator(inst, costs)
    seq = greedy(ev)
    routes = {k: ev.route(k, s) for k, s in seq.items()}
    flown = {f for r in routes.values() for f, _ in r[0]}
    greedy_cost = ev.total(routes, {f.id for f in ev.F} - flown)
    seq, accepted = anneal(
        ev,
        seq,
        iterations=int(h["sa_iterations"]),
        t0=float(h["sa_t0"]),
        t_end=float(h["sa_t_end"]),
        seed=int(h["seed"]),
    )
    routes = {k: ev.route(k, s) for k, s in seq.items()}
    flown = {f for r in routes.values() for f, _ in r[0]}
    cancelled = sorted({f.id for f in ev.F} - flown)
    cost = ev.total(routes, set(cancelled))
    plan = RecoveryPlan(
        routes={k: r[0] for k, r in routes.items()},
        cancelled=cancelled,
        objective=cost,
        meta={
            "instance": inst.name,
            "method": "greedy+SA",
            "delay_step": inst.delay_step,
            "d_max": inst.d_max,
            "costs": dataclasses.asdict(costs),
        },
    )
    return HeuristicResult(plan, cost, greedy_cost, time.perf_counter() - start, accepted)
