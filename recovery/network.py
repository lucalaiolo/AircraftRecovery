"""Connection networks (formulation §3) and routes (§4).

Copy-to-copy arcs do not depend on the aircraft, so one Network serves every aircraft; only the
source and the swap costs differ. Copies are numbered by departure time, a topological order of
every G_k.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from recovery.instance import Closure, Costs, Flight, Instance, preprocess

# Sweep events. At equal times ready events go first: a connection may leave when a turn ends.
SOURCE, READY, DEPART = 0, 1, 2


@dataclass(frozen=True)
class Route:
    """A column of the master: one aircraft's day as (flight, delay) pairs in flying order.

    The empty route has no copies. `end` is None when the master has no positioning rows.
    """

    aircraft: str
    copies: tuple[tuple[str, int], ...]
    cost: float
    end: str | None = None

    @property
    def key(self) -> tuple[str, tuple[tuple[str, int], ...]]:
        return (self.aircraft, self.copies)

    @property
    def flights(self) -> tuple[str, ...]:
        return tuple(f for f, _ in self.copies)

    def label(self) -> str:
        if not self.copies:
            return f"{self.aircraft}: stay"
        return f"{self.aircraft}: " + " -> ".join(f"{f}+{d}" for f, d in self.copies)


def closure_deletes(f: Flight, d: int, closure: Closure | None) -> bool:
    """Is copy (f, d) in the deleted set X of formulation §3?"""
    if closure is None:
        return False
    s, ta, tb = closure.station, closure.ta, closure.tb
    off, on = f.dep + d, f.dep + d + f.blk
    return (f.origin == s and ta <= off < tb) or (f.dest == s and ta <= on < tb)


class Network:
    """The shared copy graph plus each aircraft's source, for one instance and delay grid."""

    def __init__(self, inst: Instance) -> None:
        prep = preprocess(inst)
        self.instance = inst
        self.prep = prep
        self.TAT = inst.TAT
        self.D = inst.D
        self.F: list[Flight] = [inst.flight[f] for f in prep.F]
        self.K: list[str] = list(inst.K)
        self.S: list[str] = list(inst.S)
        self.f_index = {f.id: i for i, f in enumerate(self.F)}
        self.k_index = {k: i for i, k in enumerate(self.K)}
        self.s_index = {s: i for i, s in enumerate(self.S)}

        closure = inst.disruption if isinstance(inst.disruption, Closure) else None
        self.deleted: list[tuple[str, int]] = []
        rows: list[tuple[int, int, int]] = []  # (departure, flight index, delay)
        for i, f in enumerate(self.F):
            for d in self.D:
                if closure_deletes(f, d, closure):
                    self.deleted.append((f.id, d))
                else:
                    rows.append((f.dep + d, i, d))
        rows.sort()
        self.n = len(rows)
        self.flight_of = np.array([i for _, i, _ in rows], dtype=np.int64)
        self.delay = np.array([d for _, _, d in rows], dtype=np.int64)
        self.dep = np.array([t for t, _, _ in rows], dtype=np.int64)
        blk = np.array([f.blk for f in self.F], dtype=np.int64)
        self.arr = self.dep + blk[self.flight_of] if self.n else self.dep.copy()
        self.ready = self.arr + self.TAT
        org = np.array([self.s_index[f.origin] for f in self.F], dtype=np.int64)
        dst = np.array([self.s_index[f.dest] for f in self.F], dtype=np.int64)
        self.origin = org[self.flight_of] if self.n else self.flight_of.copy()
        self.dest = dst[self.flight_of] if self.n else self.flight_of.copy()
        self.copy_index = {
            (self.F[i].id, d): v for v, (_, i, d) in enumerate(rows)
        }  # (flight id, delay) -> copy

        # Source of each aircraft: ready station s_k and ready time rho_k (formulation §2.2).
        self.src_station = np.array(
            [self.s_index[prep.ready_station[k]] for k in self.K], dtype=np.int64
        )
        self.src_time = np.array([prep.ready_time[k] for k in self.K], dtype=np.int64)

        times = np.concatenate([self.src_time, self.ready, self.dep])
        kinds = np.concatenate(
            [np.full(len(self.K), SOURCE), np.full(self.n, READY), np.full(self.n, DEPART)]
        )
        idx = np.concatenate([np.arange(len(self.K)), np.arange(self.n), np.arange(self.n)])
        order = np.lexsort((idx, kinds, times))
        self.event_kind = kinds[order]
        self.event_idx = idx[order]

    @cached_property
    def start_ok(self) -> np.ndarray:
        """(copies x aircraft): does arc src_k -> (f, d) exist? origin_f = s_k, dep >= rho_k."""
        return (self.origin[:, None] == self.src_station[None, :]) & (
            self.dep[:, None] >= self.src_time[None, :]
        )

    @cached_property
    def swapped(self) -> np.ndarray:
        """(copies x aircraft): is k a swap for this copy's flight, orig_f != k?"""
        orig = np.array([self.k_index[f.orig] for f in self.F], dtype=np.int64)
        return orig[self.flight_of][:, None] != np.arange(len(self.K))[None, :]

    def copy_costs(self, costs: Costs) -> np.ndarray:
        """(copies x aircraft): c_delay pax_f d + c_swap [orig_f != k]"""
        pax = np.array([f.pax for f in self.F], dtype=float)
        delay_cost = costs.c_delay * pax[self.flight_of] * self.delay
        return delay_cost[:, None] + costs.c_swap * self.swapped

    @cached_property
    def reachable(self) -> np.ndarray:
        """(copies x aircraft): is copy v on some path from src_k? One forward sweep."""
        K = len(self.K)
        reach = np.zeros((self.n, K), dtype=bool)
        open_at = np.zeros((len(self.S), K), dtype=bool)  # some path is ready at station s
        for kind, i in zip(self.event_kind, self.event_idx, strict=True):
            if kind == DEPART:
                reach[i] = open_at[self.origin[i]]
            elif kind == READY:
                open_at[self.dest[i]] |= reach[i]
            else:
                open_at[self.src_station[i], i] = True
        return reach

    @cached_property
    def in_ptr(self) -> np.ndarray:
        return self._arcs[0]

    @cached_property
    def in_tail(self) -> np.ndarray:
        """Copy-to-copy arc tails by head: the in-arcs of v are in_tail[in_ptr[v]:in_ptr[v + 1]]."""
        return self._arcs[1]

    @property
    def n_arcs(self) -> int:
        return int(self.in_ptr[-1])

    @cached_property
    def arc_head(self) -> np.ndarray:
        return np.repeat(np.arange(self.n), np.diff(self.in_ptr))

    @cached_property
    def _arcs(self) -> tuple[np.ndarray, np.ndarray]:
        # The in-arcs of a copy leaving s at time t come from the copies landing at s that are
        # turned around by t: a prefix of the arrivals at s sorted by ready time.
        counts = np.zeros(self.n, dtype=np.int64)
        blocks, offsets, size = [], np.zeros(len(self.S), dtype=np.int64), 0
        for s in range(len(self.S)):
            arrivals = np.flatnonzero(self.dest == s)
            arrivals = arrivals[np.argsort(self.ready[arrivals], kind="stable")]
            departures = np.flatnonzero(self.origin == s)
            counts[departures] = np.searchsorted(
                self.ready[arrivals], self.dep[departures], side="right"
            )
            blocks.append(arrivals)
            offsets[s] = size
            size += len(arrivals)
        in_ptr = np.concatenate([[0], np.cumsum(counts)])
        within = np.arange(in_ptr[-1]) - np.repeat(in_ptr[:-1], counts)
        station = np.repeat(self.origin, counts)
        flat = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int64)
        return in_ptr, flat[offsets[station] + within]

    @cached_property
    def out_arcs(self) -> list[np.ndarray]:
        """Successor copies of each copy (for enumeration and checks on small networks)."""
        order = np.argsort(self.in_tail, kind="stable")
        heads, tails = self.arc_head[order], self.in_tail[order]
        cuts = np.searchsorted(tails, np.arange(self.n + 1))
        return [heads[cuts[v] : cuts[v + 1]] for v in range(self.n)]

    def arcs(self) -> Iterator[tuple[int, int]]:
        """All copy-to-copy arcs as (tail, head)."""
        return zip(self.in_tail.tolist(), self.arc_head.tolist(), strict=True)

    def label(self, v: int) -> str:
        return f"{self.F[self.flight_of[v]].id}+{self.delay[v]}"

    def route(self, k: str, copies: Sequence[int], costs: Costs) -> Route:
        """The route of aircraft k through `copies` (copy indices in flying order)."""
        ki = self.k_index[k]
        cost = 0.0
        for v in copies:
            f = self.F[self.flight_of[v]]
            cost += costs.c_delay * f.pax * int(self.delay[v]) + costs.c_swap * (f.orig != k)
        end = self.dest[copies[-1]] if len(copies) else self.src_station[ki]
        return Route(
            aircraft=k,
            copies=tuple((self.F[self.flight_of[v]].id, int(self.delay[v])) for v in copies),
            cost=cost,
            end=self.S[end],
        )

    def copies_of(self, route: Route) -> list[int]:
        return [self.copy_index[c] for c in route.copies]

    def is_path(self, k: str, copies: Sequence[int]) -> bool:
        """Is `copies` a path from src_k in G_k?"""
        ki = self.k_index[k]
        if not len(copies):
            return True
        if not self.start_ok[copies[0], ki]:
            return False
        return all(
            self.dest[u] == self.origin[v] and self.ready[u] <= self.dep[v]
            for u, v in zip(copies[:-1], copies[1:], strict=True)
        )

    def enumerate_routes(self, k: str, costs: Costs) -> list[Route]:
        """Every route of aircraft k: brute force, for tests and figures on small networks."""
        ki = self.k_index[k]
        out = self.out_arcs
        found = [self.route(k, [], costs)]
        stack = [[int(v)] for v in np.flatnonzero(self.start_ok[:, ki])[::-1]]
        while stack:
            path = stack.pop()
            found.append(self.route(k, path, costs))
            stack.extend(path + [int(w)] for w in out[path[-1]][::-1])
        return found

    def repeated_flights(self) -> list[tuple[str, int]]:
        """Copies (f, d) from which another copy of f is reachable (formulation §2.3).

        Must be empty. One backward pass with bitsets over each station's departures computes it.
        """
        # TODO: maintenance and crew extensions add resources, and with them elementary
        # shortest-path pricing (labels that carry the visited flights); that replaces this cap.
        per_station: list[list[int]] = [[] for _ in self.S]
        for v in range(self.n):  # copies are numbered by departure time already
            per_station[self.origin[v]].append(v)
        position = np.zeros(self.n, dtype=np.int64)
        dep_times: list[list[int]] = []
        for vs in per_station:
            position[vs] = np.arange(len(vs))
            dep_times.append([int(self.dep[v]) for v in vs])
        suffix = [[0] * (len(vs) + 1) for vs in per_station]
        bad = []
        for v in range(self.n - 1, -1, -1):
            s = self.dest[v]
            reach = suffix[s][bisect_left(dep_times[s], int(self.ready[v]))]
            bit = 1 << int(self.flight_of[v])
            if reach & bit:
                bad.append((self.F[self.flight_of[v]].id, int(self.delay[v])))
            o, j = self.origin[v], position[v]
            suffix[o][j] = bit | reach | suffix[o][j + 1]
        return bad

    def assert_elementary(self) -> None:
        bad = self.repeated_flights()
        if bad:
            raise AssertionError(f"a path can fly the same flight twice, from copies {bad[:5]}")
