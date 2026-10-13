"""Pricing (formulation §7.2): the route of minimum reduced cost of each aircraft.

`price` sweeps the copies in time order with a running minimum per station and never lists the
arcs. `price_on_arcs` runs the recurrence over the explicit arc lists; the tests use it as a check.
Both work on (copies x aircraft) arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovery.instance import Costs
from recovery.network import DEPART, READY, Network, Route

NO_PRED, FROM_SOURCE = -2, -1


@dataclass(frozen=True)
class Duals:
    pi: dict[str, float]  # cover rows, per flight
    sigma: dict[str, float]  # one-route rows, per aircraft
    mu: dict[str, float]  # positioning rows, per station; a station without a row counts 0


@dataclass(frozen=True)
class PricedRoute:
    route: Route
    reduced_cost: float
    copies: tuple[int, ...]  # copy indices in the network


@dataclass(frozen=True)
class PricingResult:
    best: dict[str, float]  # min reduced cost per aircraft, rc_k*
    routes: dict[str, list[PricedRoute]]  # best routes per aircraft, cheapest first


def weights(net: Network, costs: Costs, duals: Duals) -> tuple[np.ndarray, ...]:
    """W (copies x aircraft) node weights, mu per station, sigma per aircraft."""
    pi = np.array([duals.pi[f.id] for f in net.F], dtype=float)
    mu = np.array([duals.mu.get(s, 0.0) for s in net.S], dtype=float)
    sigma = np.array([duals.sigma[k] for k in net.K], dtype=float)
    W = net.copy_costs(costs) - pi[net.flight_of][:, None]
    return W, mu, sigma


def price(net: Network, costs: Costs, duals: Duals, *, max_routes: int = 1) -> PricingResult:
    """Exact pricing of every aircraft by the time sweep."""
    W, mu, sigma = weights(net, costs, duals)
    K = len(net.K)
    L = np.full((net.n, K), np.inf)  # L_k(v): cheapest path weight from src_k to v, v included
    pred = np.full((net.n, K), NO_PRED, dtype=np.int64)
    best = np.full((len(net.S), K), np.inf)  # running minimum of L over copies ready at s
    arg = np.full((len(net.S), K), NO_PRED, dtype=np.int64)
    origin, dest = net.origin, net.dest
    for kind, i in zip(net.event_kind.tolist(), net.event_idx.tolist(), strict=True):
        if kind == DEPART:
            s = origin[i]
            L[i] = W[i] + best[s]
            pred[i] = arg[s]
        elif kind == READY:
            s = dest[i]
            better = L[i] < best[s]
            if better.any():
                best[s] = np.where(better, L[i], best[s])
                arg[s] = np.where(better, i, arg[s])
        else:  # source of aircraft i: L = 0 at s_i from rho_i on; nothing earlier reaches i
            s = net.src_station[i]
            best[s, i] = 0.0
            arg[s, i] = FROM_SOURCE
    total = L - mu[dest][:, None]  # path ending with the arc into snk of the copy's destination
    empty = -mu[net.src_station]
    return _collect(net, costs, total, empty, sigma, pred, max_routes)


def price_on_arcs(
    net: Network, costs: Costs, duals: Duals, *, max_routes: int = 1
) -> PricingResult:
    """Exact pricing by the recurrence of §7.2 over the explicit in-arc lists."""
    W, mu, sigma = weights(net, costs, duals)
    K = len(net.K)
    cols = np.arange(K)
    start = net.start_ok
    L = np.full((net.n, K), np.inf)
    pred = np.full((net.n, K), NO_PRED, dtype=np.int64)
    in_ptr, in_tail = net.in_ptr, net.in_tail
    no_val, no_arg = np.full(K, np.inf), np.full(K, NO_PRED, dtype=np.int64)
    for v in range(net.n):  # copies are numbered in topological order
        a0, a1 = in_ptr[v], in_ptr[v + 1]
        if a1 > a0:
            tails = in_tail[a0:a1]
            vals = L[tails]
            j = vals.argmin(axis=0)
            m, a = vals[j, cols], tails[j]
        else:
            m, a = no_val, no_arg
        from_src = start[v] & (m >= 0.0)  # the start arc contributes 0; ties go to the source
        L[v] = W[v] + np.where(from_src, 0.0, m)
        pred[v] = np.where(from_src, FROM_SOURCE, a)
    total = L - mu[net.dest][:, None]
    empty = -mu[net.src_station]
    return _collect(net, costs, total, empty, sigma, pred, max_routes)


def _collect(
    net: Network,
    costs: Costs,
    total: np.ndarray,
    empty: np.ndarray,
    sigma: np.ndarray,
    pred: np.ndarray,
    max_routes: int,
) -> PricingResult:
    """The cheapest routes of each aircraft: one per end copy, plus the empty route."""
    best: dict[str, float] = {}
    routes: dict[str, list[PricedRoute]] = {}
    for ki, k in enumerate(net.K):
        col = total[:, ki]
        m = min(max_routes, net.n)
        ends = np.argpartition(col, m - 1)[:m] if m else np.zeros(0, dtype=np.int64)
        cand = [(float(col[v]), int(v)) for v in ends if np.isfinite(col[v])]
        if np.isfinite(empty[ki]):
            cand.append((float(empty[ki]), -1))
        cand.sort(key=lambda c: (c[0], c[1] >= 0))  # on ties prefer the empty route
        found = []
        for value, v in cand[:max_routes]:
            path: list[int] = []
            while v >= 0:
                path.append(v)
                v = int(pred[v, ki])
            path.reverse()
            found.append(
                PricedRoute(net.route(k, path, costs), value - float(sigma[ki]), tuple(path))
            )
        routes[k] = found
        best[k] = found[0].reduced_cost if found else np.inf
    return PricingResult(best=best, routes=routes)


def reduced_cost(route: Route, duals: Duals) -> float:
    """c_r - sum_f a_fr pi_f - sigma_k - sum_s e_sr mu_s, recomputed from the column (§7.1)."""
    rc = route.cost - sum(duals.pi[f] for f in route.flights) - duals.sigma[route.aircraft]
    if route.end is not None:
        rc -= duals.mu.get(route.end, 0.0)
    return rc
