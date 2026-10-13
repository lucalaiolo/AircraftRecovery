"""Column generation (formulation §8-§9) and price-and-branch (§10)."""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from recovery.instance import Costs, Instance, RecoveryPlan, load_config
from recovery.master import Master, MIPSolution
from recovery.network import Network, Route
from recovery.pricing import Duals, PricingResult, price, reduced_cost

logger = logging.getLogger(__name__)

Pricer = Callable[[Duals], PricingResult]


@dataclass(frozen=True)
class CGSettings:
    eps: float = 1e-6  # add a column when its reduced cost is below -eps
    gap_tol: float = 0.0  # stop once (z_RMP - LB) / z_RMP <= gap_tol; 0 waits for convergence
    max_iter: int = 100_000
    time_limit: float = math.inf  # seconds for the whole loop
    columns_per_aircraft: int = 1
    smoothing: float = 0.0  # weight alpha of the stability center (Wentges); 0 is plain CG
    purge_threshold: float = 1000.0  # reduced cost above which a non-basic column may go
    purge_age: int = 10  # LP solves it must have stayed non-basic first
    pool_cap: int | None = None  # None: the Gurobi license cap on gurobi, no cap on highs
    keep_duals: bool = False  # store each iteration's pricing duals and new routes in the log

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, **overrides: Any) -> CGSettings:
        cfg = load_config() if cfg is None else cfg
        values = {k: v for k, v in cfg["colgen"].items() if v is not None}
        return dataclasses.replace(cls(**values), **overrides)


@dataclass
class Iteration:
    iteration: int
    z_rmp: float
    min_rc: float  # most negative reduced cost, at the duals of the last pricing of the iteration
    exact: bool  # True if that pricing used the RMP duals themselves (not smoothed ones)
    lagrangian: float  # L at the pricing duals; at the RMP duals z_RMP + sum_k min(0, rc_k*)
    lb: float  # best Lagrangian bound so far
    added: int
    duplicates: int
    purged: int
    pool: int  # route columns in the RMP that produced this iteration's duals
    master_s: float
    pricing_s: float
    elapsed: float
    duals: Duals | None = None  # duals of the last pricing, when keep_duals is set
    added_routes: list[Route] = field(default_factory=list)


@dataclass
class CGResult:
    master: Master
    log: list[Iteration]
    stop: str  # converged, zero, gap, time or iterations
    z_rmp: float
    lb: float  # z_LP if converged, else the best Lagrangian bound
    duals: Duals  # duals of the last RMP
    fractional: bool  # the last RMP solution has a route column strictly between 0 and 1
    columns_generated: int
    purged: int
    master_s: float
    pricing_s: float
    seconds: float

    @property
    def converged(self) -> bool:
        return self.stop in ("converged", "zero")


def _mix(center: Duals, current: Duals, alpha: float) -> Duals:
    def blend(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
        return {key: alpha * a[key] + (1 - alpha) * b[key] for key in b}

    return Duals(
        pi=blend(center.pi, current.pi),
        sigma=blend(center.sigma, current.sigma),
        mu=blend(center.mu, current.mu),
    )


def column_generation(
    master: Master,
    pricer: Pricer,
    settings: CGSettings | None = None,
    *,
    on_iteration: Callable[[Iteration], None] | None = None,
) -> CGResult:
    """Column generation (formulation §8), optionally with Wentges dual smoothing.

    With smoothing, pricing runs at alpha * center + (1 - alpha) * RMP duals, and a route enters
    only if its reduced cost at the RMP duals is below -eps. If none does, the iteration prices
    again at the RMP duals, so the stopping test stays exact.
    """
    start = time.perf_counter()
    settings = settings or CGSettings()
    cap = settings.pool_cap
    if cap is None and master.backend == "gurobi":
        cap = master.gurobi_route_cap()
    best_lb = -math.inf
    center: Duals | None = None
    log: list[Iteration] = []
    generated = purged_total = 0
    master_s = pricing_s = 0.0

    def select(priced: PricingResult, rmp: Duals | None) -> tuple[list[Route], int]:
        """Routes with reduced cost below -eps at the RMP duals."""
        new, duplicates = [], 0
        for k in master.aircraft:
            for cand in priced.routes[k][: settings.columns_per_aircraft]:
                rc = cand.reduced_cost if rmp is None else reduced_cost(cand.route, rmp)
                if rc >= -settings.eps:
                    continue
                if cand.route.key in master:
                    # A pool column priced below -eps: solver tolerance, not progress.
                    duplicates += 1
                    logger.warning(
                        "pricing returned a column already in the pool (%s, reduced cost "
                        "%.3g); %s counts as converged this iteration",
                        cand.route.label(),
                        rc,
                        k,
                    )
                    break
                new.append(cand.route)
        return new, duplicates

    for it in range(max(settings.max_iter, 1)):
        lp = master.solve_lp()
        master_s += lp.seconds
        t = time.perf_counter()
        exact = settings.smoothing <= 0 or center is None
        point = lp.duals if exact else _mix(center, lp.duals, settings.smoothing)
        priced = pricer(point)
        if exact:
            # At the RMP duals, sum pi + sum sigma + sum b mu = z_RMP (strong duality).
            lagrangian = lp.objective + sum(min(0.0, rc) for rc in priced.best.values())
        else:
            lagrangian = master.lagrangian_bound(point, priced.best)
        if lagrangian > best_lb:
            best_lb, center = lagrangian, point
        new, duplicates = select(priced, None if exact else lp.duals)
        if not exact and not new:  # mispricing: check the RMP duals themselves
            exact, point = True, lp.duals
            priced = pricer(point)
            lagrangian = lp.objective + sum(min(0.0, rc) for rc in priced.best.values())
            if lagrangian > best_lb:
                best_lb, center = lagrangian, point
            new, duplicates = select(priced, None)
        t_pricing = time.perf_counter() - t
        pricing_s += t_pricing

        elapsed = time.perf_counter() - start
        stop = None
        if lp.objective <= 1e-9:
            stop = "zero"  # costs are nonnegative, so z_LP = 0 as well
        elif not new:
            stop = "converged"
        elif settings.gap_tol > 0 and (lp.objective - best_lb) / lp.objective <= settings.gap_tol:
            stop = "gap"
        elif elapsed >= settings.time_limit:
            stop = "time"
        elif it + 1 >= settings.max_iter:
            stop = "iterations"

        pool = master.n_routes
        purged = []
        if stop is None:
            purged = master.purge(
                threshold=settings.purge_threshold,
                min_age=settings.purge_age,
                room=len(new),
                cap=cap,
            )
            master.add_routes(new)
            generated += len(new)
            purged_total += len(purged)
        entry = Iteration(
            iteration=it,
            z_rmp=lp.objective,
            min_rc=min(priced.best.values()),
            exact=exact,
            lagrangian=lagrangian,
            lb=best_lb,
            added=len(new) if stop is None else 0,
            duplicates=duplicates,
            purged=len(purged),
            pool=pool,
            master_s=lp.seconds,
            pricing_s=t_pricing,
            elapsed=elapsed,
            duals=point if settings.keep_duals else None,
            added_routes=list(new) if settings.keep_duals and stop is None else [],
        )
        log.append(entry)
        logger.info(
            "it %4d  z_RMP %14.2f  min rc %13.2f  LB %14.2f  added %3d  pool %5d  %7.2fs",
            it,
            entry.z_rmp,
            entry.min_rc,
            entry.lb,
            entry.added,
            master.n_routes,
            elapsed,
        )
        if on_iteration is not None:
            on_iteration(entry)
        if stop is not None:
            break

    if stop in ("converged", "zero"):
        lb = 0.0 if stop == "zero" else lp.objective
    else:
        lb = best_lb
    return CGResult(
        master=master,
        log=log,
        stop=stop,
        z_rmp=lp.objective,
        lb=lb,
        duals=lp.duals,
        fractional=master.is_fractional(),
        columns_generated=generated,
        purged=purged_total,
        master_s=master_s,
        pricing_s=pricing_s,
        seconds=time.perf_counter() - start,
    )


def seed_routes(net: Network, costs: Costs) -> list[Route]:
    """Each aircraft's original rotation made feasible (formulation §8, step 0)."""
    routes = []
    for ki, k in enumerate(net.K):
        ready, station = int(net.src_time[ki]), net.S[net.src_station[ki]]
        path: list[int] = []
        for f in (f for f in net.F if f.orig == k):
            if f.origin != station:
                break
            v = next(
                (
                    net.copy_index[f.id, d]
                    for d in net.D
                    if f.dep + d >= ready and (f.id, d) in net.copy_index
                ),
                None,
            )
            if v is None:
                break
            path.append(v)
            ready, station = int(net.ready[v]), f.dest
        if path:
            routes.append(net.route(k, path, costs))
    return routes


def build_master(
    net: Network,
    costs: Costs,
    *,
    backend: str = "gurobi",
    seed: bool = True,
    tol: float = 1e-9,
    threads: int = 1,
) -> Master:
    """The initial RMP: all z_f and y_s, every empty route, and the seeded rotations."""
    master = Master(
        flights=[f.id for f in net.F],
        cancel_cost={f.id: costs.cancel(f) for f in net.F},
        aircraft=net.K,
        b=net.instance.b,
        c_pos=costs.c_pos,
        backend=backend,
        tol=tol,
        threads=threads,
    )
    master.add_routes([net.route(k, [], costs) for k in net.K], protected=True)
    if seed:
        master.add_routes([r for r in seed_routes(net, costs) if r.key not in master])
    return master


def network_pricer(net: Network, costs: Costs, max_routes: int = 1) -> Pricer:
    return lambda duals: price(net, costs, duals, max_routes=max_routes)


@dataclass
class Solution:
    instance: Instance
    network: Network
    cg: CGResult
    mip: MIPSolution
    plan: RecoveryPlan
    lb: float
    z_pb: float
    gap: float
    seconds: float


def relative_gap(upper: float, lower: float) -> float:
    """(upper - lower) / upper; 0 when the plan costs nothing (formulation §10)."""
    return 0.0 if upper <= 0 else max(0.0, (upper - lower) / upper)


def price_and_branch(
    inst: Instance,
    costs: Costs,
    *,
    backend: str = "gurobi",
    settings: CGSettings | None = None,
    mip_time_limit: float | None = None,
    mip_gap: float = 0.0,
    time_limit: float | None = None,
    cg_share: float = 0.8,
    tol: float = 1e-9,
    threads: int = 1,
) -> Solution:
    """Column generation, then the master as a MIP over the generated pool.

    A `time_limit` covers both: column generation may use `cg_share` of it, the MIP gets the rest.
    """
    start = time.perf_counter()
    settings = settings or CGSettings()
    if time_limit is not None:
        settings = dataclasses.replace(settings, time_limit=cg_share * time_limit)
    net = Network(inst)
    master = build_master(net, costs, backend=backend, tol=tol, threads=threads)
    cg = column_generation(
        master, network_pricer(net, costs, settings.columns_per_aircraft), settings
    )
    remaining = mip_time_limit
    if time_limit is not None:
        left = max(time_limit - (time.perf_counter() - start), 0.05 * time_limit)
        remaining = left if remaining is None else min(remaining, left)
    mip = master.solve_mip(time_limit=remaining, mip_gap=mip_gap)
    seconds = time.perf_counter() - start
    plan = RecoveryPlan(
        routes={k: list(r.copies) for k, r in mip.routes.items()},
        cancelled=sorted(mip.cancelled),
        objective=mip.objective,
        meta={
            "instance": inst.name,
            "solver": backend,
            "method": "price-and-branch",
            "delay_step": inst.delay_step,
            "d_max": inst.d_max,
            "costs": dataclasses.asdict(costs),
            "lb": cg.lb,
            "gap": relative_gap(mip.objective, cg.lb),
            "cg_stop": cg.stop,
        },
    )
    return Solution(
        instance=inst,
        network=net,
        cg=cg,
        mip=mip,
        plan=plan,
        lb=cg.lb,
        z_pb=mip.objective,
        gap=relative_gap(mip.objective, cg.lb),
        seconds=seconds,
    )
