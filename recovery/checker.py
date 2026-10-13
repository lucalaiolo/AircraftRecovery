"""Solver-independent feasibility and cost check of a recovery plan.

The rules, the ready state included, are re-derived here rather than imported from the model code.

    python -m recovery.checker data/small_aog_seed000.json results/plans/small_aog_seed000.pb.json
    python -m recovery.checker --all results/plans
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recovery.instance import (
    AOG,
    Closure,
    Costs,
    Flight,
    Instance,
    RecoveryPlan,
    hhmm,
    load_config,
    load_instance,
    load_plan,
)


@dataclass(frozen=True)
class CostBreakdown:
    delay: float
    swap: float
    cancel: float
    positioning: float

    @property
    def total(self) -> float:
        return self.delay + self.swap + self.cancel + self.positioning


@dataclass(frozen=True)
class CheckResult:
    violations: tuple[str, ...]
    cost: CostBreakdown
    end_station: dict[str, str]  # where each aircraft ends the day
    shortfall: dict[str, int]  # aircraft missing from b_s, per station

    @property
    def ok(self) -> bool:
        return not self.violations


def ready_state(inst: Instance) -> tuple[dict[str, str], dict[str, int]]:
    """Where each aircraft is at t0 and when it can next take off."""
    t0 = inst.t0
    station: dict[str, str] = {}
    ready: dict[str, int] = {}
    for a in inst.aircraft:
        flown = [f for f in inst.flights if f.orig == a.id and f.dep < t0]
        if flown:
            last = max(flown, key=lambda f: f.dep)
            station[a.id] = last.dest
            ready[a.id] = max(t0, last.dep + last.blk + inst.TAT)
        else:
            station[a.id] = a.start_station
            ready[a.id] = t0
    if isinstance(inst.disruption, AOG):
        k = inst.disruption.aircraft
        ready[k] = max(ready[k], inst.disruption.t2)
    return station, ready


def check_plan(
    inst: Instance,
    plan: RecoveryPlan,
    costs: Costs,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-6,
) -> CheckResult:
    v: list[str] = []
    t0, by_id = inst.t0, inst.flight
    station, ready = ready_state(inst)

    v += [f"unknown aircraft {k}" for k in plan.routes if k not in station]
    flown = Counter(fid for r in plan.routes.values() for fid, _ in r)
    cancelled = Counter(plan.cancelled)
    for fid in sorted(flown.keys() | cancelled.keys()):
        if fid not in by_id:
            v.append(f"unknown flight {fid}")
        elif by_id[fid].dep < t0:
            v.append(f"{fid} departed before t0 = {hhmm(t0)}: it is frozen, not a decision")

    # Each flight still to operate is flown exactly once or cancelled.
    for f in inst.flights:
        if f.dep < t0:
            continue
        n_fly, n_cancel = flown[f.id], cancelled[f.id]
        if n_fly + n_cancel == 0:
            v.append(f"{f.id} is neither flown nor cancelled")
        elif n_fly > 1:
            v.append(f"{f.id} is flown {n_fly} times")
        elif n_fly and n_cancel:
            v.append(f"{f.id} is both flown and cancelled")
        elif n_cancel > 1:
            v.append(f"{f.id} is cancelled {n_cancel} times")

    grid = set(inst.D)
    end: dict[str, str] = {}
    legs: list[tuple[str, Flight, int]] = []  # (aircraft, flight, delay), known flights only
    for k in inst.K:
        prev: tuple[Flight, int] | None = None
        for fid, d in plan.routes.get(k, []):
            if fid not in by_id:
                continue
            f = by_id[fid]
            legs.append((k, f, d))
            if d not in grid:
                v.append(
                    f"{k}: {fid} delay {d} is not on the grid "
                    f"{{0, {inst.delay_step}, ..., {inst.d_max}}}"
                )
            off = f.dep + d
            if prev is None:
                if f.origin != station[k]:
                    v.append(
                        f"{k}: first flight {fid} leaves {f.origin}, but {k} is at {station[k]}"
                    )
                if off < ready[k]:
                    v.append(
                        f"{k}: {fid} leaves at {hhmm(off)}, before {k} is ready at {hhmm(ready[k])}"
                    )
            else:
                g, e = prev
                if g.dest != f.origin:
                    v.append(f"{k}: {g.id} lands at {g.dest} but {fid} leaves {f.origin}")
                turned = g.dep + e + g.blk + inst.TAT
                if off < turned:
                    v.append(
                        f"{k}: {fid} leaves at {hhmm(off)}, before {g.id}+{e} is turned around "
                        f"at {hhmm(turned)}"
                    )
            prev = (f, d)
        end[k] = prev[0].dest if prev else station[k]

    dis = inst.disruption
    if isinstance(dis, Closure):
        s, ta, tb = dis.station, dis.ta, dis.tb
        for k, f, d in legs:
            off, on = f.dep + d, f.dep + d + f.blk
            if f.origin == s and ta <= off < tb:
                v.append(f"{k}: {f.id}+{d} leaves closed {s} at {hhmm(off)}")
            if f.dest == s and ta <= on < tb:
                v.append(f"{k}: {f.id}+{d} lands at closed {s} at {hhmm(on)}")

    at_end = Counter(end.values())
    shortfall = {s: max(0, inst.b[s] - at_end[s]) for s in inst.S}
    cost = CostBreakdown(
        delay=sum(costs.delay(f, d) for _, f, d in legs),
        swap=costs.c_swap * sum(f.orig != k for k, f, _ in legs),
        cancel=sum(
            costs.cancel(by_id[fid])
            for fid in plan.cancelled
            if fid in by_id and by_id[fid].dep >= t0
        ),
        positioning=costs.c_pos * sum(shortfall.values()),
    )
    if plan.objective is not None and not math.isclose(
        plan.objective, cost.total, rel_tol=rel_tol, abs_tol=abs_tol
    ):
        v.append(
            f"reported objective {plan.objective!r} differs from recomputed cost {cost.total!r}"
        )
    return CheckResult(tuple(v), cost, end, shortfall)


def plan_context(
    inst: Instance, plan: RecoveryPlan, cfg: dict[str, Any] | None = None
) -> tuple[Instance, Costs]:
    """The instance and costs a saved plan was solved with (its meta can override the grid)."""
    grid = {k: plan.meta[k] for k in ("delay_step", "d_max") if k in plan.meta}
    if grid:
        inst = dataclasses.replace(inst, **grid)
    costs = Costs(**plan.meta["costs"]) if "costs" in plan.meta else Costs.from_config(cfg)
    return inst, costs


def check_saved_plan(
    instance_path: str | Path, plan_path: str | Path, cfg: dict[str, Any] | None = None
) -> CheckResult:
    plan = load_plan(plan_path)
    inst, costs = plan_context(load_instance(instance_path), plan, cfg)
    return check_plan(inst, plan, costs)


def check_all(plans: Path, data: Path, cfg: dict[str, Any] | None = None) -> int:
    """Check every plan JSON under `plans` against data/<meta.instance>.json; returns failures."""
    failures = 0
    files = sorted(plans.rglob("*.json"))
    for path in files:
        plan = load_plan(path)
        res = check_saved_plan(data / f"{plan.meta['instance']}.json", path, cfg)
        if not res.ok:
            failures += 1
            print(f"FAILED {path}: " + "; ".join(res.violations[:3]))
    print(f"{len(files) - failures} of {len(files)} plans pass the checker")
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check a recovery plan against its instance.")
    ap.add_argument("instance", type=Path, nargs="?")
    ap.add_argument("plan", type=Path, nargs="?")
    ap.add_argument("--all", type=Path, metavar="DIR", help="check every plan under DIR")
    ap.add_argument("--data", type=Path, default=Path("data"), help="instances for --all")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.all is not None:
        sys.exit(1 if check_all(args.all, args.data, load_config(args.config)) else 0)
    if args.instance is None or args.plan is None:
        ap.error("give an instance and a plan, or --all DIR")
    res = check_saved_plan(args.instance, args.plan, load_config(args.config))
    c = res.cost
    print(
        f"cost {c.total:.2f} = delay {c.delay:.2f} + swap {c.swap:.2f} + cancel {c.cancel:.2f} "
        f"+ positioning {c.positioning:.2f}"
    )
    if res.ok:
        print("OK: plan is feasible and its cost recomputes")
        return
    print(f"{len(res.violations)} violation(s):")
    for line in res.violations:
        print(f"  - {line}")
    sys.exit(1)


if __name__ == "__main__":
    main()
