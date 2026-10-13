"""Synthetic hub-and-spoke schedules and disruptions.

Planned rotations are built first and flights derived from them, so the planned day is feasible.
The AOG and the closure drawn for the same (scale, seed) disrupt the same day.

    python -m recovery.generator --scale tiny small --seeds 0-9
"""

from __future__ import annotations

import argparse
import logging
import math
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from recovery.instance import (
    AOG,
    REPO_ROOT,
    Aircraft,
    Closure,
    Disruption,
    Flight,
    Instance,
    NoDisruption,
    describe_disruption,
    elementarity_cap,
    load_config,
    save_instance,
)

logger = logging.getLogger(__name__)

DISRUPTION_KINDS = ("aog", "closure", "none")


@dataclass(frozen=True)
class ScaleSpec:
    aircraft: int
    hubs: int
    spokes: int
    flights: int  # target; never exceeded


@dataclass(frozen=True)
class AOGParams:
    t1_min: int
    t1_max: int
    duration_min: int
    duration_max: int


@dataclass(frozen=True)
class ClosureParams:
    ta_min: int
    ta_max: int
    duration_min: int
    duration_max: int
    hub_prob: float
    min_affected: int


@dataclass(frozen=True)
class GeneratorParams:
    TAT: int
    block_min: int
    block_max: int
    first_dep_min: int
    first_dep_max: int
    last_arr_max: int
    slack_max: int
    spoke_start_share: float
    hub_hub_prob: float
    second_hub_prob: float
    pax_min: int
    pax_max: int
    aog: AOGParams
    closure: ClosureParams

    @classmethod
    def from_config(cls, gen: dict[str, Any]) -> GeneratorParams:
        gen = dict(gen)
        aog, closure = AOGParams(**gen.pop("aog")), ClosureParams(**gen.pop("closure"))
        return cls(aog=aog, closure=closure, **gen)


@dataclass(frozen=True)
class Schedule:
    """A planned day before any disruption."""

    stations: tuple[str, ...]
    hubs: tuple[str, ...]
    aircraft: tuple[Aircraft, ...]
    flights: tuple[Flight, ...]
    b: dict[str, int]
    TAT: int


def _draw(rng: np.random.Generator, lo: int, hi: int, step: int = 5) -> int:
    """A uniform multiple of `step` in [lo, hi]."""
    lo_i, hi_i = -(-lo // step), hi // step
    if lo_i > hi_i:
        raise ValueError(f"no multiple of {step} in [{lo}, {hi}]")
    return int(rng.integers(lo_i, hi_i + 1)) * step


def generate_schedule(spec: ScaleSpec, p: GeneratorParams, rng: np.random.Generator) -> Schedule:
    hubs = [f"H{i + 1}" for i in range(spec.hubs)]
    spokes = [f"S{i + 1:02d}" for i in range(spec.spokes)]
    is_hub = set(hubs)

    # Each spoke hangs off a home hub, and sometimes off a second one. Block times are drawn per
    # station pair and are the same in both directions.
    links: dict[str, list[str]] = {s: [] for s in hubs + spokes}
    blk: dict[tuple[str, str], int] = {}

    def link(a: str, b: str) -> None:
        blk[a, b] = blk[b, a] = _draw(rng, p.block_min, p.block_max)
        links[a].append(b)
        links[b].append(a)

    for i, s in enumerate(spokes):
        home = hubs[i % len(hubs)]
        link(home, s)
        others = [h for h in hubs if h != home]
        if others and rng.random() < p.second_hub_prob:
            link(others[int(rng.integers(len(others)))], s)
    for h, g in pairwise(hubs):
        link(h, g)

    n_spoke_start = round(p.spoke_start_share * spec.aircraft)
    starts = [hubs[i % len(hubs)] for i in range(spec.aircraft - n_spoke_start)]
    starts += [spokes[int(i)] for i in rng.integers(len(spokes), size=n_spoke_start)]
    rng.shuffle(starts)
    base, extra = divmod(spec.flights, spec.aircraft)
    n_legs = [base + (i < extra) for i in range(spec.aircraft)]
    rng.shuffle(n_legs)

    aircraft = tuple(Aircraft(f"A{i + 1:02d}", s) for i, s in enumerate(starts))
    legs: list[tuple[int, str, str, str, int]] = []  # (dep, aircraft, origin, dest, blk)
    ends: dict[str, str] = {}
    for a, n in zip(aircraft, n_legs, strict=True):
        route = [a.start_station]
        for _ in range(n):
            here = route[-1]
            hub_links = [x for x in links[here] if x in is_hub]
            spoke_links = [x for x in links[here] if x not in is_hub]
            if here in is_hub and spoke_links and not (hub_links and rng.random() < p.hub_hub_prob):
                options = spoke_links
            else:
                options = hub_links
            route.append(options[int(rng.integers(len(options)))])
        blocks = [blk[o, d] for o, d in pairwise(route)]
        slacks = [_draw(rng, 0, p.slack_max) for _ in range(max(n - 1, 0))]
        # span[m]: first off-block to last on-block of the first m legs
        span = [0] + [
            sum(blocks[:m]) + (m - 1) * p.TAT + sum(slacks[: m - 1]) for m in range(1, n + 1)
        ]
        while n > 0 and p.first_dep_min + span[n] > p.last_arr_max:
            n -= 1  # drop trailing legs that do not fit in the day
        t = _draw(rng, p.first_dep_min, min(p.first_dep_max, p.last_arr_max - span[n])) if n else 0
        for i in range(n):
            legs.append((t, a.id, route[i], route[i + 1], blocks[i]))
            if i < n - 1:
                t += blocks[i] + p.TAT + slacks[i]
        ends[a.id] = route[n]

    legs.sort()
    flights = tuple(
        Flight(
            id=f"F{i + 1:03d}",
            origin=o,
            dest=d,
            dep=dep,
            blk=b,
            pax=int(rng.integers(p.pax_min, p.pax_max + 1)),
            orig=k,
        )
        for i, (dep, k, o, d, b) in enumerate(legs)
    )
    used = {f.origin for f in flights} | {f.dest for f in flights} | set(starts)
    return Schedule(
        stations=tuple(s for s in hubs + spokes if s in used),
        hubs=tuple(hubs),
        aircraft=aircraft,
        flights=flights,
        b=dict(Counter(ends.values())),
        TAT=p.TAT,
    )


def draw_aog(sched: Schedule, p: AOGParams, rng: np.random.Generator) -> AOG:
    """An aircraft goes AOG at a time t1 when it is on the ground, before one of its departures."""
    duration = _draw(rng, p.duration_min, p.duration_max, 15)
    options: dict[str, list[int]] = {}
    for a in sched.aircraft:
        own = [f for f in sched.flights if f.orig == a.id]
        if len(own) < 2:
            continue
        # Ground intervals: before the first departure, and between landing and next departure.
        ground = [(p.t1_min, own[0].dep)] + [(f.arr, g.dep) for f, g in pairwise(own)]
        times = [
            t
            for lo, hi in ground
            for t in range(-(-max(lo, p.t1_min) // 5) * 5, min(hi, p.t1_max) + 1, 5)
            if sum(f.dep >= t for f in own) >= 2 and any(t <= f.dep < t + duration for f in own)
        ]
        if times:
            options[a.id] = times
    if not options:
        raise RuntimeError("no aircraft can go AOG with these parameters")
    k = sorted(options)[int(rng.integers(len(options)))]
    t1 = options[k][int(rng.integers(len(options[k])))]
    return AOG(aircraft=k, t1=t1, t2=t1 + duration)


def _closure_valid(flights: Sequence[Flight], s: str, ta: int, tb: int, min_affected: int) -> bool:
    affected = 0
    for f in flights:
        if f.dep < ta:
            # Already in the air at ta: it must not land inside the window (no diversions).
            if f.dest == s and ta <= f.arr < tb:
                return False
        elif (f.origin == s and f.dep < tb) or (f.dest == s and ta <= f.arr < tb):
            affected += 1
    return affected >= min_affected


def draw_closure(sched: Schedule, p: ClosureParams, rng: np.random.Generator) -> Closure:
    """A station closes for [ta, tb).

    The window must not catch a flight already in the air and must delete the on-time copy of at
    least `min_affected` flights. Hubs are tried first with probability `hub_prob`.
    """
    duration = _draw(rng, p.duration_min, p.duration_max, 15)
    hubs = [s for s in sched.stations if s in sched.hubs]
    spokes = [s for s in sched.stations if s not in sched.hubs]
    pools = [hubs, spokes] if rng.random() < p.hub_prob else [spokes, hubs]
    for pool in pools:
        candidates = []
        for s in pool:
            touching = [f for f in sched.flights if s in (f.origin, f.dest)]
            for ta in range(-(-p.ta_min // 5) * 5, p.ta_max + 1, 5):
                if _closure_valid(touching, s, ta, ta + duration, p.min_affected):
                    candidates.append((s, ta))
        if candidates:
            s, ta = candidates[int(rng.integers(len(candidates)))]
            return Closure(station=s, ta=ta, tb=ta + duration)
    raise RuntimeError("no valid closure window with these parameters")


def choose_d_max(
    flights: Sequence[Flight], TAT: int, step: int, d_max: int | None = None, ceiling: int = 180
) -> int:
    """D_max for the flights still to operate (formulation §2.3).

    Default: min(ceiling, largest multiple of `step` below the elementarity cap). An explicit
    `d_max` must be a multiple of `step` and below the cap.
    """
    cap = elementarity_cap(flights, TAT)
    if d_max is not None:
        if d_max < 0 or d_max % step:
            raise ValueError(f"D_max = {d_max} is not a nonnegative multiple of the step {step}")
        if d_max >= cap:
            raise ValueError(
                f"D_max = {d_max} is not below the elementarity cap {cap}: a route could fly "
                "the same flight twice (formulation §2.3)"
            )
        return d_max
    below_cap = math.inf if math.isinf(cap) else (math.ceil(cap / step) - 1) * step
    chosen = int(min(ceiling // step * step, below_cap))
    logger.info("D_max = %d (elementarity cap %s, delay step %d)", chosen, cap, step)
    return chosen


def generate_instance(
    scale: str,
    seed: int,
    disruption: str,
    *,
    cfg: dict[str, Any] | None = None,
    delay_step: int | None = None,
    d_max: int | None = None,
) -> Instance:
    """`disruption` is "aog", "closure" or "none" (the planned day, t0 = 00:00)."""
    cfg = load_config() if cfg is None else cfg
    spec = ScaleSpec(**cfg["scales"][scale])
    params = GeneratorParams.from_config(cfg["generator"])
    step = delay_step if delay_step is not None else int(cfg["delays"]["step"])
    if d_max is None and cfg["delays"]["d_max"] is not None:
        d_max = int(cfg["delays"]["d_max"])

    # Children of one SeedSequence: the schedule stream does not depend on the disruption kind.
    sched_ss, aog_ss, closure_ss = np.random.SeedSequence([seed, zlib.crc32(scale.encode())]).spawn(
        3
    )
    sched = generate_schedule(spec, params, np.random.default_rng(sched_ss))
    dis: Disruption
    if disruption == "aog":
        dis = draw_aog(sched, params.aog, np.random.default_rng(aog_ss))
    elif disruption == "closure":
        dis = draw_closure(sched, params.closure, np.random.default_rng(closure_ss))
    elif disruption == "none":
        dis = NoDisruption(t0=0)
    else:
        raise ValueError(f"unknown disruption {disruption!r}, expected one of {DISRUPTION_KINDS}")

    F = [f for f in sched.flights if f.dep >= dis.t0]
    chosen = choose_d_max(F, sched.TAT, step, d_max, int(cfg["delays"]["d_max_default"]))
    return Instance(
        name=f"{scale}_{disruption}_seed{seed:03d}",
        seed=seed,
        stations=sched.stations,
        hubs=sched.hubs,
        aircraft=sched.aircraft,
        flights=sched.flights,
        TAT=sched.TAT,
        b=sched.b,
        delay_step=step,
        d_max=chosen,
        disruption=dis,
    )


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def main(argv: Sequence[str] | None = None) -> None:
    cfg_default = load_config()
    ap = argparse.ArgumentParser(description="Generate synthetic recovery instances as JSON.")
    ap.add_argument("--scale", nargs="+", default=list(cfg_default["scales"]))
    ap.add_argument("--seeds", default="0-9", help="e.g. 3, 0-9 or 0,2,5")
    ap.add_argument("--disruption", nargs="+", default=["aog", "closure"], choices=DISRUPTION_KINDS)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for scale in args.scale:
        for seed in _parse_seeds(args.seeds):
            for kind in args.disruption:
                inst = generate_instance(scale, seed, kind, cfg=cfg)
                save_instance(inst, args.out / f"{inst.name}.json")
                n_frozen = sum(f.dep < inst.t0 for f in inst.flights)
                print(
                    f"{inst.name:28s} |K|={len(inst.K):3d} flights={len(inst.flights):4d} "
                    f"(frozen {n_frozen:3d}) |S|={len(inst.S):3d} D_max={inst.d_max:3d}  "
                    f"{describe_disruption(inst.disruption)}"
                )


if __name__ == "__main__":
    main()
