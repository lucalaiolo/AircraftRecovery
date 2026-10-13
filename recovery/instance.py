"""Instance data, JSON io and preprocessing (formulation §1 and §2).

Times are integer minutes from 00:00 of the recovery day; 00:30 the next day is 1470.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def hhmm(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Flight:
    id: str
    origin: str
    dest: str
    dep: int  # scheduled off-block t_f
    blk: int  # block time beta_f, off-block to on-block
    pax: int
    orig: str  # originally assigned aircraft k0_f

    @property
    def arr(self) -> int:
        """Scheduled on-block time."""
        return self.dep + self.blk


@dataclass(frozen=True)
class Aircraft:
    id: str
    start_station: str


@dataclass(frozen=True)
class AOG:
    """Aircraft unavailable from t1 to t2 at its current station. Recovery starts at t1."""

    aircraft: str
    t1: int
    t2: int
    kind: ClassVar[str] = "aog"

    @property
    def t0(self) -> int:
        return self.t1


@dataclass(frozen=True)
class Closure:
    """No departures or arrivals at `station` during [ta, tb). Recovery starts at ta."""

    station: str
    ta: int
    tb: int
    kind: ClassVar[str] = "closure"

    @property
    def t0(self) -> int:
        return self.ta


@dataclass(frozen=True)
class NoDisruption:
    """Recovery at an explicit t0 with nothing broken: the planned-schedule check and the toy."""

    t0: int
    kind: ClassVar[str] = "none"


Disruption = AOG | Closure | NoDisruption
_DISRUPTION_KINDS: dict[str, type] = {c.kind: c for c in (AOG, Closure, NoDisruption)}


@dataclass(frozen=True)
class Costs:
    c_delay: float
    c_cancel_fixed: float
    c_cancel_pax: float
    c_swap: float
    c_pos: float

    def __post_init__(self) -> None:
        # c_cancel_fixed > 0 gives C_f > 0 for every flight, which formulation §1 requires.
        if min(self.c_delay, self.c_cancel_pax, self.c_swap) < 0:
            raise ValueError("c_delay, c_cancel_pax and c_swap must be nonnegative")
        if self.c_cancel_fixed <= 0 or self.c_pos <= 0:
            raise ValueError("c_cancel_fixed and c_pos must be positive")

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> Costs:
        cfg = load_config() if cfg is None else cfg
        return cls(**{k: float(v) for k, v in cfg["costs"].items()})

    def delay(self, f: Flight, d: int) -> float:
        return self.c_delay * f.pax * d

    def cancel(self, f: Flight) -> float:
        """C_f."""
        return self.c_cancel_fixed + self.c_cancel_pax * f.pax


def elementarity_cap(flights: Iterable[Flight], TAT: int) -> float:
    """D-bar of formulation §2.3: min of blk_f + blk_g over pairs with dest_f = origin_g, + 2 TAT.

    Per station: the shortest flight in plus the shortest flight out; inf if there is no pair.
    """
    min_in: dict[str, int] = {}
    min_out: dict[str, int] = {}
    for f in flights:
        min_in[f.dest] = min(min_in.get(f.dest, f.blk), f.blk)
        min_out[f.origin] = min(min_out.get(f.origin, f.blk), f.blk)
    pair = min((min_in[s] + min_out[s] for s in min_in.keys() & min_out.keys()), default=math.inf)
    return pair + 2 * TAT


@dataclass(frozen=True)
class Instance:
    """A planned day plus one disruption. `flights` is the whole day, frozen flights included."""

    name: str
    stations: tuple[str, ...]
    aircraft: tuple[Aircraft, ...]
    flights: tuple[Flight, ...]
    TAT: int
    b: dict[str, int]  # end-of-day requirement b_s; stations left out count 0
    delay_step: int
    d_max: int
    disruption: Disruption
    seed: int | None = None
    hubs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("stations", "aircraft", "flights", "hubs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        problems = self.problems()
        if problems:
            raise ValueError(f"invalid instance {self.name!r}:\n  " + "\n  ".join(problems))
        object.__setattr__(self, "b", {s: int(self.b.get(s, 0)) for s in self.stations})

    @property
    def K(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.aircraft)

    @property
    def S(self) -> tuple[str, ...]:
        return self.stations

    @property
    def D(self) -> tuple[int, ...]:
        return tuple(range(0, self.d_max + 1, self.delay_step))

    @property
    def t0(self) -> int:
        return self.disruption.t0

    @cached_property
    def flight(self) -> dict[str, Flight]:
        return {f.id: f for f in self.flights}

    def problems(self) -> list[str]:
        """Everything that makes the instance unusable; empty if it is valid."""
        out: list[str] = []
        S = set(self.stations)
        K = {a.id for a in self.aircraft}
        if len(S) != len(self.stations):
            out.append("duplicate station")
        if len(K) != len(self.aircraft):
            out.append("duplicate aircraft id")
        if len({f.id for f in self.flights}) != len(self.flights):
            out.append("duplicate flight id")
        out += [f"hub {h} is not a station" for h in self.hubs if h not in S]
        for a in self.aircraft:
            if a.start_station not in S:
                out.append(f"aircraft {a.id} starts at unknown station {a.start_station}")
        for f in self.flights:
            if f.origin not in S or f.dest not in S:
                out.append(f"flight {f.id} uses an unknown station")
            if f.origin == f.dest:
                out.append(f"flight {f.id} has origin = destination")
            if f.blk <= 0 or f.pax < 0:
                out.append(f"flight {f.id} needs blk > 0 and pax >= 0")
            if f.orig not in K:
                out.append(f"flight {f.id} is assigned to unknown aircraft {f.orig}")
        if self.TAT <= 0:
            out.append("TAT must be positive")
        out += [f"b has unknown station {s}" for s in self.b if s not in S]
        if any(v < 0 for v in self.b.values()) or sum(self.b.values()) != len(K):
            out.append(f"b_s must be nonnegative and sum to |K| = {len(K)}")
        if self.delay_step <= 0 or self.d_max < 0 or self.d_max % self.delay_step:
            out.append("d_max must be a nonnegative multiple of a positive delay_step")
        out += _disruption_problems(self, S, K)
        if not out:
            cap = elementarity_cap((f for f in self.flights if f.dep >= self.t0), self.TAT)
            if self.d_max >= cap:
                out.append(f"d_max = {self.d_max} is not below the elementarity cap {cap}")
        return out


def _disruption_problems(inst: Instance, S: set[str], K: set[str]) -> list[str]:
    dis = inst.disruption
    if isinstance(dis, AOG):
        if dis.aircraft not in K:
            return [f"AOG aircraft {dis.aircraft} is unknown"]
        if dis.t2 <= dis.t1:
            return ["AOG needs t1 < t2"]
        airborne = [f.id for f in inst.flights if f.orig == dis.aircraft and f.dep < dis.t1 < f.arr]
        if airborne:
            return [f"AOG aircraft {dis.aircraft} is in the air at t1 (flight {airborne[0]})"]
    elif isinstance(dis, Closure):
        if dis.station not in S:
            return [f"closed station {dis.station} is unknown"]
        if dis.tb <= dis.ta:
            return ["closure needs ta < tb"]
        # Flights already in the air at ta must not land inside the window: no diversions.
        landing = [
            f.id
            for f in inst.flights
            if f.dep < dis.ta and f.dest == dis.station and dis.ta <= f.arr < dis.tb
        ]
        if landing:
            return [f"flight {landing[0]} is in the air at ta and lands inside the closure"]
    elif not isinstance(dis, NoDisruption):
        return [f"unknown disruption {dis!r}"]
    return []


def describe_disruption(dis: Disruption) -> str:
    if isinstance(dis, AOG):
        return f"AOG {dis.aircraft} {hhmm(dis.t1)}-{hhmm(dis.t2)}"
    if isinstance(dis, Closure):
        return f"closure {dis.station} [{hhmm(dis.ta)}, {hhmm(dis.tb)})"
    return f"no disruption, t0 = {hhmm(dis.t0)}"


def planned_rotations(inst: Instance) -> dict[str, list[Flight]]:
    """Each aircraft's planned flights for the whole day, in departure order."""
    rot: dict[str, list[Flight]] = {k: [] for k in inst.K}
    for f in sorted(inst.flights, key=lambda f: (f.dep, f.id)):
        rot[f.orig].append(f)
    return rot


@dataclass(frozen=True)
class Preprocessed:
    """Formulation §2: the flights to re-plan and the ready state (s_k, rho_k) of each aircraft."""

    t0: int
    F: tuple[str, ...]  # dep >= t0, sorted by (dep, id)
    frozen: tuple[str, ...]  # dep < t0: operated on schedule, fixed data
    ready_station: dict[str, str]  # s_k
    ready_time: dict[str, int]  # rho_k


def preprocess(inst: Instance) -> Preprocessed:
    t0 = inst.t0
    by_dep = sorted(inst.flights, key=lambda f: (f.dep, f.id))
    last_frozen: dict[str, Flight] = {}
    for f in by_dep:
        if f.dep < t0:
            last_frozen[f.orig] = f  # by_dep order leaves the latest one
    station: dict[str, str] = {}
    ready: dict[str, int] = {}
    for a in inst.aircraft:
        f = last_frozen.get(a.id)
        if f is None:
            station[a.id], ready[a.id] = a.start_station, t0
        else:
            station[a.id], ready[a.id] = f.dest, max(t0, f.arr + inst.TAT)
    if isinstance(inst.disruption, AOG):
        k = inst.disruption.aircraft
        ready[k] = max(ready[k], inst.disruption.t2)
    return Preprocessed(
        t0=t0,
        F=tuple(f.id for f in by_dep if f.dep >= t0),
        frozen=tuple(f.id for f in by_dep if f.dep < t0),
        ready_station=station,
        ready_time=ready,
    )


@dataclass
class RecoveryPlan:
    """One route per aircraft as (flight, delay) pairs in flying order, plus cancellations.

    Aircraft missing from `routes` stay on the ground; the checker recomputes `objective`.
    """

    routes: dict[str, list[tuple[str, int]]]
    cancelled: list[str]
    objective: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    return {
        "name": inst.name,
        "seed": inst.seed,
        "stations": list(inst.stations),
        "hubs": list(inst.hubs),
        "TAT": inst.TAT,
        "delay_step": inst.delay_step,
        "d_max": inst.d_max,
        "disruption": {"kind": inst.disruption.kind, **asdict(inst.disruption)},
        "b": {s: v for s, v in inst.b.items() if v},
        "aircraft": [asdict(a) for a in inst.aircraft],
        "flights": [asdict(f) for f in inst.flights],
    }


def instance_from_dict(d: dict[str, Any]) -> Instance:
    dis = dict(d["disruption"])
    return Instance(
        name=d["name"],
        seed=d.get("seed"),
        stations=tuple(d["stations"]),
        hubs=tuple(d.get("hubs", ())),
        TAT=int(d["TAT"]),
        delay_step=int(d["delay_step"]),
        d_max=int(d["d_max"]),
        disruption=_DISRUPTION_KINDS[dis.pop("kind")](**dis),
        b={s: int(v) for s, v in d["b"].items()},
        aircraft=tuple(Aircraft(**a) for a in d["aircraft"]),
        flights=tuple(Flight(**f) for f in d["flights"]),
    )


def save_instance(inst: Instance, path: str | Path) -> None:
    Path(path).write_text(_dumps(instance_to_dict(inst)))


def load_instance(path: str | Path) -> Instance:
    return instance_from_dict(json.loads(Path(path).read_text()))


def plan_to_dict(plan: RecoveryPlan) -> dict[str, Any]:
    return {
        "objective": plan.objective,
        "meta": plan.meta,
        "cancelled": sorted(plan.cancelled),
        "routes": {k: [[f, d] for f, d in r] for k, r in plan.routes.items()},
    }


def plan_from_dict(d: dict[str, Any]) -> RecoveryPlan:
    return RecoveryPlan(
        routes={k: [(f, int(dl)) for f, dl in r] for k, r in d["routes"].items()},
        cancelled=list(d["cancelled"]),
        objective=d.get("objective"),
        meta=dict(d.get("meta", {})),
    )


def save_plan(plan: RecoveryPlan, path: str | Path) -> None:
    Path(path).write_text(_dumps(plan_to_dict(plan)))


def load_plan(path: str | Path) -> RecoveryPlan:
    return plan_from_dict(json.loads(Path(path).read_text()))


def _dumps(d: dict[str, Any]) -> str:
    """JSON with one flight, aircraft or route per line, so files diff and read well."""
    lines = []
    for key, value in d.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            body = ",\n".join(f"    {json.dumps(x)}" for x in value)
            lines.append(f"  {json.dumps(key)}: [\n{body}\n  ]")
        elif isinstance(value, dict) and value and all(isinstance(v, list) for v in value.values()):
            body = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in value.items())
            lines.append(f"  {json.dumps(key)}: {{\n{body}\n  }}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"
