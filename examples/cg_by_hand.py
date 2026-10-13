"""Column generation by hand: three iterations on an explicit route list.

Aircraft A is AOG and can only stay. Spares P and Q can each fly one of two flights, f1 and f2,
which leave at the same time; cancelling either costs 100. No positioning rows. Pricing enumerates
the route list.

    python examples/cg_by_hand.py [--solver gurobi|highs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.colgen import CGSettings, column_generation  # noqa: E402
from recovery.master import Master  # noqa: E402
from recovery.network import Route  # noqa: E402
from recovery.pricing import Duals, PricedRoute, PricingResult, reduced_cost  # noqa: E402

FLIGHTS = ["f1", "f2"]
AIRCRAFT = ["A", "P", "Q"]
CANCEL = 100.0
ROUTES = [
    Route("A", (), 0.0),
    Route("P", (), 0.0),
    Route("P", (("f1", 0),), 10.0),
    Route("P", (("f2", 0),), 20.0),
    Route("Q", (), 0.0),
    Route("Q", (("f1", 0),), 10.0),
    Route("Q", (("f2", 0),), 50.0),
]


def name(route: Route) -> str:
    return f"{route.aircraft}→{route.flights[0]}" if route.flights else f"{route.aircraft} stays"


def enumerate_pricer(routes: list[Route]):
    """Pricing by enumeration: the reduced cost of every listed route, cheapest first."""

    def pricer(duals: Duals) -> PricingResult:
        by_aircraft: dict[str, list[PricedRoute]] = {}
        for r in routes:
            if r.aircraft in duals.sigma:
                by_aircraft.setdefault(r.aircraft, []).append(
                    PricedRoute(r, reduced_cost(r, duals), ())
                )
        for found in by_aircraft.values():
            found.sort(key=lambda p: p.reduced_cost)
        return PricingResult({k: v[0].reduced_cost for k, v in by_aircraft.items()}, by_aircraft)

    return pricer


def run(backend: str = "gurobi", aircraft: list[str] = AIRCRAFT) -> list[dict]:
    """One row per iteration: RMP cost, prices, sigma and the routes added."""
    routes = [r for r in ROUTES if r.aircraft in aircraft]
    master = Master(FLIGHTS, {f: CANCEL for f in FLIGHTS}, aircraft, {}, 1.0, backend=backend)
    master.add_routes([r for r in routes if not r.flights], protected=True)
    settings = CGSettings(smoothing=0.0, keep_duals=True)
    cg = column_generation(master, enumerate_pricer(routes), settings)
    return [
        {
            "iteration": it.iteration,
            "z_rmp": it.z_rmp,
            "pi": tuple(it.duals.pi[f] for f in FLIGHTS),
            "sigma": tuple(it.duals.sigma[k] for k in aircraft),
            "added": [name(r) for r in it.added_routes],
            "lagrangian": it.lagrangian,
        }
        for it in cg.log
    ]


def table(rows: list[dict]) -> str:
    def num(x: float) -> str:
        return f"{x:g}" if x != 0 else "0"  # no "-0"

    lines = [
        "| Iteration | RMP cost | Prices (π_f1, π_f2) | σ (A, P, Q) | Routes added |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        pi = "(" + ", ".join(num(p) for p in row["pi"]) + ")"
        sigma = "(" + ", ".join(num(s) for s in row["sigma"]) + ")"
        if last:
            pi, sigma = pi + ", one valid set", sigma + ", one valid set"
        added = ", ".join(row["added"]) or f"none: stop, LP optimum {num(row['z_rmp'])}"
        lines.append(f"| {row['iteration']} | {num(row['z_rmp'])} | {pi} | {sigma} | {added} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", default="gurobi", choices=["gurobi", "highs"])
    args = ap.parse_args()
    rows = run(args.solver)
    print(table(rows))
    print(
        "\nThe duals of the last iteration are not unique (the LP is degenerate); the solver "
        "returned one valid set."
    )


if __name__ == "__main__":
    main()
