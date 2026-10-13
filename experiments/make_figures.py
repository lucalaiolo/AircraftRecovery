"""Regenerate every figure in docs/figures/.

python experiments/make_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recovery.colgen import (  # noqa: E402
    CGSettings,
    build_master,
    column_generation,
    network_pricer,
    price_and_branch,
)
from recovery.instance import Costs, load_config, load_instance  # noqa: E402
from recovery.network import Network  # noqa: E402
from recovery.viz import (  # noqa: E402
    plot_anytime,
    plot_connection_network,
    plot_convergence,
    plot_grid_sweep,
    plot_recovery_gantt,
    plot_scale,
    save,
)

FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "results"
TOY = ROOT / "examples" / "toy_network.json"
SMALL = ROOT / "data" / "small_aog_seed000.json"


def toy_networks() -> None:
    inst = load_instance(TOY)
    net = Network(inst)
    save(plot_connection_network(inst, net, "A"), FIGURES / "network_toy.svg")
    route = [("f1", 30), ("f4", 30)]
    save(plot_connection_network(inst, net, "A", route=route), FIGURES / "network_toy_route.svg")


def small_cg():
    """Column generation on one small instance, keeping each iteration's pricing duals."""
    cfg = load_config()
    inst = load_instance(SMALL)
    net = Network(inst)
    costs = Costs.from_config(cfg)
    settings = CGSettings.from_config(cfg, keep_duals=True)
    master = build_master(net, costs, backend=cfg["solver"]["backend"])
    cg = column_generation(master, network_pricer(net, costs), settings)
    return inst, net, costs, cg


def convergence(cg) -> None:
    save(plot_convergence(cg.log), FIGURES / "convergence.svg")


def network_pricing(inst, net, costs, cg) -> None:
    """One aircraft at one iteration: that iteration's duals and the route pricing returned."""
    first_delays = list(net.D[:3])
    for it in cg.log[len(cg.log) // 4 :]:
        for route in it.added_routes:
            if not 2 <= len(route.copies) <= 3 or any(
                d not in first_delays for _, d in route.copies
            ):
                continue
            path = net.copies_of(route)
            stations = {net.F[net.flight_of[v]].origin for v in path}
            lo, hi = net.dep[path[0]] - 30, net.ready[path[-1]] + 30
            nearby = sorted(
                (
                    f
                    for f in net.F
                    if f.id not in route.flights and f.origin in stations and lo <= f.dep <= hi
                ),
                key=lambda f: (abs(f.dep - net.dep[path[0]]), f.id),
            )
            flights = list(route.flights) + [f.id for f in nearby][: 8 - len(route.copies)]
            fig = plot_connection_network(
                inst,
                net,
                route.aircraft,
                flights=flights,
                delays=first_delays,
                route=list(route.copies),
                duals=it.duals,
                costs=costs,
            )
            fig.axes[0].set_title(
                f"{inst.name}: aircraft {route.aircraft}, pricing weights at iteration "
                f"{it.iteration}, the route pricing returned in green",
                fontsize=9,
                color="#222222",
                loc="left",
            )
            save(fig, FIGURES / "network_pricing.svg")
            return
    raise RuntimeError("no iteration added a short route on the first three delays")


def gantt() -> None:
    cfg = load_config()
    inst = load_instance(SMALL)
    costs = Costs.from_config(cfg)
    sol = price_and_branch(
        inst, costs, backend=cfg["solver"]["backend"], settings=CGSettings.from_config(cfg)
    )
    title = f"{inst.name}: price-and-branch plan, cost {sol.z_pb:,.0f}"
    save(plot_recovery_gantt(inst, sol.plan, title=title), FIGURES / "gantt.svg")


def _rows(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        print(f"skipped: {path} not found (run the experiments first)")
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def results() -> None:
    cfg = load_config()
    if (rows := _rows("scale.csv")) is not None:
        save(plot_scale(rows), FIGURES / "scale_time.svg")
    if (rows := _rows("tradeoff_time.csv")) is not None:
        save(plot_anytime(rows), FIGURES / "anytime.svg")
    if (rows := _rows("tradeoff_dmax.csv")) is not None:
        ref = max(int(r["d_max"]) for r in rows)
        save(plot_grid_sweep(rows, "d_max", "D_max (min)", ref), FIGURES / "grid_dmax.svg")
    if (rows := _rows("tradeoff_step.csv")) is not None:
        ref = int(cfg["delays"]["step"])
        save(
            plot_grid_sweep(rows, "delay_step", "delay step Δ (min)", ref),
            FIGURES / "grid_step.svg",
        )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    toy_networks()
    print("done: network_toy.svg, network_toy_route.svg")
    inst, net, costs, cg = small_cg()
    convergence(cg)
    network_pricing(inst, net, costs, cg)
    print("done: convergence.svg, network_pricing.svg")
    gantt()
    print("done: gantt.svg")
    results()
    print("done: result figures")


if __name__ == "__main__":
    main()
