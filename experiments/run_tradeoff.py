"""Solution quality against solve time: time limit, delay grid and dual smoothing.

python experiments/run_tradeoff.py [--solver gurobi] [--part anytime dmax step smoothing]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recovery.checker import check_plan  # noqa: E402
from recovery.colgen import (  # noqa: E402
    CGSettings,
    build_master,
    column_generation,
    network_pricer,
    price_and_branch,
)
from recovery.generator import _parse_seeds, generate_instance  # noqa: E402
from recovery.instance import Costs, load_config  # noqa: E402
from recovery.network import Network  # noqa: E402

RESULTS = ROOT / "results"


def solve(inst, cfg, costs, solver, time_limit=None) -> dict[str, Any]:
    sol = price_and_branch(
        inst,
        costs,
        backend=solver,
        settings=CGSettings.from_config(cfg),
        mip_time_limit=cfg["mip"]["time_limit"],
        mip_gap=cfg["mip"]["gap"],
        time_limit=time_limit,
        tol=cfg["solver"]["tol"],
        threads=cfg["solver"]["threads"],
    )
    return {
        "instance": inst.name,
        "z_pb": round(sol.z_pb, 6),
        "lb": round(sol.lb, 6),
        "gap": round(sol.gap, 8),
        "cg_stop": sol.cg.stop,
        "cg_iterations": len(sol.cg.log),
        "total_s": round(sol.seconds, 4),
        "copies": sol.network.n,
        "plan_ok": int(check_plan(inst, sol.plan, costs).ok),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def instances(cfg, scale: str, seeds: str):
    for seed in _parse_seeds(seeds):
        for kind in ("aog", "closure"):
            yield generate_instance(scale, seed, kind, cfg=cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", choices=["gurobi", "highs"], help="default: config.yaml")
    ap.add_argument("--part", nargs="+", default=["anytime", "dmax", "step", "smoothing"])
    ap.add_argument("--config", type=Path)
    opts = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = load_config(opts.config)
    costs = Costs.from_config(cfg)
    solver = opts.solver or cfg["solver"]["backend"]
    tr = cfg["tradeoff"]
    RESULTS.mkdir(exist_ok=True)

    if "anytime" in opts.part:
        rows = []
        for inst in instances(cfg, tr["anytime_scale"], tr["anytime_seeds"]):
            ref = solve(inst, cfg, costs, solver)  # no limit: converged z_LP and z_PB
            for limit in tr["time_limits"]:
                r = solve(inst, cfg, costs, solver, time_limit=float(limit))
                rows.append(
                    {"time_limit": limit, **r, "z_lp": ref["lb"], "z_pb_unlimited": ref["z_pb"]}
                )
                print(
                    f"anytime {inst.name:26s} T={limit:6.2f}s z_PB {r['z_pb']:12.1f} "
                    f"LB {r['lb']:12.1f} ({r['cg_stop']}) z_LP {ref['lb']:12.1f}",
                    flush=True,
                )
            write_csv(RESULTS / "tradeoff_time.csv", rows)

    if "smoothing" in opts.part:
        rows = []
        for inst in instances(cfg, tr["smoothing_scale"], tr["smoothing_seeds"]):
            net = Network(inst)
            for alpha in tr["smoothing_values"]:
                settings = CGSettings.from_config(cfg, smoothing=float(alpha))
                master = build_master(
                    net,
                    costs,
                    backend=solver,
                    tol=cfg["solver"]["tol"],
                    threads=cfg["solver"]["threads"],
                )
                cg = column_generation(master, network_pricer(net, costs), settings)
                rows.append(
                    {
                        "instance": inst.name,
                        "smoothing": alpha,
                        "cg_iterations": len(cg.log),
                        "cg_stop": cg.stop,
                        "cg_s": round(cg.seconds, 4),
                        "z_lp": round(cg.lb, 6),
                        "columns_generated": cg.columns_generated,
                    }
                )
                print(
                    f"smooth {inst.name:26s} alpha={alpha:.1f} it {len(cg.log):5d} "
                    f"{cg.seconds:7.2f}s z_LP {cg.lb:12.1f} ({cg.stop})",
                    flush=True,
                )
            write_csv(RESULTS / "tradeoff_smoothing.csv", rows)

    for part, key, values in (
        ("dmax", "d_max", tr["d_max_values"]),
        ("step", "delay_step", tr["step_values"]),
    ):
        if part not in opts.part:
            continue
        rows = []
        for inst in instances(cfg, tr["grid_scale"], tr["grid_seeds"]):
            for value in values:
                if key == "d_max":
                    variant = dataclasses.replace(inst, d_max=int(value))
                else:
                    variant = dataclasses.replace(
                        inst, delay_step=int(value), d_max=tr["step_d_max"]
                    )
                r = solve(variant, cfg, costs, solver)
                rows.append(
                    {key: value, "delay_step": variant.delay_step, "d_max_used": variant.d_max, **r}
                )
                print(
                    f"{part:5s} {inst.name:26s} {key}={value:4d} z_PB {r['z_pb']:12.1f} "
                    f"gap {100 * r['gap']:6.2f}% {r['total_s']:7.2f}s copies {r['copies']}",
                    flush=True,
                )
            write_csv(RESULTS / f"tradeoff_{part}.csv", rows)


if __name__ == "__main__":
    main()
