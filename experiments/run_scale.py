"""Scale experiment: every scale x seed x disruption type.

Column generation and price-and-branch, the compact MILP where configured, and the heuristic.
Plans are checked and saved under results/plans/; rows go to results/scale.csv.

    python experiments/run_scale.py --solver gurobi
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recovery.checker import check_plan  # noqa: E402
from recovery.colgen import CGSettings, price_and_branch, relative_gap  # noqa: E402
from recovery.compact import compact_size, solve_compact  # noqa: E402
from recovery.generator import _parse_seeds, generate_instance  # noqa: E402
from recovery.heuristic import solve_heuristic  # noqa: E402
from recovery.instance import Costs, load_config, save_instance, save_plan  # noqa: E402
from recovery.network import Network  # noqa: E402
from recovery.solvers import LicenseLimitError  # noqa: E402

RESULTS = ROOT / "results"
PLANS = RESULTS / "plans"
DATA = ROOT / "data"


def run_instance(inst, cfg: dict[str, Any], costs: Costs, solver: str, opts) -> dict[str, Any]:
    net = Network(inst)
    row: dict[str, Any] = {
        "instance": inst.name,
        "scale": inst.name.split("_")[0],
        "disruption": inst.disruption.kind,
        "seed": inst.seed,
        "aircraft": len(inst.K),
        "flights": len(net.F),
        "flights_planned": len(inst.flights),
        "stations": len(inst.S),
        "copies": net.n,
        "d_max": inst.d_max,
    }
    sol = price_and_branch(
        inst,
        costs,
        backend=solver,
        settings=CGSettings.from_config(cfg),
        mip_time_limit=cfg["mip"]["time_limit"],
        mip_gap=cfg["mip"]["gap"],
        tol=cfg["solver"]["tol"],
        threads=cfg["solver"]["threads"],
    )
    cg = sol.cg
    save_plan(sol.plan, PLANS / f"{inst.name}.pb.json")
    row.update(
        cg_iterations=len(cg.log),
        cg_stop=cg.stop,
        columns_generated=cg.columns_generated,
        columns_purged=cg.purged,
        pricing_s=round(cg.pricing_s, 4),
        master_s=round(cg.master_s, 4),
        cg_s=round(cg.seconds, 4),
        lp_fractional=int(cg.fractional),
        lb=round(sol.lb, 6),
        z_pb=round(sol.z_pb, 6),
        gap=round(sol.gap, 8),
        mip_s=round(sol.mip.seconds, 4),
        total_s=round(sol.seconds, 4),
        plan_ok=int(check_plan(inst, sol.plan, costs).ok),
    )

    size = compact_size(net)
    row["compact_vars"] = size.variables
    if opts.no_compact or row["scale"] not in cfg["compact"].get("scales", [row["scale"]]):
        row["compact_status"] = "skipped"
    elif size.variables > cfg["compact"]["max_variables"]:
        row["compact_status"] = "too large"
    else:
        ip = solve_compact(
            net,
            costs,
            backend=cfg["compact"]["backend"],
            time_limit=cfg["compact"]["time_limit"],
            threads=cfg["solver"]["threads"],
        )
        row.update(
            compact_status=ip.status,
            compact_z=round(ip.objective, 6) if ip.plan else "",
            compact_bound=round(ip.bound, 6),
            compact_s=round(ip.build_s + ip.solve_s, 4),
        )
        if ip.plan is not None:
            save_plan(ip.plan, PLANS / f"{inst.name}.compact.json")
            row["compact_ok"] = int(check_plan(inst, ip.plan, costs).ok)

    if not opts.no_heuristic:
        h = solve_heuristic(inst, costs, cfg=cfg)
        save_plan(h.plan, PLANS / f"{inst.name}.heuristic.json")
        row.update(
            heur_cost=round(h.cost, 6),
            heur_greedy=round(h.greedy_cost, 6),
            heur_gap=round(relative_gap(h.cost, sol.lb), 8),
            heur_s=round(h.seconds, 4),
            heur_ok=int(check_plan(inst, h.plan, costs).ok),
        )
    return row


FIELDS = [
    "instance", "scale", "disruption", "seed", "aircraft", "flights", "flights_planned",
    "stations", "copies", "d_max", "cg_iterations", "cg_stop", "columns_generated",
    "columns_purged", "pricing_s", "master_s", "cg_s", "lp_fractional", "lb", "z_pb", "gap",
    "mip_s", "total_s", "plan_ok", "compact_vars", "compact_status", "compact_z",
    "compact_bound", "compact_s", "compact_ok", "heur_cost", "heur_greedy", "heur_gap",
    "heur_s", "heur_ok",
]  # fmt: skip


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["scale"], r["disruption"]), []).append(r)
    order = {"tiny": 0, "small": 1, "medium": 2, "large": 3}
    for (scale, dis), rs in sorted(groups.items(), key=lambda g: (order.get(g[0][0], 9), g[0][1])):

        def mean(key, rs=rs):
            vals = [float(r[key]) for r in rs if r.get(key) not in (None, "")]
            return round(statistics.fmean(vals), 6) if vals else ""

        attempted = [r for r in rs if r["compact_status"] not in ("skipped", "too large")]
        solved = [r for r in attempted if r["compact_status"] == "optimal"]
        out.append(
            {
                "scale": scale,
                "disruption": dis,
                "instances": len(rs),
                "aircraft": mean("aircraft"),
                "flights": mean("flights"),
                "copies": mean("copies"),
                "cg_iterations": mean("cg_iterations"),
                "columns_generated": mean("columns_generated"),
                "pricing_s": mean("pricing_s"),
                "master_s": mean("master_s"),
                "total_s": mean("total_s"),
                "max_total_s": round(max(float(r["total_s"]) for r in rs), 4),
                "gap": mean("gap"),
                "max_gap": round(max(float(r["gap"]) for r in rs), 8),
                "lp_fractional": sum(int(r["lp_fractional"]) for r in rs),
                "gap_positive": sum(float(r["gap"]) > 1e-9 for r in rs),
                "converged": sum(r["cg_stop"] in ("converged", "zero") for r in rs),
                "plans_ok": sum(int(r["plan_ok"]) for r in rs),
                "compact_attempted": len(attempted),
                "compact_solved": len(solved),
                "compact_s": round(statistics.fmean(float(r["compact_s"]) for r in solved), 4)
                if solved
                else "",
                "compact_equals_pb": sum(
                    abs(float(r["compact_z"]) - float(r["z_pb"]))
                    <= 1e-6 * max(1.0, float(r["z_pb"]))
                    for r in solved
                ),
                "heur_gap": mean("heur_gap"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", choices=["gurobi", "highs"], help="default: config.yaml")
    ap.add_argument("--scales", nargs="+", default=["tiny", "small", "medium", "large"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--disruptions", nargs="+", default=["aog", "closure"])
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--no-heuristic", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "scale.csv")
    ap.add_argument(
        "--resume", action="store_true", help="keep the rows already in --out and skip them"
    )
    ap.add_argument("--config", type=Path)
    opts = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = load_config(opts.config)
    costs = Costs.from_config(cfg)
    solver = opts.solver or cfg["solver"]["backend"]
    PLANS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    if opts.resume and opts.out.exists():
        with open(opts.out, newline="") as fh:
            rows = list(csv.DictReader(fh))
    done = {r["instance"] for r in rows}
    for scale in opts.scales:
        for seed in _parse_seeds(opts.seeds):
            for kind in opts.disruptions:
                inst = generate_instance(scale, seed, kind, cfg=cfg)
                save_instance(inst, DATA / f"{inst.name}.json")
                if inst.name in done:
                    continue
                try:
                    row = run_instance(inst, cfg, costs, solver, opts)
                except LicenseLimitError as err:
                    sys.exit(f"error on {inst.name}: {err}")
                rows.append(row)
                write_csv(opts.out, rows, FIELDS)
                print(
                    f"{row['instance']:26s} it {row['cg_iterations']:4d} LB {row['lb']:12.1f} "
                    f"z_PB {row['z_pb']:12.1f} gap {100 * row['gap']:6.2f}% "
                    f"{row['total_s']:7.2f}s | compact {row['compact_status']}"
                    + (f" {row.get('compact_s', 0):.1f}s" if row.get("compact_s") else "")
                    + (f" | heur {100 * row['heur_gap']:.1f}%" if "heur_gap" in row else ""),
                    flush=True,
                )
    summary = summarize(rows)
    write_csv(opts.out.with_name(opts.out.stem + "_summary.csv"), summary, list(summary[0]))
    print(f"wrote {opts.out} and its summary")


if __name__ == "__main__":
    main()
