"""Solve one instance, print the KPIs and save the checked plan.

python experiments/solve_one.py data/small_aog_seed000.json
python experiments/solve_one.py --scale medium --seed 3 --disruption closure --solver highs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recovery.checker import check_plan  # noqa: E402
from recovery.colgen import CGSettings, price_and_branch, relative_gap  # noqa: E402
from recovery.compact import compact_size, solve_compact  # noqa: E402
from recovery.generator import generate_instance  # noqa: E402
from recovery.instance import (  # noqa: E402
    Costs,
    describe_disruption,
    load_config,
    load_instance,
    save_plan,
)
from recovery.network import Network  # noqa: E402
from recovery.solvers import LicenseLimitError  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("instance", nargs="?", type=Path, help="instance JSON (or use --scale/--seed)")
    ap.add_argument("--scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disruption", default="aog", choices=["aog", "closure"])
    ap.add_argument("--solver", choices=["gurobi", "highs"], help="default: config.yaml")
    ap.add_argument("--time-limit", type=float, help="seconds for column generation + MIP")
    ap.add_argument("--compact", action="store_true", help="also solve the compact MILP")
    ap.add_argument("--heuristic", action="store_true", help="also run the greedy + SA baseline")
    ap.add_argument("--out", type=Path, help="plan JSON (default results/plans/<name>.json)")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.instance is not None:
        inst = load_instance(args.instance)
    elif args.scale:
        inst = generate_instance(args.scale, args.seed, args.disruption, cfg=cfg)
    else:
        ap.error("give an instance file or --scale")
    costs = Costs.from_config(cfg)
    solver = args.solver or cfg["solver"]["backend"]
    net = Network(inst)
    print(
        f"{inst.name}: |K|={len(inst.K)} |F|={len(net.F)} (of {len(inst.flights)} planned) "
        f"|S|={len(inst.S)} copies={net.n} D=0..{inst.d_max} step {inst.delay_step}"
    )
    print(f"  {describe_disruption(inst.disruption)}")

    try:
        sol = price_and_branch(
            inst,
            costs,
            backend=solver,
            settings=CGSettings.from_config(cfg),
            mip_time_limit=cfg["mip"]["time_limit"],
            mip_gap=cfg["mip"]["gap"],
            time_limit=args.time_limit,
            tol=cfg["solver"]["tol"],
            threads=cfg["solver"]["threads"],
        )
    except LicenseLimitError as err:
        sys.exit(f"error: {err}")
    cg = sol.cg
    print(
        f"column generation ({solver}): {len(cg.log)} iterations, stop: {cg.stop}, "
        f"{cg.columns_generated} columns generated, {cg.purged} purged"
    )
    print(
        f"  time: pricing {cg.pricing_s:.2f}s, master {cg.master_s:.2f}s, "
        f"total {cg.seconds:.2f}s; LP solution fractional: {cg.fractional}"
    )
    print(
        f"  LB {sol.lb:,.2f}   z_PB {sol.z_pb:,.2f}   gap {100 * sol.gap:.3f}%   "
        f"MIP {sol.mip.status} {sol.mip.seconds:.2f}s   total {sol.seconds:.2f}s"
    )

    res = check_plan(inst, sol.plan, costs)
    c = res.cost
    print(
        f"plan: {sum(len(r) for r in sol.plan.routes.values())} flights flown, "
        f"{len(sol.plan.cancelled)} cancelled; cost = delay {c.delay:,.0f} + swap {c.swap:,.0f}"
        f" + cancel {c.cancel:,.0f} + positioning {c.positioning:,.0f}"
    )
    print(f"checker: {'OK' if res.ok else 'FAILED: ' + '; '.join(res.violations)}")
    out = args.out or ROOT / "results" / "plans" / f"{inst.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_plan(sol.plan, out)
    print(f"plan saved to {out}")

    if args.compact:
        size = compact_size(net)
        limit = cfg["compact"]["max_variables"]
        if size.variables > limit:
            print(f"compact: not attempted, {size.variables:,} variables > {limit:,}")
        else:
            try:
                ip = solve_compact(
                    net,
                    costs,
                    backend=cfg["compact"]["backend"],
                    time_limit=cfg["compact"]["time_limit"],
                )
            except LicenseLimitError as err:
                sys.exit(f"error: {err}")
            ok = check_plan(inst, ip.plan, costs).ok if ip.plan else None
            print(
                f"compact ({cfg['compact']['backend']}, {size.variables:,} variables): "
                f"{ip.status}, z_IP {ip.objective:,.2f}, bound {ip.bound:,.2f}, "
                f"{ip.solve_s:.2f}s, checker {'OK' if ok else ok}"
            )

    if args.heuristic:
        from recovery.heuristic import solve_heuristic

        h = solve_heuristic(inst, costs, cfg=cfg)
        hres = check_plan(inst, h.plan, costs)
        print(
            f"heuristic: cost {h.cost:,.2f} (greedy {h.greedy_cost:,.2f}), "
            f"gap to LB {100 * relative_gap(h.cost, sol.lb):.3f}%, {h.seconds:.2f}s, "
            f"checker {'OK' if hres.ok else 'FAILED'}"
        )


if __name__ == "__main__":
    main()
