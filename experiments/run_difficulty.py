"""Share of instances whose LP optimum is fractional, per generator setting.

If almost every LP optimum is integral, the gap says nothing. This tries tighter schedules, longer
disruptions and cheaper delays on small and medium instances.

    python experiments/run_difficulty.py [--solver highs]
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recovery.colgen import CGSettings, price_and_branch  # noqa: E402
from recovery.generator import _parse_seeds, generate_instance  # noqa: E402
from recovery.instance import Costs, load_config  # noqa: E402

RESULTS = ROOT / "results"
LONG = {
    "generator.aog.duration_min": 240,
    "generator.aog.duration_max": 480,
    "generator.closure.duration_min": 120,
    "generator.closure.duration_max": 240,
}
SETTINGS: dict[str, dict[str, Any]] = {
    "slack 40 (first generator)": {"generator.slack_max": 40},
    "slack 20": {"generator.slack_max": 20},
    "slack 10": {"generator.slack_max": 10},
    "slack 0": {"generator.slack_max": 0},
    "slack 40, long disruptions": {"generator.slack_max": 40, **LONG},
    "slack 40, c_delay 0.5": {"generator.slack_max": 40, "costs.c_delay": 0.5},
    "slack 40, c_delay 0.25": {"generator.slack_max": 40, "costs.c_delay": 0.25},
    "slack 10, long disruptions": {"generator.slack_max": 10, **LONG},
}


def with_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for path, value in overrides.items():
        node = cfg
        *head, last = path.split(".")
        for key in head:
            node = node[key]
        node[last] = value
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", choices=["gurobi", "highs"], default="highs")
    ap.add_argument("--scales", nargs="+", default=["small", "medium"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--config", type=Path)
    opts = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    base = load_config(opts.config)
    rows, summary = [], []
    for name, overrides in SETTINGS.items():
        cfg = with_overrides(base, overrides)
        costs = Costs.from_config(cfg)
        for scale in opts.scales:
            group = []
            for seed in _parse_seeds(opts.seeds):
                for kind in ("aog", "closure"):
                    inst = generate_instance(scale, seed, kind, cfg=cfg)
                    sol = price_and_branch(
                        inst, costs, backend=opts.solver, settings=CGSettings.from_config(cfg)
                    )
                    group.append(
                        {
                            "setting": name,
                            "scale": scale,
                            "instance": inst.name,
                            "lp_fractional": int(sol.cg.fractional),
                            "gap": round(sol.gap, 8),
                            "lb": round(sol.lb, 6),
                            "z_pb": round(sol.z_pb, 6),
                        }
                    )
            rows += group
            n = len(group)
            summary.append(
                {
                    "setting": name,
                    "scale": scale,
                    "instances": n,
                    "lp_fractional": sum(r["lp_fractional"] for r in group),
                    "gap_positive": sum(r["gap"] > 1e-9 for r in group),
                    "mean_gap": round(sum(r["gap"] for r in group) / n, 6),
                }
            )
            s = summary[-1]
            print(
                f"{name:28s} {scale:6s} fractional LP {s['lp_fractional']:2d}/{n}  "
                f"gap>0 {s['gap_positive']:2d}/{n}  mean gap {100 * s['mean_gap']:.2f}%",
                flush=True,
            )
    RESULTS.mkdir(exist_ok=True)
    for path, data in (
        (RESULTS / "difficulty.csv", rows),
        (RESULTS / "difficulty_summary.csv", summary),
    ):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
