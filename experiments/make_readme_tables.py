"""Write the result tables into README.md between <!-- results:NAME:start/end --> markers.

python experiments/make_readme_tables.py
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from recovery.instance import load_config  # noqa: E402

RESULTS = ROOT / "results"
README = ROOT / "README.md"
SCALES = ["tiny", "small", "medium", "large"]


def read(name: str) -> list[dict[str, str]]:
    with open(RESULTS / name, newline="") as fh:
        return list(csv.DictReader(fh))


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def sec(x: float) -> str:
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    marks = {"l": "---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def by_group(rows):
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r["scale"], r["disruption"]), []).append(r)
    return sorted(groups.items(), key=lambda g: (SCALES.index(g[0][0]), g[0][1]))


def f(r: dict[str, str], key: str) -> float:
    return float(r[key])


def scale_table(rows) -> str:
    out = []
    for (scale, dis), rs in by_group(rows):
        mean = lambda key, rs=rs: statistics.fmean(f(r, key) for r in rs)  # noqa: E731
        out.append(
            [
                f"{scale} / {dis}",
                str(len(rs)),
                f"{mean('aircraft'):.0f}",
                f"{mean('flights'):.0f}",
                f"{mean('copies'):.0f}",
                f"{mean('cg_iterations'):.0f}",
                f"{mean('columns_generated'):.0f}",
                sec(mean("pricing_s")),
                sec(mean("master_s")),
                f"{sec(mean('total_s'))} / {sec(max(f(r, 'total_s') for r in rs))}",
                f"{sum(r['cg_stop'] in ('converged', 'zero') for r in rs)}/{len(rs)}",
                pct(mean("gap")),
                pct(max(f(r, "gap") for r in rs)),
            ]
        )
    return table(
        [
            "scale / disruption",
            "instances",
            "aircraft",
            "flights to re-plan",
            "copies",
            "CG iterations",
            "columns generated",
            "pricing s",
            "master s",
            "total s (mean / max)",
            "CG converged",
            "mean gap",
            "max gap",
        ],
        out,
    )


def validation_table(rows) -> str:
    out = []
    for (scale, dis), rs in by_group(rows):
        attempted = [r for r in rs if r["compact_status"] not in ("skipped", "too large")]
        solved = [r for r in attempted if r["compact_status"] == "optimal"]
        too_large = sum(r["compact_status"] == "too large" for r in rs)
        same = sum(
            abs(f(r, "compact_z") - f(r, "z_pb")) <= 1e-6 * max(1, f(r, "z_pb")) for r in solved
        )
        below = sum(f(r, "compact_z") < f(r, "z_pb") - 1e-6 * max(1, f(r, "z_pb")) for r in solved)
        compact = f"{len(solved)}/{len(rs)}"
        if too_large:
            compact += f" ({too_large} too large)"
        if all(r["compact_status"] == "skipped" for r in rs):
            size = statistics.median(f(r, "compact_vars") for r in rs)
            compact = f"not run ({size / 1e6:.1f}M variables, median)"
        oks = [r[k] for r in rs for k in ("plan_ok", "compact_ok", "heur_ok") if r.get(k)]
        out.append(
            [
                f"{scale} / {dis}",
                f"{sum(int(r['lp_fractional']) for r in rs)}/{len(rs)}",
                f"{sum(f(r, 'gap') > 1e-9 for r in rs)}/{len(rs)}",
                compact,
                sec(statistics.fmean(f(r, "compact_s") for r in solved)) if solved else "–",
                f"{same} / {below}" if solved else "–",
                pct(statistics.fmean(f(r, "heur_gap") for r in rs)),
                f"{sum(int(v) for v in oks)}/{len(oks)}",
            ]
        )
    return table(
        [
            "scale / disruption",
            "LP fractional",
            "gap > 0",
            "compact solved",
            "compact s",
            "compact = z_PB / below z_PB",
            "heuristic gap",
            "saved plans pass checker",
        ],
        out,
    )


def missed_table(rows) -> str:
    """Instances where the compact MILP finds a cheaper plan than price-and-branch."""
    out = [
        [r["instance"], f"{f(r, 'lb'):,.0f}", f"{f(r, 'z_pb'):,.0f}", f"{f(r, 'compact_z'):,.0f}"]
        for r in rows
        if r["compact_status"] == "optimal"
        and f(r, "compact_z") < f(r, "z_pb") - 1e-6 * max(1.0, f(r, "z_pb"))
    ]
    return table(["instance", "LB = z_LP", "z_PB", "compact optimum"], out)


def difficulty_table(rows) -> str:
    return table(
        ["generator setting", "scale", "LP fractional", "gap > 0", "mean gap"],
        [
            [
                r["setting"],
                r["scale"],
                f"{r['lp_fractional']}/{r['instances']}",
                f"{r['gap_positive']}/{r['instances']}",
                pct(f(r, "mean_gap")),
            ]
            for r in rows
        ],
        "llrrr",
    )


def sweep_table(rows, key: str, label: str) -> str:
    values = sorted({int(r[key]) for r in rows})
    ref_value = values[-1] if key == "d_max" else int(load_config()["delays"]["step"])
    ref = {r["instance"]: f(r, "z_pb") for r in rows if int(r[key]) == ref_value}
    out = []
    for v in values:
        rs = [r for r in rows if int(r[key]) == v]
        out.append(
            [
                str(v),
                f"{statistics.median(f(r, 'copies') for r in rs):,.0f}",
                f"{statistics.median(f(r, 'z_pb') / ref[r['instance']] for r in rs):.3f}",
                pct(statistics.fmean(f(r, "gap") for r in rs)),
                sec(statistics.median(f(r, "total_s") for r in rs)),
            ]
        )
    return table(
        [
            label,
            "copies (median)",
            f"plan cost / cost at {ref_value} (median)",
            "mean gap",
            "solve s (median)",
        ],
        out,
    )


def anytime_table(rows) -> str:
    limits = sorted({float(r["time_limit"]) for r in rows})
    out = []
    for t in limits:
        rs = [r for r in rows if float(r["time_limit"]) == t]
        out.append(
            [
                f"{t:g}",
                f"{statistics.median(f(r, 'z_pb') / f(r, 'z_lp') for r in rs):.3f}",
                f"{statistics.median(f(r, 'lb') / f(r, 'z_lp') for r in rs):.3f}",
                f"{sum(r['cg_stop'] in ('converged', 'zero') for r in rs)}/{len(rs)}",
                sec(statistics.median(f(r, "total_s") for r in rs)),
            ]
        )
    return table(
        [
            "time limit s",
            "z_PB / z_LP (median)",
            "LB / z_LP (median)",
            "CG converged",
            "time used s (median)",
        ],
        out,
    )


def cg_by_hand_table() -> str:
    import cg_by_hand

    return cg_by_hand.table(cg_by_hand.run(load_config()["solver"]["backend"]))


def settings_block() -> str:
    """Cost parameters and the delay grid, from config.yaml."""
    cfg = load_config()
    c, d = cfg["costs"], cfg["delays"]
    return table(
        ["parameter", "value"],
        [
            ["delay, per passenger-minute", f"{c['c_delay']:g}"],
            [
                "cancellation, fixed + per passenger",
                f"{c['c_cancel_fixed']:,.0f} + {c['c_cancel_pax']:g}",
            ],
            ["swap, per flight", f"{c['c_swap']:,.0f}"],
            ["end-of-day shortfall, per aircraft", f"{c['c_pos']:,.0f}"],
            [
                "delay step Δ / largest delay D_max (min)",
                f"{d['step']} / {d['d_max_default']} (lower if the elementarity cap requires)",
            ],
            ["minimum turnaround τ (min)", f"{cfg['generator']['TAT']}"],
        ],
        "lr",
    )


def span(values) -> str:
    values = list(values)
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}–{hi}"


def ladder_block(rows) -> str:
    cfg = load_config()
    out = []
    for scale in SCALES:
        rs = [r for r in rows if r["scale"] == scale]
        if not rs:
            continue
        spec = cfg["scales"][scale]
        out.append(
            [
                scale,
                str(spec["aircraft"]),
                f"{spec['hubs']} + {spec['spokes']}",
                span(int(r["flights_planned"]) for r in rs),
                str(len(rs)),
            ]
        )
    return table(["scale", "aircraft", "hubs + spokes", "planned flights", "instances"], out)


def smoothing_table(rows) -> str:
    alphas = sorted({float(r["smoothing"]) for r in rows})
    base = {r["instance"]: f(r, "z_lp") for r in rows if float(r["smoothing"]) == 0.0}
    out = []
    for a in alphas:
        rs = [r for r in rows if float(r["smoothing"]) == a]
        diff = max(
            abs(f(r, "z_lp") - base[r["instance"]]) / max(1.0, base[r["instance"]]) for r in rs
        )
        out.append(
            [
                f"{a:g}",
                f"{statistics.median(f(r, 'cg_iterations') for r in rs):.0f}",
                f"{max(f(r, 'cg_iterations') for r in rs):.0f}",
                sec(statistics.median(f(r, "cg_s") for r in rs)),
                f"{sum(r['cg_stop'] in ('converged', 'zero') for r in rs)}/{len(rs)}",
                f"{diff:.1e}",
            ]
        )
    return table(
        [
            "smoothing α",
            "iterations (median)",
            "iterations (max)",
            "CG s (median)",
            "converged",
            "max relative z_LP difference to α = 0",
        ],
        out,
    )


def headline(rows) -> str:
    n = len(rows)
    largest = [r for r in rows if r["scale"] == "large"]
    converged = sum(r["cg_stop"] in ("converged", "zero") for r in rows)
    gaps = [f(r, "gap") for r in rows]
    return (
        f"On {n} generated instances ({min(int(r['aircraft']) for r in rows)} to "
        f"{max(int(r['aircraft']) for r in rows)} aircraft, up to "
        f"{max(int(r['flights']) for r in rows)} flights to re-plan), column generation "
        f"converged on {converged}, and the gap between the price-and-branch plan and the LP "
        f"lower bound, (z_PB - LB) / z_PB, averaged {pct(statistics.fmean(gaps))} "
        f"(at most {pct(max(gaps))}). "
        f"The {len(largest)} large instances took "
        f"{sec(statistics.median(f(r, 'total_s') for r in largest))} s in median and at most "
        f"{sec(max(f(r, 'total_s') for r in largest))} s with Gurobi."
    )


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- results:{name}:start -->\n).*?(<!-- results:{name}:end -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise KeyError(f"marker results:{name} not found in README.md")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> None:
    text = README.read_text()
    scale = read("scale.csv")
    smooth, anytime = read("tradeoff_smoothing.csv"), read("tradeoff_time.csv")
    dmax, step = read("tradeoff_dmax.csv"), read("tradeoff_step.csv")
    blocks = {
        "headline": headline(scale),
        "costs": settings_block(),
        "ladder": ladder_block(scale),
        "smoothing": smoothing_table(smooth),
        "scale": scale_table(scale),
        "validation": validation_table(scale),
        "missed": missed_table(scale),
        "difficulty": difficulty_table(read("difficulty_summary.csv")),
        "cgbyhand": cg_by_hand_table(),
        "anytime": anytime_table(anytime),
        "dmax": sweep_table(dmax, "d_max", "D_max (min)"),
        "step": sweep_table(step, "delay_step", "delay step Δ (min)"),
    }
    for name, body in blocks.items():
        if f"<!-- results:{name}:start -->" in text:
            text = replace_block(text, name, body)
    README.write_text(text)
    print(f"README.md: wrote {', '.join(blocks)}")


if __name__ == "__main__":
    main()
