"""All plotting (matplotlib), saved as SVG on a white background.

Okabe-Ito colors; anything shown by color is also shown by fill, line style or a label. Nothing here
recomputes model logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

from recovery.instance import Costs, Instance, hhmm  # noqa: E402
from recovery.network import Network  # noqa: E402

# Okabe-Ito
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE, YELLOW = "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
INK, MUTED, GRID, BAND = "#222222", "#6b6b6b", "#e3e3e3", "#f3f3f3"
FADE = 0.15

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",  # keep text as text in the SVG
        "svg.hashsalt": "aircraft-recovery",  # stable ids, so regenerated SVGs diff cleanly
    }
)


def signed(x: float) -> str:
    """+1,234 / −1,234 / 0, with a typographic minus."""
    if abs(x) < 0.5:
        return "0"
    return f"{x:+,.0f}".replace("-", "−")


def save(fig: Figure, path: str | Path) -> None:
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def _time_axis(ax: plt.Axes, lo: float, hi: float, step: int = 30) -> None:
    """Ticks every `step` minutes labelled HH:MM; over long spans only full hours get a label."""
    first = int(lo // step + 1) * step
    ticks = list(range(first, int(hi) + 1, step))
    crowded = hi - lo > 6 * 60
    ax.set_xticks(ticks)
    ax.set_xticklabels([hhmm(t) if not crowded or t % 60 == 0 else "" for t in ticks])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_connection_network(
    instance: Instance,
    network: Network,
    aircraft: str,
    *,
    flights: Sequence[str] | None = None,
    delays: Sequence[int] | None = None,
    route: Sequence[tuple[str, int]] | None = None,
    duals: Any = None,
    costs: Costs | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """The connection network of one aircraft over clock time.

    `route` highlights one route given as (flight, delay) pairs ([] for the empty route). `duals`
    annotates reachable copies with their pricing weight and sinks with -mu_s.
    """
    net, k = network, aircraft
    ki = net.k_index[k]
    route = None if route is None else list(route)
    f_ids = [f.id for f in net.F] if flights is None else list(flights)
    d_set = list(net.D) if delays is None else list(delays)
    for f, d in route or []:
        if f not in f_ids:
            f_ids.append(f)
        if d not in d_set:
            d_set.append(d)
    lanes = sorted((net.F[net.f_index[f]] for f in f_ids), key=lambda f: (f.dep, f.id))
    d_set = sorted(d_set)
    if len(lanes) > 12 or len(d_set) > 4:
        raise ValueError(
            f"{len(lanes)} flights and {len(d_set)} delay rows requested; the figure is for "
            "explanation and draws at most 12 flights and 4 delays: pass flights= and delays="
        )

    # Layout: one lane per flight, one row per delay; y grows downwards from 0.
    row_h, gap = 1.0, 0.7
    row_y: dict[tuple[str, int], float] = {}
    lane_span: list[tuple[float, float]] = []
    y = 0.0
    for f in lanes:
        top = y
        for d in d_set:
            row_y[f.id, d] = -(y + row_h / 2)
            y += row_h
        lane_span.append((-top + 0.15, -y - 0.15))
        y += gap
    bottom = -(y - gap)
    drawn = {
        (f, d): net.copy_index[f, d] for (f, d) in row_y if (f, d) in net.copy_index
    }  # closure-deleted copies have no node
    reach = net.reachable[:, ki]
    on_route = set() if route is None else {net.copy_index[c] for c in route}
    path = [] if route is None else [net.copy_index[c] for c in route]
    route_arcs = set(zip(path[:-1], path[1:], strict=True))
    fade = route is not None

    def alpha(highlight: bool) -> float:
        return 1.0 if (highlight or not fade) else FADE

    rho, s_k = int(net.src_time[ki]), net.S[net.src_station[ki]]
    xs = [net.dep[v] for v in drawn.values()] + [rho]
    x_lo = (min(xs) - 25) // 30 * 30
    x_data = max([net.ready[v] for v in drawn.values()] + [rho + 60])
    stations = sorted({net.S[net.dest[v]] for v in drawn.values()} | {s_k})
    chan0, chan_dx = x_data + 12, 9
    box_x = chan0 + chan_dx * len(stations) + 8
    box_w = 58
    x_empty = box_x + box_w + 8  # the empty route runs along the top, then down the right edge
    x_hi = x_empty + 6

    if ax is None:
        height = 1.2 + 0.24 * (-bottom + 2.2)
        fig, ax = plt.subplots(figsize=(10.5, height))
    else:
        fig = ax.figure
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(bottom - 1.1, 2.4)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    _time_axis(ax, x_lo, x_data)

    tw = ax.get_yaxis_transform()  # x in axes coordinates, y in data
    for f, (y0, y1) in zip(lanes, lane_span, strict=True):
        ax.add_patch(Rectangle((x_lo, y1), x_data + 6 - x_lo, y0 - y1, color=BAND, lw=0, zorder=0))
        mid = (y0 + y1) / 2
        ax.text(
            -0.075,
            mid + 0.25,
            f"{f.id}  {f.origin}→{f.dest}",
            transform=tw,
            ha="right",
            va="center",
            color=INK,
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            -0.075,
            mid - 0.45,
            f"{hhmm(f.dep)} · {f.blk} min",
            transform=tw,
            ha="right",
            va="center",
            color=MUTED,
            fontsize=8,
        )
        for d in d_set:
            ax.text(
                -0.012,
                row_y[f.id, d],
                f"+{d}",
                transform=tw,
                ha="right",
                va="center",
                color=MUTED,
                fontsize=8,
            )

    weights = None
    if duals is not None:
        from recovery.pricing import weights as pricing_weights

        W, mu, _ = pricing_weights(net, costs or Costs.from_config(), duals)
        weights = (W[:, ki], mu)

    # Flight copies: bar in the air, dashed turnaround.
    bar_h = 0.46
    for (f, d), v in drawn.items():
        yv, hl = row_y[f, d], v in on_route
        a = alpha(hl)
        color = GREEN if hl else BLUE
        x0, x1, x2 = net.dep[v], net.arr[v], net.ready[v]
        if reach[v]:
            ax.add_patch(
                Rectangle(
                    (x0, yv - bar_h / 2),
                    x1 - x0,
                    bar_h,
                    facecolor=color,
                    edgecolor=color,
                    lw=1.6 if hl else 0.6,
                    alpha=a,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x0, yv - bar_h / 2),
                    x1 - x0,
                    bar_h,
                    facecolor="none",
                    edgecolor=MUTED,
                    lw=0.8,
                    ls=(0, (3, 2)),
                    alpha=a,
                    zorder=3,
                )
            )
        ax.plot(
            [x1, x2],
            [yv, yv],
            color=color if reach[v] else MUTED,
            lw=1.6 if hl else 0.9,
            ls=(0, (2, 1.5)),
            alpha=a,
            zorder=3,
        )
        if weights is not None and reach[v]:
            ax.text(
                (x0 + x1) / 2,
                yv + bar_h / 2 + 0.05,
                signed(weights[0][v]),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
                alpha=a,
                zorder=6,
            )

    # Source, ready time and start arcs.
    src_y = 1.35
    src_hl = route is not None and len(path) == 0  # the empty route starts at the source too
    ax.plot([rho, rho], [bottom - 0.4, src_y], color=VERMILLION, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.text(
        rho + 4,
        bottom - 0.55,
        f"ρ = {hhmm(rho)}",
        color=INK,
        fontsize=8,
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        zorder=6,
    )
    ax.plot([rho], [src_y], "o", ms=8, color=INK, zorder=7, mec="white", mew=1.5)
    ax.text(
        rho + 6, src_y + 0.3, f"src: {s_k}, ready {hhmm(rho)}", color=INK, fontsize=8.5, va="bottom"
    )
    for (f, d), v in drawn.items():
        if net.start_ok[v, ki]:
            hl = bool(path) and v == path[0]
            ax.add_patch(
                FancyArrowPatch(
                    (rho, src_y),
                    (net.dep[v], row_y[f, d]),
                    connectionstyle="arc3,rad=0.12",
                    arrowstyle="-",
                    color=GREEN if hl else INK,
                    lw=1.8 if hl else 0.5,
                    alpha=alpha(hl) * (1 if hl else 0.7),
                    zorder=2,
                )
            )

    # Connection arcs of the shared copy graph whose tail is reachable.
    for u, v in net.arcs():
        if not reach[u]:
            continue
        cu, cv = (
            (net.F[net.flight_of[u]].id, int(net.delay[u])),
            (net.F[net.flight_of[v]].id, int(net.delay[v])),
        )
        if cu not in drawn or cv not in drawn:
            continue
        hl = (u, v) in route_arcs
        ax.add_patch(
            FancyArrowPatch(
                (net.ready[u], row_y[cu]),
                (net.dep[v], row_y[cv]),
                connectionstyle="arc3,rad=-0.18",
                arrowstyle="-|>",
                mutation_scale=9,
                color=GREEN if hl else INK,
                lw=1.8 if hl else 0.7,
                alpha=alpha(hl),
                zorder=4,
                shrinkA=1,
                shrinkB=1,
            )
        )

    # End-of-day sinks: one box per station, dotted sink arcs along per-station channels.
    span = max(-bottom, 1.0)
    box_y = {s: -(span * (i + 0.5) / len(stations)) for i, s in enumerate(stations)}
    chan_x = {s: chan0 + chan_dx * i for i, s in enumerate(stations)}
    end_station = None
    if route is not None:
        end_station = net.S[net.dest[path[-1]]] if path else s_k
    for s in stations:
        hl = s == end_station
        ax.add_patch(
            Rectangle(
                (box_x, box_y[s] - 0.45),
                box_w,
                0.9,
                facecolor="white",
                edgecolor=GREEN if hl else INK,
                lw=2.0 if hl else 0.8,
                alpha=alpha(hl),
                zorder=5,
            )
        )
        ax.text(
            box_x + box_w / 2,
            box_y[s],
            f"ends {s}",
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            alpha=alpha(hl),
            zorder=6,
        )
        if weights is not None:
            ax.text(
                box_x + box_w / 2,
                box_y[s] - 0.6,
                f"−μ = {signed(-weights[1][net.s_index[s]])}",
                ha="center",
                va="top",
                fontsize=6.5,
                color=INK,
                alpha=alpha(hl),
                zorder=6,
            )
    for (f, d), v in drawn.items():
        if not reach[v]:
            continue
        s = net.S[net.dest[v]]
        hl = bool(path) and v == path[-1]
        xs_ = [net.ready[v], chan_x[s], chan_x[s], box_x]
        ys_ = [row_y[f, d], row_y[f, d], box_y[s], box_y[s]]
        ax.plot(
            xs_,
            ys_,
            color=GREEN if hl else MUTED,
            lw=1.8 if hl else 0.7,
            ls=":",
            alpha=alpha(hl),
            zorder=1 if not hl else 4,
        )
    empty_style = {
        "color": GREEN if src_hl else INK,
        "lw": 1.8 if src_hl else 0.8,
        "alpha": alpha(src_hl),
        "zorder": 2,
    }
    ax.plot([rho, x_empty, x_empty], [src_y, src_y, box_y[s_k]], ls=(0, (6, 3)), **empty_style)
    ax.add_patch(
        FancyArrowPatch(
            (x_empty, box_y[s_k]),
            (box_x + box_w, box_y[s_k]),
            arrowstyle="-|>",
            mutation_scale=9,
            shrinkA=0,
            shrinkB=0,
            **empty_style,
        )
    )

    handles = [
        Patch(facecolor=BLUE, edgecolor=BLUE, label="copy reachable from src"),
        Patch(facecolor="none", edgecolor=MUTED, ls=(0, (3, 2)), label="copy not reachable"),
        Line2D([], [], color=BLUE, ls=(0, (2, 1.5)), label="turnaround (TAT)"),
        Line2D([], [], color=INK, lw=0.7, marker=">", ms=4, label="connection arc"),
        Line2D([], [], color=INK, lw=0.5, label="start arc"),
        Line2D([], [], color=MUTED, ls=":", label="sink arc"),
        Line2D([], [], color=INK, ls=(0, (6, 3)), label="empty route"),
        Line2D([], [], color=VERMILLION, ls=(0, (5, 3)), label="ready time ρ"),
    ]
    if route is not None:
        handles.append(Line2D([], [], color=GREEN, lw=2.2, label="highlighted route"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=2.4,
    )
    return fig


def plot_convergence(log: Sequence[Any], *, ax: plt.Axes | None = None) -> Figure:
    """z_RMP and the best Lagrangian bound so far per column-generation iteration.

    Early bounds can be very negative, so the y-range is clipped below.
    """
    its = [e.iteration for e in log]
    z = [e.z_rmp for e in log]
    lb = [e.lb for e in log]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
    else:
        fig = ax.figure
    final = z[-1]
    spread = max(max(z) - final, 1.0)
    floor = final - 0.6 * spread  # clip: bounds further down would flatten the plot
    lb_plot = [max(v, floor) for v in lb]
    ax.fill_between(
        its, lb_plot, z, color=SKY, alpha=0.25, lw=0, step="post", label="gap between them"
    )
    ax.step(its, z, where="post", color=BLUE, lw=1.6, label="$z_{RMP}$")
    ax.step(
        its,
        lb_plot,
        where="post",
        color=VERMILLION,
        lw=1.6,
        ls=(0, (5, 2)),
        label="best Lagrangian bound so far",
    )
    if log[-1].lb >= final - 1e-6 * max(1.0, abs(final)):
        ax.plot([its[-1]], [final], "o", ms=8, color=INK, mec="white", mew=1.5, zorder=5)
        ax.annotate(
            f"converged: $z_{{LP}}$ = {final:,.0f}",
            (its[-1], final),
            xytext=(-8, 14),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=INK,
        )
    if min(lb) < floor:
        ax.text(
            0.01,
            0.02,
            "earlier bounds lie below the axis (clipped)",
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            va="bottom",
        )
    ax.set_ylim(floor, max(z) + 0.05 * spread)
    ax.set_xlim(0, its[-1] + max(1, 0.02 * its[-1]))
    ax.set_xlabel("column generation iteration", color=INK)
    ax.set_ylabel("cost", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return fig


def _bar_label(ax: plt.Axes, x0: float, x1: float, y: float, text: str, fill: str | None) -> None:
    """Flight id inside its bar when it fits, on a small box so hatching does not cross it."""
    if x1 - x0 < 38:
        return
    dark = fill in (BLUE, VERMILLION)
    box = None if fill is None else {"facecolor": fill, "edgecolor": "none", "pad": 0.6}
    ax.text(
        (x0 + x1) / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=6.5,
        color="white" if dark else INK,
        bbox=box,
        zorder=6,
    )


def plot_recovery_gantt(instance: Instance, plan: Any, *, title: str | None = None) -> Figure:
    """Planned (top) and recovered (bottom) day of every aircraft.

    Delayed flights keep a thin outline at the original slot, swapped ones are hatched and cancelled
    ones sit in their own row. The AOG or closure window is shaded and a line marks t0.
    """
    from recovery.instance import AOG, Closure, planned_rotations

    K = list(instance.K)
    row = {k: i for i, k in enumerate(K)}
    t0, by_id = instance.t0, instance.flight
    rot = planned_rotations(instance)
    lo = min(f.dep for f in instance.flights) - 30
    hi = max(f.arr for f in instance.flights) + 30
    for route in plan.routes.values():
        for fid, d in route:
            hi = max(hi, by_id[fid].arr + d + 30)
    h = 0.62
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 1.4 + 0.42 * (2 * len(K) + 1)),
        gridspec_kw={"height_ratios": [len(K), len(K) + 1], "hspace": 0.12},
    )
    dis = instance.disruption
    cancel_row = len(K)
    for ax, panel in ((top, "planned"), (bottom, "recovered")):
        n_rows = len(K) + (panel == "recovered")
        ax.set_ylim(n_rows - 0.4, -0.6)
        labels = K + (["cancelled"] if panel == "recovered" else [])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.tick_params(axis="y", length=0)
        for side in ("right", "top", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(panel, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        if isinstance(dis, AOG):
            y = row[dis.aircraft]
            ax.add_patch(
                Rectangle(
                    (dis.t1, y - 0.45),
                    dis.t2 - dis.t1,
                    0.9,
                    facecolor=VERMILLION,
                    alpha=0.12,
                    hatch="xx",
                    edgecolor=VERMILLION,
                    lw=0,
                    zorder=1,
                )
            )
            ax.text(dis.t1 + 4, y - 0.47, "AOG", fontsize=7.5, color=INK, va="bottom", zorder=7)
        elif isinstance(dis, Closure):
            ax.axvspan(dis.ta, dis.tb, color=VERMILLION, alpha=0.10, lw=0, zorder=0)
            ax.text(
                (dis.ta + dis.tb) / 2,
                -0.55,
                f"{dis.station} closed",
                fontsize=7.5,
                color=INK,
                ha="center",
                va="bottom",
            )
        ax.axvline(t0, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=5)
        if panel == "planned":
            ax.text(t0 + 3, -0.55, f"t0 = {hhmm(t0)}", fontsize=7.5, color=INK, va="bottom")

    for k, flights in rot.items():
        for f in flights:
            top.add_patch(
                Rectangle(
                    (f.dep, row[k] - h / 2),
                    f.blk,
                    h,
                    facecolor=BLUE,
                    edgecolor="white",
                    lw=0.8,
                    zorder=3,
                )
            )
            _bar_label(top, f.dep, f.arr, row[k], f.id, BLUE)

    def draw(k: str, f, d: int, frozen: bool = False) -> None:
        y, x0 = row[k], f.dep + d
        swapped, delayed = f.orig != k, d > 0
        color = ORANGE if delayed else BLUE
        if delayed:  # thin outline at the original slot
            bottom.add_patch(
                Rectangle(
                    (f.dep, y - h / 2),
                    f.blk,
                    h,
                    facecolor="none",
                    edgecolor=MUTED,
                    lw=0.7,
                    ls=(0, (2, 1.5)),
                    zorder=2,
                )
            )
        bottom.add_patch(
            Rectangle(
                (x0, y - h / 2),
                f.blk,
                h,
                facecolor=color,
                edgecolor="white",
                lw=0.8,
                hatch="////" if swapped else None,
                alpha=0.55 if frozen else 1.0,
                zorder=3,
            )
        )
        if not frozen:
            _bar_label(bottom, x0, x0 + f.blk, y, f.id, color)
        else:
            _bar_label(bottom, x0, x0 + f.blk, y, f.id, None)

    for f in instance.flights:
        if f.dep < t0:
            draw(f.orig, f, 0, frozen=True)
    for k, route in plan.routes.items():
        for fid, d in route:
            draw(k, by_id[fid], d)
    for fid in plan.cancelled:
        f = by_id[fid]
        bottom.add_patch(
            Rectangle(
                (f.dep, cancel_row - h / 2),
                f.blk,
                h,
                facecolor="none",
                edgecolor=VERMILLION,
                lw=1.0,
                ls=(0, (3, 1.5)),
                zorder=3,
            )
        )
        _bar_label(bottom, f.dep, f.arr, cancel_row, f.id, None)

    bottom.set_xlim(lo, hi)
    _time_axis(bottom, lo, hi, step=60)
    top.grid(axis="x", color=GRID, linewidth=0.6)
    top.set_axisbelow(True)
    handles = [
        Patch(facecolor=BLUE, edgecolor="white", label="on time"),
        Patch(facecolor=ORANGE, edgecolor="white", label="delayed"),
        Patch(facecolor="none", edgecolor=MUTED, ls=(0, (2, 1.5)), label="original slot"),
        Patch(facecolor=BLUE, edgecolor="white", hatch="////", label="swapped"),
        Patch(facecolor="none", edgecolor=VERMILLION, ls=(0, (3, 1.5)), label="cancelled"),
        Patch(facecolor=BLUE, alpha=0.55, edgecolor="white", label="flown before t0"),
    ]
    if isinstance(dis, AOG):
        handles.append(
            Patch(
                facecolor=VERMILLION,
                alpha=0.12,
                hatch="xx",
                edgecolor=VERMILLION,
                lw=0,
                label="AOG window",
            )
        )
    elif isinstance(dis, Closure):
        handles.append(Patch(facecolor=VERMILLION, alpha=0.10, label="closure window"))
    bottom.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    if title:
        y = top.get_position().y1 + 0.035
        fig.text(top.get_position().x0, y, title, ha="left", va="bottom", fontsize=10, color=INK)
    return fig


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _style(ax: plt.Axes) -> None:
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_scale(rows: Sequence[dict[str, Any]]) -> Figure:
    """Solve time per instance against flights to re-plan."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = [int(r["flights"]) for r in rows]
    ax.scatter(
        x,
        [float(r["total_s"]) for r in rows],
        s=22,
        color=BLUE,
        marker="o",
        edgecolor="white",
        linewidth=0.6,
        label="column generation + price-and-branch",
        zorder=3,
    )
    solved = [r for r in rows if r.get("compact_status") == "optimal"]
    ax.scatter(
        [int(r["flights"]) for r in solved],
        [float(r["compact_s"]) for r in solved],
        s=26,
        color=VERMILLION,
        marker="s",
        edgecolor="white",
        linewidth=0.6,
        label="compact MILP on HiGHS (solved within the limit)",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("flights to re-plan |F|", color=INK)
    ax.set_ylabel("solve time (s, log scale)", color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    return fig


def plot_anytime(rows: Sequence[dict[str, Any]]) -> Figure:
    """Plan cost and lower bound relative to z_LP against the time limit."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    limits = sorted({float(r["time_limit"]) for r in rows})
    by_inst: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_inst.setdefault(r["instance"], []).append(r)
    for series, key, color, ls, marker in (
        ("plan cost z_PB / z_LP", "z_pb", BLUE, "-", "o"),
        ("lower bound / z_LP", "lb", VERMILLION, (0, (5, 2)), "s"),
    ):
        for rs in by_inst.values():
            rs = sorted(rs, key=lambda r: float(r["time_limit"]))
            ax.plot(
                [float(r["time_limit"]) for r in rs],
                [float(r[key]) / float(r["z_lp"]) for r in rs],
                color=color,
                lw=0.6,
                ls=ls,
                alpha=0.25,
            )
        med = [
            _median([float(r[key]) / float(r["z_lp"]) for r in rows if float(r["time_limit"]) == t])
            for t in limits
        ]
        ax.plot(limits, med, color=color, lw=2, ls=ls, marker=marker, ms=5, label=series)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(limits)
    ax.set_xticklabels([f"{t:g}" for t in limits])
    ax.set_xlabel("time limit for column generation + MIP (s, log scale)", color=INK)
    ax.set_ylabel("relative to $z_{LP}$", color=INK)
    lowest = min(float(r["lb"]) / float(r["z_lp"]) for r in rows)
    highest = max(float(r["z_pb"]) / float(r["z_lp"]) for r in rows)
    ax.set_ylim(max(0.0, lowest) - 0.05, 1.08 * highest)
    if lowest < 0:  # early Lagrangian bounds can be far below zero
        ax.text(
            0.01,
            0.02,
            "bounds below the axis are clipped",
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            va="bottom",
        )
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return fig


def plot_grid_sweep(rows: Sequence[dict[str, Any]], key: str, label: str, ref: int) -> Figure:
    """Plan cost, gap and solve time against one delay-grid parameter."""
    values = sorted({int(r[key]) for r in rows})
    ref_cost = {r["instance"]: float(r["z_pb"]) for r in rows if int(r[key]) == ref}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    panels = (
        (
            "plan cost / cost at " + label + f" = {ref}",
            lambda r: float(r["z_pb"]) / ref_cost[r["instance"]],
            False,
        ),
        ("gap (z_PB − LB) / z_PB, %", lambda r: 100 * float(r["gap"]), False),
        ("solve time (s, log scale)", lambda r: float(r["total_s"]), True),
    )
    for ax, (title, fn, log) in zip(axes, panels, strict=True):
        med = [_median([fn(r) for r in rows if int(r[key]) == v]) for v in values]
        for inst in ref_cost:
            rs = sorted((r for r in rows if r["instance"] == inst), key=lambda r: int(r[key]))
            ax.plot([int(r[key]) for r in rs], [fn(r) for r in rs], color=BLUE, lw=0.6, alpha=0.25)
        ax.plot(values, med, color=BLUE, lw=2, marker="o", ms=5)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(values)
        ax.set_xlabel(label, color=INK)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _style(ax)
    fig.tight_layout()
    return fig
