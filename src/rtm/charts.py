"""Matplotlib charts shared by the dashboard, the README figures and the one-page PDF.

Each route keeps one colour everywhere (colour follows the route, never its
rank). The three hues are the first three slots of a colour-blind-checked
categorical palette; dimensions and low/high cases use one-hue ramps so they
never compete with route identity.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from rtm import fmt
from rtm.config import DIMENSIONS, Scenario
from rtm.finance import Financials
from rtm.scoring import ScoreResult

ROUTE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
DIMENSION_SHADES = {"velocity": "#1c5cab", "margin": "#3987e5", "robustness": "#9ec5f4"}
DIMENSION_TEXT = {"velocity": "#ffffff", "margin": "#ffffff", "robustness": "#0b0b0b"}
LOW_SHADE, HIGH_SHADE = "#86b6ef", "#1c5cab"
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8f8d86", "#e4e2dc", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    }
)


def route_colours(scenario: Scenario) -> dict[str, str]:
    """Fixed colour per route, in the order routes appear in the scenario file."""
    return dict(zip(scenario.route_keys, ROUTE_COLOURS, strict=False))


def _money_axis(ax: Axes, currency: str) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.money(v, currency)))


def plot_scores(ax: Axes, scenario: Scenario, result: ScoreResult) -> None:
    """Horizontal stacked bars: each route's total split into weighted V, M and R points."""
    order = result.ranking[::-1]  # winner on top
    left = np.zeros(len(order))
    for dim in DIMENSIONS:
        points = result.dimension_scores.loc[order, dim].to_numpy() * result.weights[dim]
        bars = ax.barh(
            [scenario.label(k) for k in order],
            points,
            left=left,
            color=DIMENSION_SHADES[dim],
            edgecolor=SURFACE,
            linewidth=2,
            height=0.6,
            label=f"{dim.title()} ({fmt.pct(result.weights[dim])})",
        )
        ax.bar_label(
            bars,
            labels=[f"{p:.0f}" if p >= 5 else "" for p in points],
            label_type="center",
            color=DIMENSION_TEXT[dim],
            fontsize=8,
        )
        left += points
    for y, total in enumerate(left):
        ax.text(total + 1, y, f"{total:.1f}", va="center", color=INK, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Weighted VMR score (0-100)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("VMR score by route")


def plot_cash(ax: Axes, scenario: Scenario, fin: Financials, downside: bool = False) -> None:
    """Cumulative cash per route over the horizon, base case solid, downside dashed."""
    colours = route_colours(scenario)
    for key in scenario.route_keys:
        base = fin.base[key]
        ax.plot(base.months, base.cumulative_cash, color=colours[key], label=scenario.label(key))
        if downside:
            d = fin.downside[key]
            ax.plot(d.months, d.cumulative_cash, color=colours[key], linestyle="--", linewidth=1.5)
        be = base.breakeven_month
        if be is not None:
            ax.plot(
                be,
                base.cumulative_cash[be - 1],
                "o",
                color=colours[key],
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
    ax.axhline(0, color=INK_3, linewidth=1)
    _money_axis(ax, scenario.currency)
    ax.set_xlim(1, scenario.horizon_months)
    ax.set_xlabel("Month")
    title = "Cumulative cash (dots mark breakeven"
    ax.set_title(title + ("; dashed = downside case)" if downside else ")"))
    ax.legend(loc="lower left", fontsize=8)


def plot_revenue(ax: Axes, scenario: Scenario, fin: Financials) -> None:
    """Monthly end-customer revenue ramp per route."""
    colours = route_colours(scenario)
    for key in scenario.route_keys:
        f = fin.base[key]
        ax.plot(f.months, f.revenue, color=colours[key], label=scenario.label(key))
    _money_axis(ax, scenario.currency)
    ax.set_xlim(1, scenario.horizon_months)
    ax.set_xlabel("Month")
    ax.set_title("Monthly customer revenue")
    ax.legend(loc="upper left", fontsize=8)


def plot_weight_sweep(
    axes: list[Axes], scenario: Scenario, result: ScoreResult, sweep: pd.DataFrame
) -> None:
    """One panel per dimension: route totals as that dimension's weight moves 0 to 1."""
    colours = route_colours(scenario)
    for ax, dim in zip(axes, DIMENSIONS, strict=True):
        panel = sweep[sweep.dimension == dim]
        for key in scenario.route_keys:
            line = panel[panel.route == key]
            ax.plot(line.weight, line.total, color=colours[key], label=scenario.label(key))
        ax.axvline(result.weights[dim], color=INK_3, linestyle=":", linewidth=1.2)
        ax.text(
            result.weights[dim],
            1,
            " default",
            color=INK_2,
            fontsize=8,
            transform=ax.get_xaxis_transform(),
            va="top",
        )
        winners = panel.drop_duplicates("weight")
        changes = winners[winners.winner != winners.winner.shift()].iloc[1:]
        for w in changes.weight:
            ax.axvline(w, color=INK, linewidth=0.8, alpha=0.4)
        ax.set_title(f"{dim.title()} weight")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.pct(v)))
    axes[0].set_ylabel("Total score")
    axes[0].legend(loc="lower left", fontsize=8)


def plot_tornado(ax: Axes, scenario: Scenario, table: pd.DataFrame, top: int = 10) -> None:
    """Winner's lead over the next-best route when each input moves -/+ the swing."""
    rows = table.head(top).iloc[::-1]
    base = table.attrs["base_lead"]
    y = np.arange(len(rows))
    swing = fmt.pct(scenario.sensitivity.tornado_swing)
    for col, shade, label in (
        ("lead_low", LOW_SHADE, f"Input -{swing}"),
        ("lead_high", HIGH_SHADE, f"Input +{swing}"),
    ):
        ax.barh(
            y,
            rows[col] - base,
            left=base,
            color=shade,
            height=0.6,
            label=label,
            edgecolor=SURFACE,
            linewidth=1,
        )
    ax.axvline(base, color=INK_2, linewidth=1)
    ax.axvline(0, color="#e34948", linewidth=1.2, linestyle="--")
    ax.text(0, -0.75, "ranking flips ", color="#e34948", fontsize=8, ha="right", va="center")
    ax.set_ylim(-1.1, len(rows) - 0.5)
    ax.set_yticks(y, rows.label)
    ax.grid(axis="y", visible=False)
    winner = scenario.label(table.attrs["winner"])
    ax.set_xlabel(f"{winner} lead over next-best route (points; base = {base:.1f})")
    ax.set_title("What moves the ranking most")
    ax.legend(loc="lower right", fontsize=8)


def plot_weight_map(ax: Axes, scenario: Scenario, result: ScoreResult, grid: pd.DataFrame) -> None:
    """Triangle of every weight mix, each point coloured by the route that wins there."""
    colours = route_colours(scenario)
    # Barycentric -> 2D: velocity at bottom-left, margin at bottom-right, robustness at top.
    x = grid.margin + 0.5 * grid.robustness
    y = grid.robustness * np.sqrt(3) / 2
    for key in scenario.route_keys:
        mask = grid.winner == key
        share = mask.mean()
        ax.scatter(
            x[mask],
            y[mask],
            s=30,
            color=colours[key],
            edgecolor=SURFACE,
            linewidth=1,
            label=f"{scenario.label(key)} ({fmt.pct(share)})",
        )
    w = result.weights
    ax.plot(
        w["margin"] + 0.5 * w["robustness"],
        w["robustness"] * np.sqrt(3) / 2,
        "*",
        color=INK,
        markersize=14,
        markeredgecolor=SURFACE,
        label="Default weights",
    )
    for label, (tx, ty, ha) in {
        "Velocity": (0, -0.06, "center"),
        "Margin": (1, -0.06, "center"),
        "Robustness": (0.5, np.sqrt(3) / 2 + 0.04, "center"),
    }.items():
        ax.text(tx, ty, f"100% {label}", ha=ha, color=INK_2, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Which route wins for each weight mix (share of mixes)", pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(0.62, 1.0), fontsize=8)


def key_chart(scenario: Scenario, result: ScoreResult, fin: Financials) -> Figure:
    """The README headline figure: score ranking next to the cash curves."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=[1, 1.15])
    plot_scores(left, scenario, result)
    plot_cash(right, scenario, fin)
    fig.tight_layout()
    return fig
