"""Sensitivity analysis: when does the recommended route stop winning?

Four views:

* weight sweep: move one dimension weight from 0 to 1 (the other two keep
  their default ratio) and find where the top route changes;
* weight map: the winner for every weight combination on a grid;
* tornado: move each input +/- a fixed swing and measure the winner's lead;
* flip thresholds and downside case: how far an input has to move before
  the ranking changes, and how the routes rank under stress.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rtm.config import DIMENSIONS, Scenario
from rtm.finance import revenue_retention, run_financials
from rtm.scoring import ScoreResult, score_routes


def _line_weights(defaults: dict[str, float], dim: str, weight: float) -> dict[str, float]:
    """Weights with ``dim`` set to ``weight`` and the rest sharing 1 - weight in default ratio."""
    others = [d for d in DIMENSIONS if d != dim]
    rest = sum(defaults[d] for d in others)
    shares = {d: defaults[d] / rest if rest > 0 else 1 / len(others) for d in others}
    return {dim: weight, **{d: (1 - weight) * shares[d] for d in others}}


def _totals_for(result: ScoreResult, weights: list[dict[str, float]]) -> np.ndarray:
    """Total score of every route (columns) for many weight sets (rows) at once."""
    w = np.array([[ws[d] for d in DIMENSIONS] for ws in weights])
    return w @ result.dimension_scores[list(DIMENSIONS)].to_numpy().T


def _winners(result: ScoreResult, totals: np.ndarray) -> np.ndarray:
    """Top route per row; ties go to the route listed first, as in :func:`combine`."""
    return result.dimension_scores.index.to_numpy()[np.argmax(totals, axis=1)]


def weight_sweep(scenario: Scenario, result: ScoreResult) -> pd.DataFrame:
    """Total score of every route as each dimension weight moves from 0 to 1.

    Returns one row per (dimension, weight, route) with the route's total and
    the winner at that weight.
    """
    grid = np.linspace(0.0, 1.0, round(1 / scenario.sensitivity.weight_resolution) + 1)
    routes = list(result.dimension_scores.index)
    frames = []
    for dim in DIMENSIONS:
        totals = _totals_for(result, [_line_weights(result.weights, dim, w) for w in grid])
        winners = _winners(result, totals)
        frame = pd.DataFrame(totals, columns=routes).assign(weight=grid, winner=winners)
        frame = frame.melt(id_vars=["weight", "winner"], var_name="route", value_name="total")
        frames.append(frame.assign(dimension=dim))
    sweep = pd.concat(frames, ignore_index=True)
    return sweep[["dimension", "weight", "route", "total", "winner"]]


def weight_flips(scenario: Scenario, result: ScoreResult) -> pd.DataFrame:
    """Nearest weight, above and below the default, at which the top route changes.

    One row per dimension and direction where a flip exists.
    """
    sweep = weight_sweep(scenario, result)
    winners = sweep.drop_duplicates(["dimension", "weight"])
    rows = []
    for dim in DIMENSIONS:
        line = winners[winners.dimension == dim]
        default = result.weights[dim]
        above = line[(line.weight > default) & (line.winner != result.winner)]
        below = line[(line.weight < default) & (line.winner != result.winner)]
        if not above.empty:
            first = above.iloc[0]
            rows.append((dim, default, "up", first.weight, first.winner))
        if not below.empty:
            first = below.iloc[-1]
            rows.append((dim, default, "down", first.weight, first.winner))
    return pd.DataFrame(
        rows, columns=["dimension", "default_weight", "direction", "flip_weight", "new_winner"]
    )


def weight_map(scenario: Scenario, result: ScoreResult) -> pd.DataFrame:
    """Winning route for every weight combination on a grid over the simplex."""
    steps = round(1 / scenario.sensitivity.weight_map_step)
    weights = [
        {"velocity": i / steps, "margin": j / steps, "robustness": (steps - i - j) / steps}
        for i in range(steps + 1)
        for j in range(steps + 1 - i)
    ]
    totals = _totals_for(result, weights)
    ordered = np.sort(totals, axis=1)
    table = pd.DataFrame(weights)
    table["winner"] = _winners(result, totals)
    table["lead"] = ordered[:, -1] - ordered[:, -2]
    return table


def weight_space_share(scenario: Scenario, result: ScoreResult) -> pd.Series:
    """Share of all weight combinations in which each route ranks first."""
    wins = weight_map(scenario, result).winner.value_counts(normalize=True)
    return wins.reindex(scenario.route_keys, fill_value=0.0)


def _lead_of(route: str, totals: pd.Series) -> float:
    """Points by which ``route`` beats the best other route (negative if it loses)."""
    return float(totals[route] - totals.drop(route).max())


def tornado(scenario: Scenario) -> pd.DataFrame:
    """Move each input by +/- the tornado swing and record the base winner's lead.

    Inputs that are zero in the base case (a relative swing leaves them at zero)
    are skipped. Rows are sorted by how much the lead moves, largest first.
    """
    base = score_routes(scenario)
    winner, swing = base.winner, scenario.sensitivity.tornado_swing
    base_lead = base.lead
    rows = []
    for path in scenario.input_paths():
        value = scenario.get(path)
        if value == 0:
            continue
        leads, winners = [], []
        for factor in (1 - swing, 1 + swing):
            totals = score_routes(scenario.with_value(path, value * factor)).totals
            leads.append(_lead_of(winner, totals))
            winners.append(totals.index[0])
        rows.append(
            {
                "input": path,
                "label": scenario.input_label(path),
                "base_value": value,
                "lead_low": leads[0],
                "lead_high": leads[1],
                "winner_low": winners[0],
                "winner_high": winners[1],
                "range": abs(leads[1] - leads[0]),
            }
        )
    table = pd.DataFrame(rows).sort_values("range", ascending=False, kind="stable")
    table.attrs["base_lead"] = base_lead
    table.attrs["winner"] = winner
    return table.reset_index(drop=True)


def flip_threshold(scenario: Scenario, path: str) -> dict:
    """Smallest relative change in one input, up or down, that changes the top route.

    Searches outwards in steps of ``flip_search_step`` up to ``flip_search_max``.
    Returns the change (e.g. -0.23 for 23% lower) and the new winner for each
    direction, or ``None`` if the ranking holds across the whole search range.
    """
    settings = scenario.sensitivity
    winner = score_routes(scenario).winner
    value = scenario.get(path)
    out: dict = {"input": path, "label": scenario.input_label(path), "base_value": value}
    n_steps = round(settings.flip_search_max / settings.flip_search_step)
    for direction, sign in (("down", -1), ("up", 1)):
        out[f"change_{direction}"] = None
        out[f"winner_{direction}"] = None
        for k in range(1, n_steps + 1):
            change = sign * k * settings.flip_search_step
            new_winner = score_routes(scenario.with_value(path, value * (1 + change))).winner
            if new_winner != winner:
                out[f"change_{direction}"] = change
                out[f"winner_{direction}"] = new_winner
                break
    return out


def flip_thresholds(scenario: Scenario, paths: list[str]) -> pd.DataFrame:
    """Flip thresholds for several inputs."""
    return pd.DataFrame([flip_threshold(scenario, p) for p in paths])


def downside_table(scenario: Scenario) -> pd.DataFrame:
    """Base case against downside case, per route, with the ranking under each."""
    fin = run_financials(scenario)
    base_rank = score_routes(scenario, fin).ranking
    stress_rank = score_routes(scenario, fin, stressed=True).ranking
    rows = []
    for key in scenario.route_keys:
        b, d = fin.base[key], fin.downside[key]
        rows.append(
            {
                "route": key,
                "revenue_base": b.total_revenue,
                "revenue_downside": d.total_revenue,
                "retention": revenue_retention(b, d),
                "breakeven_base": b.projected_breakeven_month,
                "breakeven_downside": d.projected_breakeven_month,
                "peak_funding_base": b.peak_funding,
                "peak_funding_downside": d.peak_funding,
                "rank_base": base_rank.index(key) + 1,
                "rank_downside": stress_rank.index(key) + 1,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_base").reset_index(drop=True)
