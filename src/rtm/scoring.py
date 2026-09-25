"""VMR scoring: turn each route's inputs and financials into 0-100 scores.

1. Each metric is scaled linearly between fixed anchors from the scenario
   file: the ``worst`` anchor scores 0, the ``best`` anchor 100, and values
   beyond either end are capped.
2. Metric scores are averaged inside each dimension (Velocity, Margin,
   Robustness) using the metric weights.
3. Dimension scores are combined with the dimension weights into a total.

Every intermediate number is kept in a workings table so the result can be
audited line by line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rtm.config import DIMENSIONS, Anchor, Route, Scenario
from rtm.finance import Financials, RouteFinancials, revenue_retention, run_financials

SCORE_MAX = 100.0


def scale(value: float, anchor: Anchor) -> float:
    """Map a raw metric value to 0-100 between the worst and best anchors."""
    share = (value - anchor.worst) / (anchor.best - anchor.worst)
    return SCORE_MAX * float(np.clip(share, 0.0, 1.0)) + 0.0  # + 0.0 avoids printing -0.0


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Rescale weights so they sum to one."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


def metric_values(route: Route, base: RouteFinancials, downside: RouteFinancials) -> dict:
    """Raw value of every scoring metric for one route."""
    return {
        "months_to_first_revenue": route.months_to_first_revenue,
        "breakeven_month": base.projected_breakeven_month,
        "onboarding_months": route.onboarding_months,
        "run_rate_margin": base.run_rate_margin,
        "cumulative_margin": base.cumulative_margin,
        "downside_revenue_retention": revenue_retention(base, downside),
        "concentration_risk": route.concentration_risk,
        "partner_dependency": route.partner_dependency,
        "lock_in_months": route.lock_in_months,
    }


@dataclass(frozen=True)
class ScoreResult:
    """Scores, ranking and workings for every route."""

    workings: pd.DataFrame
    dimension_scores: pd.DataFrame
    weights: dict[str, float]
    totals: pd.Series

    @property
    def ranking(self) -> list[str]:
        """Route keys from best to worst total score."""
        return list(self.totals.index)

    @property
    def winner(self) -> str:
        return self.ranking[0]

    @property
    def runner_up(self) -> str:
        return self.ranking[1]

    @property
    def lead(self) -> float:
        """Points by which the winner beats the runner-up."""
        return float(self.totals.iloc[0] - self.totals.iloc[1])


def combine(dimension_scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted total per route, sorted from best to worst.

    Ties keep the order of routes in the scenario file, so results are stable.
    """
    w = normalise_weights(weights)
    totals = sum(dimension_scores[d] * w[d] for d in DIMENSIONS)
    return totals.sort_values(ascending=False, kind="stable").rename("total")


def _workings_rows(scenario: Scenario, fin: Financials, stressed: bool) -> list[dict]:
    """One row per (route, metric): raw value, 0-100 score and weighted contribution."""
    rows = []
    for r in scenario.routes:
        view = fin.downside[r.key] if stressed else fin.base[r.key]
        values = metric_values(r, fin.base[r.key], fin.downside[r.key])
        if stressed:
            values |= {
                "breakeven_month": view.projected_breakeven_month,
                "run_rate_margin": view.run_rate_margin,
                "cumulative_margin": view.cumulative_margin,
            }
        for dim in DIMENSIONS:
            metric_weights = normalise_weights(
                {m: a.weight for m, a in scenario.anchors[dim].items()}
            )
            for metric, anchor in scenario.anchors[dim].items():
                score = scale(values[metric], anchor)
                rows.append(
                    {
                        "route": r.key,
                        "dimension": dim,
                        "metric": metric,
                        "value": values[metric],
                        "worst": anchor.worst,
                        "best": anchor.best,
                        "score": score,
                        "metric_weight": metric_weights[metric],
                        "contribution": score * metric_weights[metric],
                    }
                )
    return rows


def score_routes(
    scenario: Scenario, financials: Financials | None = None, stressed: bool = False
) -> ScoreResult:
    """Score every route.

    ``stressed=True`` scores Velocity and Margin on the downside financials
    instead of the base case; Robustness is the same in both, since it
    already compares the two cases.
    """
    fin = financials or run_financials(scenario)
    workings = pd.DataFrame(_workings_rows(scenario, fin, stressed))
    dims = workings.pivot_table(
        index="route", columns="dimension", values="contribution", aggfunc="sum", sort=False
    )[list(DIMENSIONS)]
    dims = dims.reindex(scenario.route_keys)
    weights = normalise_weights(scenario.weights)
    return ScoreResult(
        workings=workings,
        dimension_scores=dims,
        weights=weights,
        totals=combine(dims, weights),
    )


def total_scores(scenario: Scenario) -> dict[str, float]:
    """Weighted total per route, in scenario order, without building DataFrames.

    Same arithmetic as :func:`score_routes`; used where thousands of
    scenarios are scored (the machine-learning sample).
    """
    weights = normalise_weights(scenario.weights)
    totals = dict.fromkeys(scenario.route_keys, 0.0)
    for row in _workings_rows(scenario, run_financials(scenario), stressed=False):
        totals[row["route"]] += row["contribution"] * weights[row["dimension"]]
    return totals


def summary_table(scenario: Scenario, result: ScoreResult) -> pd.DataFrame:
    """One row per route: dimension scores, weighted total and rank."""
    table = result.dimension_scores.loc[result.ranking].copy()
    table["total"] = result.totals
    table.insert(0, "route", [scenario.label(k) for k in table.index])
    table["rank"] = range(1, len(table) + 1)
    return table.reset_index(drop=True)
