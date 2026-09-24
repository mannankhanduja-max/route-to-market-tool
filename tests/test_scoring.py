"""Tests for the VMR scoring formulas and ranking."""

import math

import pandas as pd
import pytest

from rtm.config import DIMENSIONS, Anchor
from rtm.scoring import combine, normalise_weights, scale, score_routes, summary_table

HIGHER_IS_BETTER = Anchor(best=0.5, worst=0.0, weight=1)
LOWER_IS_BETTER = Anchor(best=3, worst=18, weight=1)


@pytest.mark.parametrize(
    ("value", "anchor", "expected"),
    [
        (0.5, HIGHER_IS_BETTER, 100),
        (0.0, HIGHER_IS_BETTER, 0),
        (0.25, HIGHER_IS_BETTER, 50),
        (0.9, HIGHER_IS_BETTER, 100),  # capped above best
        (-1.0, HIGHER_IS_BETTER, 0),  # capped below worst
        (3, LOWER_IS_BETTER, 100),
        (18, LOWER_IS_BETTER, 0),
        (6, LOWER_IS_BETTER, 80),
        (math.inf, LOWER_IS_BETTER, 0),  # never breaks even
        (-math.inf, HIGHER_IS_BETTER, 0),
    ],
)
def test_scale_maps_anchors_to_0_and_100(value, anchor, expected):
    assert scale(value, anchor) == pytest.approx(expected)


def test_normalise_weights_sums_to_one():
    w = normalise_weights({"a": 2, "b": 1, "c": 1})
    assert w == {"a": 0.5, "b": 0.25, "c": 0.25}


def test_normalise_weights_rejects_zero_total():
    with pytest.raises(ValueError):
        normalise_weights({"a": 0, "b": 0})


def test_combine_is_weighted_sum_and_sorted():
    dims = pd.DataFrame(
        {"velocity": [100, 0], "margin": [0, 100], "robustness": [50, 50]}, index=["x", "y"]
    )
    totals = combine(dims, {"velocity": 0.6, "margin": 0.2, "robustness": 0.2})
    assert list(totals.index) == ["x", "y"]
    assert totals["x"] == pytest.approx(70)
    assert totals["y"] == pytest.approx(30)


def test_dimension_score_equals_sum_of_workings(scenario):
    result = score_routes(scenario)
    w = result.workings
    for route in scenario.route_keys:
        for dim in DIMENSIONS:
            rows = w[(w.route == route) & (w.dimension == dim)]
            assert rows.metric_weight.sum() == pytest.approx(1.0)
            assert result.dimension_scores.loc[route, dim] == pytest.approx(rows.contribution.sum())


def test_scores_are_between_0_and_100(scenario):
    result = score_routes(scenario)
    assert result.workings.score.between(0, 100).all()
    assert result.totals.between(0, 100).all()


def test_total_uses_dimension_weights(scenario):
    result = score_routes(scenario)
    route = result.winner
    expected = sum(result.dimension_scores.loc[route, d] * result.weights[d] for d in DIMENSIONS)
    assert result.totals[route] == pytest.approx(expected)


def test_ranking_is_complete_and_ordered(scenario):
    result = score_routes(scenario)
    assert sorted(result.ranking) == sorted(scenario.route_keys)
    assert result.totals.is_monotonic_decreasing
    assert result.lead >= 0


def test_all_weight_on_one_dimension_picks_that_dimension_leader(scenario):
    base = score_routes(scenario)
    for dim in DIMENSIONS:
        shifted = scenario
        for d in DIMENSIONS:
            shifted = shifted.with_value(f"weights.{d}", float(d == dim))
        result = score_routes(shifted)
        assert result.winner == base.dimension_scores[dim].idxmax()


def test_removing_partner_dependency_improves_robustness(scenario):
    safer = scenario.with_value("routes.partner.partner_dependency", 0.0)
    before = score_routes(scenario).dimension_scores.loc["partner", "robustness"]
    after = score_routes(safer).dimension_scores.loc["partner", "robustness"]
    assert after > before


def test_stressed_scores_are_not_higher(scenario):
    base = score_routes(scenario).totals
    stressed = score_routes(scenario, stressed=True).totals
    for route in scenario.route_keys:
        assert stressed[route] <= base[route] + 1e-9


def test_summary_table_has_one_row_per_route(scenario):
    table = summary_table(scenario, score_routes(scenario))
    assert list(table["rank"]) == [1, 2, 3]
    assert set(table.route) == {r.label for r in scenario.routes}
