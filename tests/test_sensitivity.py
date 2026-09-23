"""Tests for the ranking-change logic: weight flips, tornado, flip thresholds, downside."""

import pandas as pd
import pytest

from rtm.config import DIMENSIONS
from rtm.scoring import ScoreResult, combine, score_routes
from rtm.sensitivity import (
    _line_weights,
    downside_table,
    flip_threshold,
    tornado,
    weight_flips,
    weight_map,
    weight_space_share,
    weight_sweep,
)


def _result(dims: dict[str, list[float]], weights: dict[str, float]) -> ScoreResult:
    """A hand-built ScoreResult with known dimension scores."""
    table = pd.DataFrame(dims, index=list(DIMENSIONS)).T
    return ScoreResult(pd.DataFrame(), table, weights, combine(table, weights))


def test_line_weights_sum_to_one_and_keep_the_ratio():
    w = _line_weights({"velocity": 0.4, "margin": 0.35, "robustness": 0.25}, "velocity", 0.7)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["velocity"] == 0.7
    assert w["margin"] / w["robustness"] == pytest.approx(0.35 / 0.25)


def test_weight_flip_is_found_where_the_lines_cross(scenario):
    # x: 60w + 44.17(1-w); y: 40w + 55.83(1-w); they cross at w = 11.67 / 31.67 = 0.368
    weights = {"velocity": 0.4, "margin": 0.35, "robustness": 0.25}
    result = _result({"x": [60, 40, 50], "y": [40, 60, 50]}, weights)
    assert result.winner == "x"
    flips = weight_flips(scenario, result)
    down = flips[(flips.dimension == "velocity") & (flips.direction == "down")].iloc[0]
    assert down.new_winner == "y"
    assert down.flip_weight == pytest.approx(0.3684, abs=2 * scenario.sensitivity.weight_resolution)


def test_no_flip_when_one_route_dominates(scenario):
    weights = {"velocity": 0.4, "margin": 0.35, "robustness": 0.25}
    result = _result({"x": [90, 90, 90], "y": [10, 10, 10]}, weights)
    assert weight_flips(scenario, result).empty


def test_default_scenario_flips_are_real(scenario):
    result = score_routes(scenario)
    step = scenario.sensitivity.weight_resolution
    for flip in weight_flips(scenario, result).itertuples():
        at = combine(
            result.dimension_scores, _line_weights(result.weights, flip.dimension, flip.flip_weight)
        )
        sign = 1 if flip.direction == "up" else -1
        before_w = flip.flip_weight - sign * step
        before = combine(
            result.dimension_scores, _line_weights(result.weights, flip.dimension, before_w)
        )
        assert at.index[0] == flip.new_winner != result.winner
        assert before.index[0] == result.winner


def test_weight_sweep_matches_combine(scenario):
    result = score_routes(scenario)
    sweep = weight_sweep(scenario, result)
    row = sweep[(sweep.dimension == "margin") & (sweep.weight == 0.5) & (sweep.route == "direct")]
    expected = combine(result.dimension_scores, _line_weights(result.weights, "margin", 0.5))
    assert row.total.iloc[0] == pytest.approx(expected["direct"])


def test_weight_map_covers_the_simplex(scenario):
    result = score_routes(scenario)
    grid = weight_map(scenario, result)
    n = round(1 / scenario.sensitivity.weight_map_step)
    assert len(grid) == (n + 1) * (n + 2) // 2
    assert grid[list(DIMENSIONS)].sum(axis=1).round(9).eq(1).all()
    assert (grid.lead >= 0).all()
    assert weight_space_share(scenario, result).sum() == pytest.approx(1.0)


def test_tornado_is_sorted_and_consistent(scenario):
    table = tornado(scenario)
    assert table["range"].is_monotonic_decreasing
    assert (table.base_value != 0).all()  # zero inputs are skipped
    top = table.iloc[0]
    swing = scenario.sensitivity.tornado_swing
    totals = score_routes(scenario.with_value(top.input, top.base_value * (1 + swing))).totals
    winner = table.attrs["winner"]
    assert top.lead_high == pytest.approx(totals[winner] - totals.drop(winner).max())


def test_tornado_marks_flips_with_negative_lead(scenario):
    table = tornado(scenario)
    winner = table.attrs["winner"]
    for side in ("low", "high"):
        flipped = table[f"winner_{side}"] != winner
        assert (table.loc[flipped, f"lead_{side}"] < 0).all()
        assert (table.loc[~flipped, f"lead_{side}"] >= 0).all()


def test_flip_threshold_is_the_smallest_change_that_flips(scenario):
    path = tornado(scenario).input.iloc[0]
    result = flip_threshold(scenario, path)
    winner = score_routes(scenario).winner
    step = scenario.sensitivity.flip_search_step
    value = scenario.get(path)
    found = [
        (d, result[f"change_{d}"]) for d in ("down", "up") if result[f"change_{d}"] is not None
    ]
    assert found, "the most sensitive input should flip the ranking within the search range"
    for direction, change in found:
        sign = -1 if direction == "down" else 1
        at = score_routes(scenario.with_value(path, value * (1 + change))).winner
        before = score_routes(scenario.with_value(path, value * (1 + change - sign * step))).winner
        assert at == result[f"winner_{direction}"] != winner
        assert before == winner


def test_flip_threshold_is_none_for_an_input_that_cannot_flip(scenario):
    result = flip_threshold(scenario, "routes.direct.partner_dependency")  # zero stays zero
    assert result["change_down"] is None
    assert result["change_up"] is None


def test_downside_table(scenario):
    table = downside_table(scenario)
    assert (table.retention < 1).all()
    assert sorted(table.rank_base) == sorted(table.rank_downside) == [1, 2, 3]
    assert table.rank_base.tolist() == [1, 2, 3]
