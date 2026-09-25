"""Tests for the logistic-regression and random-forest check."""

from dataclasses import replace

import numpy as np
import pytest

from rtm.ml import FOREST, LOGISTIC, model_table, run_ml_check, sample_scenarios, varied_inputs
from rtm.scoring import score_routes, total_scores


@pytest.fixture(scope="module")
def small(scenario):
    """The example scenario with a smaller, faster sample."""
    ml = replace(scenario.ml, n_samples=400, rf_trees=40, importance_repeats=2)
    return replace(scenario, ml=ml)


@pytest.fixture(scope="module")
def result(small):
    return run_ml_check(small, score_routes(small).winner)


def test_total_scores_matches_score_routes(scenario):
    fast = total_scores(scenario)
    full = score_routes(scenario).totals
    for key in scenario.route_keys:
        assert fast[key] == pytest.approx(full[key], abs=1e-9)


def test_sample_stays_within_range_and_is_reproducible(small):
    first, second = sample_scenarios(small), sample_scenarios(small)
    assert first.equals(second)  # fixed seed
    spread = small.ml.input_range
    for path in varied_inputs(small):
        base = small.get(path)
        lo, hi = sorted((base * (1 - spread), base * (1 + spread)))
        column = first[path]
        if path.split(".")[-1] in ("reach", "partner_dependency", "gross_margin"):
            hi = min(hi, 1.0)  # shares are clipped
        assert column.between(lo - 1e-9, hi + 1e-9).all(), path


def test_sample_labels_are_the_vmr_winner(small):
    sample = sample_scenarios(small).head(5)
    for _, row in sample.iterrows():
        s = small
        for path in varied_inputs(small):
            s = s.with_value(path, row[path])
        assert score_routes(s).winner == row["winner"]


def test_win_rates_sum_to_one(result, small):
    assert result.win_rates.sum() == pytest.approx(1.0)
    assert list(result.win_rates.index) == list(small.route_keys)


def test_both_models_beat_always_guessing_the_most_common_winner(result):
    assert result.trained
    for model in (LOGISTIC, FOREST):
        assert result.accuracy[model] > result.baseline_accuracy


def test_base_case_probabilities_favour_the_vmr_winner(result, small):
    winner = score_routes(small).winner
    for model in (LOGISTIC, FOREST):
        probs = result.base_probability[model]
        assert probs.sum() == pytest.approx(1.0)
        assert probs.idxmax() == winner


def test_importance_finds_the_tornado_leaders(result):
    # The two channel takes top the tornado; joint sampling should agree.
    top = set(result.importance.input.head(4))
    assert {"routes.reseller.channel_take", "routes.partner.channel_take"} <= top


def test_single_winner_is_handled(scenario):
    # Make direct sales dominate so every draw has the same winner.
    s = scenario
    for key in ("reseller", "partner"):
        s = s.with_value(f"routes.{key}.reach", 0.01)
    s = replace(s, ml=replace(s.ml, n_samples=50))
    ml = run_ml_check(s, "direct")
    assert not ml.trained
    assert ml.win_rates["direct"] > 0.95


def test_model_table_rows(result, small):
    table = model_table(small, result)
    assert list(table.model)[1:] == [LOGISTIC, FOREST]
    assert np.isfinite(table["test accuracy"]).all()
