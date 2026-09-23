"""Tests for the recommendation generator."""

import pytest

from rtm.recommend import Recommendation, build_recommendation, format_input
from rtm.scoring import score_routes


@pytest.fixture(scope="module")
def rec(scenario) -> Recommendation:
    return build_recommendation(scenario)


def test_recommends_the_top_scoring_route(scenario, rec):
    assert rec.route == score_routes(scenario).winner
    assert scenario.label(rec.route).lower() in rec.headline


def test_has_exactly_three_risks(rec):
    assert len(rec.risks) == 3
    assert all(r.strip() for r in rec.risks)


def test_every_section_is_filled(rec):
    for section in (rec.why, rec.stops_winning, rec.next_steps):
        assert section
    md = rec.to_markdown()
    for heading in ("Why it wins", "When it stops winning", "Top three risks", "Next steps"):
        assert heading in md


def test_stops_winning_covers_weights_inputs_and_downside(rec):
    text = " ".join(rec.stops_winning)
    assert "weight" in text
    assert "than assumed" in text
    assert "Downside case" in text


def test_recommendation_follows_the_inputs(scenario):
    # Make the partner route free and fast: it should become the recommendation.
    cheap = scenario
    for path, value in {
        "routes.partner.channel_take": 0.0,
        "routes.partner.partner_dependency": 0.0,
        "routes.partner.concentration_risk": 0.0,
        "routes.partner.lock_in_months": 0.0,
    }.items():
        cheap = cheap.with_value(path, value)
    assert build_recommendation(cheap).route == "partner"


def test_format_input_uses_natural_units(scenario):
    assert format_input(scenario, "routes.reseller.channel_take", 0.3) == "30%"
    assert format_input(scenario, "routes.direct.cac", 35000) == "€35k"
    assert format_input(scenario, "routes.direct.ramp_months", 24) == "24 months"
    assert format_input(scenario, "downside.partner_loss_month", 18) == "month 18"
