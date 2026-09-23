"""Tests for loading, validating and modifying the scenario."""

import pytest

from rtm.config import METRICS, ScenarioError, parse_scenario


def _walk_leaves(node, path=""):
    """Yield (path, leaf) for every numeric input in the YAML."""
    if isinstance(node, dict):
        if "source" in node:
            yield path, node
            return
        for key, child in node.items():
            yield from _walk_leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, {"value": node}


def test_default_scenario_loads(scenario):
    assert scenario.route_keys == ("direct", "reseller", "partner")
    assert scenario.horizon_months == 36
    assert scenario.weights == {"velocity": 0.40, "margin": 0.35, "robustness": 0.25}


def test_every_number_in_the_yaml_has_a_source(raw_scenario):
    for path, leaf in _walk_leaves(raw_scenario):
        assert str(leaf.get("source", "")).strip(), f"{path} has no source"


def test_sources_are_collected_for_every_input(scenario):
    for path in scenario.input_paths():
        assert scenario.sources.get(path), path


def test_missing_source_is_rejected(raw):
    del raw["market"]["gross_margin"]["source"]
    with pytest.raises(ScenarioError, match="source"):
        parse_scenario(raw)


def test_missing_input_is_rejected(raw):
    del raw["routes"]["reseller"]["cac"]
    with pytest.raises(ScenarioError, match=r"routes\.reseller\.cac"):
        parse_scenario(raw)


def test_unknown_metric_is_rejected(raw):
    raw["scoring"]["margin"]["made_up"] = {"best": 1, "worst": 0, "weight": 1, "source": "x"}
    with pytest.raises(ScenarioError, match="unknown metrics"):
        parse_scenario(raw)


def test_first_revenue_before_onboarding_is_rejected(raw):
    raw["routes"]["partner"]["months_to_first_revenue"]["value"] = 2
    with pytest.raises(ScenarioError, match="onboarding"):
        parse_scenario(raw)


def test_share_outside_unit_interval_is_rejected(raw):
    raw["routes"]["direct"]["reach"]["value"] = 1.5
    with pytest.raises(ScenarioError, match="between 0 and 1"):
        parse_scenario(raw)


def test_every_known_metric_is_scored(scenario):
    for dim, metrics in METRICS.items():
        assert set(scenario.anchors[dim]) == set(metrics)


def test_with_value_changes_only_the_target(scenario):
    changed = scenario.with_value("routes.reseller.cac", 9999)
    assert changed.get("routes.reseller.cac") == 9999
    assert changed.get("routes.direct.cac") == scenario.get("routes.direct.cac")
    assert scenario.get("routes.reseller.cac") != 9999  # original untouched


def test_with_value_clips_shares(scenario):
    assert scenario.with_value("market.gross_margin", 1.4).market.gross_margin == 1.0
    assert scenario.with_value("routes.partner.reach", -0.2).route("partner").reach == 0.0


def test_unknown_path_raises(scenario):
    with pytest.raises(KeyError):
        scenario.with_value("routes.nowhere.cac", 1)


def test_input_labels_are_readable(scenario):
    assert scenario.input_label("routes.reseller.channel_take") == "Reseller network: channel take"
    assert scenario.input_label("market.annual_market_size") == "Market size"
