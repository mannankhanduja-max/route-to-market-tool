"""Shared fixtures."""

import copy

import pytest
import yaml

from rtm.config import DEFAULT_SCENARIO, Scenario, load_scenario


@pytest.fixture(scope="session")
def raw_scenario() -> dict:
    """The parsed YAML, for tests that need to break it on purpose."""
    with open(DEFAULT_SCENARIO, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def raw(raw_scenario: dict) -> dict:
    """A fresh deep copy of the YAML that a test may modify."""
    return copy.deepcopy(raw_scenario)


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return load_scenario()
