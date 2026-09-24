"""Smoke test: the dashboard runs end to end and reacts to a slider."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT_SECONDS = 120


def test_app_runs_without_errors():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_SECONDS).run()
    assert not at.exception
    assert "reseller network" in at.markdown[0].value.lower()


def test_moving_a_weight_changes_the_recommendation():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_SECONDS).run()
    at.sidebar.slider[0].set_value(100)  # velocity
    at.sidebar.slider[1].set_value(0)  # margin
    at.sidebar.slider[2].set_value(0)  # robustness
    at.run()
    assert not at.exception
    assert "strategic partner" in at.markdown[0].value.lower()
