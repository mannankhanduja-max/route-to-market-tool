"""Tests for the revenue ramp, cash curve and breakeven logic."""

from dataclasses import replace

import numpy as np
import pytest

from rtm.finance import (
    RouteFinancials,
    partner_loss_curve,
    ramp_curve,
    run_financials,
    simulate_route,
)

MONTHS = np.arange(1, 37, dtype=float)


def _financials(cash_flow: list[float]) -> RouteFinancials:
    """Minimal RouteFinancials with a given cash-flow profile and flat revenue."""
    flow = np.array(cash_flow, dtype=float)
    months = np.arange(1, len(flow) + 1, dtype=float)
    ones = np.ones_like(flow)
    return RouteFinancials("test", months, ones, ones, ones, ones, ones, flow, np.cumsum(flow))


def test_ramp_is_zero_before_first_revenue_and_one_after_ramp():
    curve = ramp_curve(MONTHS, first_revenue_month=6, ramp_months=12)
    assert np.all(curve[:5] == 0)  # months 1-5
    assert curve[5] > 0  # month 6 bills
    assert curve[16] == pytest.approx(1.0)  # month 17 = 6 + 12 - 1
    assert np.all(np.diff(curve) >= 0)


def test_ramp_is_s_shaped():
    steps = np.diff(ramp_curve(MONTHS, first_revenue_month=1, ramp_months=20))[:19]
    peak = int(np.argmax(steps))
    assert 0 < peak < len(steps) - 1  # fastest growth in the middle, not at either end


def test_breakeven_is_first_month_cumulative_cash_is_non_negative():
    fin = _financials([-100, 20, 30, 50, 10])  # cumulative: -100, -80, -50, 0, 10
    assert fin.breakeven_month == 4


def test_breakeven_is_none_when_never_reached():
    fin = _financials([-100, 10, 10])
    assert fin.breakeven_month is None


def test_projected_breakeven_extrapolates_final_cash_flow():
    fin = _financials([-100, 10, 10])  # 80 still to recover at 10 a month
    assert fin.projected_breakeven_month == pytest.approx(3 + 8)


def test_projected_breakeven_is_infinite_when_still_losing_money():
    fin = _financials([-100, -5, -5])
    assert fin.projected_breakeven_month == float("inf")


def test_peak_funding_is_deepest_cumulative_hole():
    fin = _financials([-100, -50, 40, 200])
    assert fin.peak_funding == 150


def test_cash_flow_adds_up(scenario):
    route = scenario.route("reseller")
    fin = simulate_route(route, scenario.market, scenario.horizon_months)
    rebuilt = fin.contribution - fin.acquisition_cost - fin.fixed_cost
    np.testing.assert_allclose(fin.cash_flow, rebuilt)
    assert fin.fixed_cost[0] == route.monthly_fixed_cost + route.fixed_setup_cost


def test_steady_state_revenue_matches_market_inputs(scenario):
    route = replace(scenario.route("direct"), months_to_first_revenue=1, ramp_months=6)
    fin = simulate_route(route, scenario.market, scenario.horizon_months)
    m = scenario.market
    expected = m.annual_market_size * m.target_share * route.reach / 12
    assert fin.revenue[-1] == pytest.approx(expected)


def test_no_channel_take_means_contribution_equals_gross_profit(scenario):
    route = scenario.route("direct")
    assert route.channel_take == 0
    fin = simulate_route(route, scenario.market, scenario.horizon_months)
    np.testing.assert_allclose(fin.contribution, fin.revenue * scenario.market.gross_margin)


def test_partner_loss_hits_only_dependent_revenue(scenario):
    d = scenario.downside
    direct = partner_loss_curve(MONTHS, scenario.route("direct"), d)
    partner = partner_loss_curve(MONTHS, scenario.route("partner"), d)
    loss_idx = int(d.partner_loss_month) - 1
    assert np.all(direct == 1.0)
    assert partner[loss_idx - 1] == 1.0
    assert partner[loss_idx] == pytest.approx(1.0 - scenario.route("partner").partner_dependency)
    recovered = loss_idx + int(d.partner_loss_recovery_months)
    assert partner[recovered] == pytest.approx(1.0)


def test_downside_revenue_is_lower_and_shock_applies_to_all(scenario):
    fin = run_financials(scenario)
    shock = 1 + scenario.downside.market_shock
    direct = fin.downside["direct"].total_revenue / fin.base["direct"].total_revenue
    assert direct == pytest.approx(shock)  # no partner, so only the market shock
    for key in scenario.route_keys:
        assert fin.downside[key].total_revenue < fin.base[key].total_revenue


def test_higher_cac_delays_breakeven(scenario):
    route = scenario.route("partner")
    cheap = simulate_route(route, scenario.market, scenario.horizon_months)
    dear = simulate_route(
        replace(route, cac=route.cac * 5), scenario.market, scenario.horizon_months
    )
    assert dear.projected_breakeven_month > cheap.projected_breakeven_month
