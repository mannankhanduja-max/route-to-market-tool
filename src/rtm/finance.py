"""Monthly financial model for each route: revenue ramp, cash and breakeven.

Revenue here means end-customer revenue (what customers pay). The route's
cost structure then decides how much of it the company keeps:

    contribution   = revenue x (gross margin - channel take)
    acquisition    = new customers x CAC
    cash flow      = contribution - acquisition - monthly fixed cost - set-up cost
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtm.config import Downside, Market, Route, Scenario

MONTHS_PER_YEAR = 12


def ramp_curve(months: np.ndarray, first_revenue_month: float, ramp_months: float) -> np.ndarray:
    """Share of steady-state revenue reached in each month (0 to 1).

    Zero before the first revenue month, then a smooth S-curve that reaches
    1 after ``ramp_months``. The S-shape (slow start, fast middle, plateau)
    is a judgment call; see docs/methodology.md.
    """
    x = np.clip((months - first_revenue_month + 1) / ramp_months, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def partner_loss_curve(months: np.ndarray, route: Route, downside: Downside) -> np.ndarray:
    """Share of revenue kept after the main intermediary walks away.

    In the loss month the route loses its ``partner_dependency`` share of
    revenue, then wins it back in a straight line over the recovery period.
    A route with no dependency is unaffected.
    """
    progress = np.clip(
        (months - downside.partner_loss_month) / downside.partner_loss_recovery_months, 0.0, 1.0
    )
    lost = np.where(months >= downside.partner_loss_month, 1.0 - progress, 0.0)
    return 1.0 - route.partner_dependency * lost


@dataclass(frozen=True)
class RouteFinancials:
    """Monthly results for one route. All arrays run from month 1 to the horizon."""

    route: str
    months: np.ndarray
    revenue: np.ndarray
    customers: np.ndarray
    contribution: np.ndarray
    acquisition_cost: np.ndarray
    fixed_cost: np.ndarray
    cash_flow: np.ndarray
    cumulative_cash: np.ndarray

    @property
    def total_revenue(self) -> float:
        return float(self.revenue.sum())

    @property
    def breakeven_month(self) -> int | None:
        """First month in which cumulative cash is back to zero or above, if any."""
        hits = np.flatnonzero(self.cumulative_cash >= 0)
        return int(self.months[hits[0]]) if hits.size else None

    @property
    def projected_breakeven_month(self) -> float:
        """Breakeven month, extrapolated past the horizon at the final month's cash flow.

        Returns ``inf`` if the route is still losing money in the final month.
        The straight-line extrapolation is a judgment call.
        """
        month = self.breakeven_month
        if month is not None:
            return float(month)
        final_flow = float(self.cash_flow[-1])
        if final_flow <= 0:
            return float("inf")
        return float(self.months[-1] - self.cumulative_cash[-1] / final_flow)

    @property
    def peak_funding(self) -> float:
        """Largest cumulative cash outflow before the route pays back."""
        return float(max(0.0, -self.cumulative_cash.min()))

    @property
    def run_rate_margin(self) -> float:
        """Cash flow as a share of revenue in the final month."""
        if self.revenue[-1] <= 0:
            return float("-inf")
        return float(self.cash_flow[-1] / self.revenue[-1])

    @property
    def cumulative_margin(self) -> float:
        """Cumulative cash over cumulative revenue across the whole horizon."""
        if self.total_revenue <= 0:
            return float("-inf")
        return float(self.cumulative_cash[-1] / self.total_revenue)


def simulate_route(
    route: Route,
    market: Market,
    horizon_months: int,
    downside: Downside | None = None,
) -> RouteFinancials:
    """Project one route month by month.

    With ``downside`` set, the market is shrunk by the downside shock and the
    partner-loss event is applied; otherwise this is the base case.
    """
    months = np.arange(1, horizon_months + 1, dtype=float)
    steady_revenue = market.annual_market_size * market.target_share * route.reach
    revenue = (
        steady_revenue
        / MONTHS_PER_YEAR
        * ramp_curve(months, route.months_to_first_revenue, route.ramp_months)
    )
    if downside is not None:
        revenue = (
            revenue * (1.0 + downside.market_shock) * partner_loss_curve(months, route, downside)
        )

    customers = revenue * MONTHS_PER_YEAR / market.annual_price_per_customer
    gross_adds = np.maximum(np.diff(customers, prepend=0.0), 0.0)
    replacements = customers * market.annual_churn / MONTHS_PER_YEAR
    acquisition = (gross_adds + replacements) * route.cac

    contribution = revenue * (market.gross_margin - route.channel_take)
    fixed = np.full_like(months, route.monthly_fixed_cost)
    fixed[0] += route.fixed_setup_cost
    cash_flow = contribution - acquisition - fixed

    return RouteFinancials(
        route=route.key,
        months=months,
        revenue=revenue,
        customers=customers,
        contribution=contribution,
        acquisition_cost=acquisition,
        fixed_cost=fixed,
        cash_flow=cash_flow,
        cumulative_cash=np.cumsum(cash_flow),
    )


@dataclass(frozen=True)
class Financials:
    """Base-case and downside results for every route in a scenario."""

    base: dict[str, RouteFinancials]
    downside: dict[str, RouteFinancials]


def run_financials(scenario: Scenario) -> Financials:
    """Simulate every route in the base case and the downside case."""
    base, down = {}, {}
    for r in scenario.routes:
        base[r.key] = simulate_route(r, scenario.market, scenario.horizon_months)
        down[r.key] = simulate_route(r, scenario.market, scenario.horizon_months, scenario.downside)
    return Financials(base=base, downside=down)
