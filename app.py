"""Streamlit dashboard: ``streamlit run app.py``.

Every slider starts at the value in data/scenario.yaml. Moving one re-runs
the full model (finance, scoring, sensitivity, recommendation) on a copy of
the scenario; the file itself is never changed. The machine-learning check
runs only when switched on, since it samples and trains on thousands of draws.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from rtm import charts, fmt
from rtm.config import DIMENSIONS, Scenario, load_scenario
from rtm.finance import run_financials
from rtm.ml import model_table, run_ml_check
from rtm.recommend import build_recommendation
from rtm.scoring import score_routes, summary_table
from rtm.sensitivity import downside_table, tornado, weight_map, weight_sweep

st.set_page_config(page_title="Route-to-Market Decision Tool", layout="wide")

# Route inputs exposed as sliders: (field, label, step, upper bound as a multiple of base).
ROUTE_SLIDERS = (
    ("channel_take", "Channel take", 0.01, None),
    ("cac", "CAC", 500.0, 3.0),
    ("months_to_first_revenue", "Months to first revenue", 1.0, 3.0),
    ("ramp_months", "Ramp months", 1.0, 3.0),
    ("monthly_fixed_cost", "Monthly fixed cost", 5000.0, 3.0),
    ("partner_dependency", "Partner dependency", 0.05, None),
)


@st.cache_data(show_spinner=False)
def analyse(overrides: tuple[tuple[str, float], ...]) -> dict:
    """Run the whole model for the base scenario plus slider overrides."""
    scenario = apply(load_scenario(), overrides)
    fin = run_financials(scenario)
    result = score_routes(scenario, fin)
    return {
        "scenario": scenario,
        "fin": fin,
        "result": result,
        "summary": summary_table(scenario, result),
        "recommendation": build_recommendation(scenario),
        "tornado": tornado(scenario),
        "sweep": weight_sweep(scenario, result),
        "map": weight_map(scenario, result),
        "downside": downside_table(scenario),
    }


@st.cache_data(show_spinner=False)
def machine_learning(overrides: tuple[tuple[str, float], ...], winner: str):
    """Logistic regression and random forest check; slow, so run only on request."""
    return run_ml_check(apply(load_scenario(), overrides), winner)


def apply(scenario: Scenario, overrides: tuple[tuple[str, float], ...]) -> Scenario:
    for path, value in overrides:
        scenario = scenario.with_value(path, value)
    return scenario


def sidebar(base: Scenario) -> tuple[tuple[str, float], ...]:
    """Draw the controls and return the chosen values as (path, value) pairs."""
    out: list[tuple[str, float]] = []
    st.sidebar.header("Weights")
    st.sidebar.caption("Rescaled to sum to 100%.")
    for dim in DIMENSIONS:
        value = st.sidebar.slider(dim.title(), 0, 100, round(base.weights[dim] * 100), 5)
        out.append((f"weights.{dim}", value / 100))
    if sum(v for _, v in out) == 0:
        st.sidebar.error("At least one weight must be above zero; using the defaults.")
        out = [(f"weights.{d}", base.weights[d]) for d in DIMENSIONS]

    st.sidebar.header("Market")
    m = base.market
    size = st.sidebar.slider(
        "Market size (m)", 0.0, m.annual_market_size / 1e6 * 3, m.annual_market_size / 1e6, 5.0
    )
    out.append(("market.annual_market_size", size * 1e6))
    for field, label in (("target_share", "Target share"), ("gross_margin", "Gross margin")):
        value = st.sidebar.slider(label, 0.0, 1.0, float(getattr(m, field)), 0.01)
        out.append((f"market.{field}", value))

    for route in base.routes:
        with st.sidebar.expander(route.label):
            st.caption(route.description)
            for field, label, step, cap in ROUTE_SLIDERS:
                value = float(getattr(route, field))
                upper = 1.0 if cap is None else max(value * cap, step * 10)
                chosen = st.slider(label, 0.0, upper, value, step, key=f"{route.key}.{field}")
                out.append((f"routes.{route.key}.{field}", chosen))
    return tuple(out)


def figure(draw, *args, size=(7, 3.6)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax, *args)
    fig.tight_layout()
    return fig


def machine_learning_tab(s: Scenario, result, overrides) -> None:
    st.markdown(
        f"Draws {s.ml.n_samples:,} scenarios with every input moved at once "
        f"(±{fmt.pct(s.ml.input_range)}), labels each with the route the VMR model picks, "
        "and trains a logistic regression and a random forest to predict that route. "
        "The models learn this tool's logic, not real market data."
    )
    if not st.toggle("Run the machine-learning check (about 10 seconds)", key="run_ml"):
        return
    with st.spinner("Sampling scenarios and training both models..."):
        ml = machine_learning(overrides, result.winner)
    rates = ", ".join(f"{s.label(k)} {fmt.pct(v)}" for k, v in ml.win_rates.items())
    st.markdown(f"**Share of scenarios each route wins:** {rates}.")
    if not ml.trained:
        st.info(
            "One route wins (almost) every scenario, so there is too little variety for the "
            "models to learn from."
        )
        return
    c1, c2 = st.columns([1, 1.5])
    c1.pyplot(figure(charts.plot_win_rates, s, ml, size=(6, 3.6)))
    c2.pyplot(figure(charts.plot_ml_importance, s, ml, size=(8, 4.6)))
    st.dataframe(model_table(s, ml).round(3), hide_index=True)
    st.caption(
        "Importance is the drop in held-out accuracy when an input is shuffled. "
        "Direction is the logistic regression's standardised coefficient for the current "
        "winner: positive means a higher value helps it stay on top."
    )
    st.dataframe(ml.importance.drop(columns="input").round(3), hide_index=True)


def main() -> None:
    base = load_scenario()
    overrides = sidebar(base)
    first_revenue_ok = all(
        dict(overrides)[f"routes.{r.key}.months_to_first_revenue"] >= r.onboarding_months
        for r in base.routes
    )
    if not first_revenue_ok:
        st.warning("Months to first revenue is below onboarding time for at least one route.")

    with st.spinner("Scoring routes and running sensitivity analysis..."):
        a = analyse(overrides)
    s, fin, result, rec = a["scenario"], a["fin"], a["result"], a["recommendation"]

    st.title("Route-to-Market Decision Tool")
    st.caption(f"{s.name}. All figures are synthetic. VMR = Velocity, Margin, Robustness.")
    st.markdown(f"### {rec.headline}")

    cols = st.columns([1.6, 1, 1.2, 1])
    winner = fin.base[result.winner]
    breakeven = winner.projected_breakeven_month
    cols[0].metric("Recommended route", s.label(result.winner))
    cols[1].metric("VMR score", f"{result.totals.iloc[0]:.1f}", f"+{result.lead:.1f} vs #2")
    cols[2].metric(
        "Breakeven month",
        "none" if breakeven == float("inf") else f"{breakeven:.0f}",
        help=f"Values above {s.horizon_months} are projected past the horizon at the final "
        "month's cash flow.",
    )
    cols[3].metric("Peak funding", fmt.money(winner.peak_funding, s.currency))

    left, right = st.columns(2)
    left.pyplot(figure(charts.plot_scores, s, result))
    right.pyplot(figure(charts.plot_cash, s, fin, True))

    tabs = st.tabs(
        [
            "Recommendation",
            "Sensitivity",
            "Machine learning",
            "Financials",
            "Workings",
            "Assumptions",
        ]
    )
    with tabs[0]:
        st.markdown(rec.to_markdown())
    with tabs[1]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        charts.plot_weight_sweep(list(axes), s, result, a["sweep"])
        fig.tight_layout()
        st.pyplot(fig)
        c1, c2 = st.columns([1.3, 1])
        c1.pyplot(figure(charts.plot_tornado, s, a["tornado"], size=(8, 4.8)))
        c2.pyplot(figure(charts.plot_weight_map, s, result, a["map"], size=(6, 5)))
        st.subheader("Downside case")
        st.dataframe(a["downside"].round(2), hide_index=True)
    with tabs[2]:
        machine_learning_tab(s, result, overrides)
    with tabs[3]:
        st.pyplot(figure(charts.plot_revenue, s, fin, size=(10, 3.4)))
        monthly = pd.concat(
            {
                s.label(k): pd.DataFrame(
                    {"revenue": f.revenue, "cumulative_cash": f.cumulative_cash},
                    index=f.months.astype(int),
                )
                for k, f in fin.base.items()
            },
            axis=1,
        )
        st.dataframe(monthly.round(0))
    with tabs[4]:
        st.dataframe(a["summary"].round(1), hide_index=True)
        st.dataframe(result.workings.round(3), hide_index=True)
    with tabs[5]:
        paths = s.input_paths() + [f"weights.{d}" for d in DIMENSIONS]
        sources = pd.DataFrame(
            [(s.input_label(p), s.get(p), s.sources[p]) for p in paths],
            columns=["input", "value", "source"],
        )
        st.dataframe(sources, hide_index=True)
        st.caption("Scoring anchors and their rationale are in data/scenario.yaml.")


main()
