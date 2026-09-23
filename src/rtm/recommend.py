"""Turn the scores and sensitivity results into a plain-English recommendation.

The text is generated from rules, not written by hand, so it always matches
the numbers. Which three risks are shown is itself a rule (see
docs/methodology.md): the most fragile assumption, the cash exposure, and the
recommended route's weakest robustness metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rtm import fmt
from rtm.config import DIMENSIONS, FRACTION_INPUTS, Scenario
from rtm.finance import Financials, revenue_retention, run_financials
from rtm.scoring import ScoreResult, score_routes
from rtm.sensitivity import (
    downside_table,
    flip_thresholds,
    tornado,
    weight_flips,
    weight_space_share,
)

MONEY_INPUTS = frozenset(
    {
        "annual_market_size",
        "annual_price_per_customer",
        "cac",
        "fixed_setup_cost",
        "monthly_fixed_cost",
    }
)


def format_input(scenario: Scenario, path: str, value: float) -> str:
    """Show an input value in its natural unit."""
    name = path.split(".")[-1]
    if name in FRACTION_INPUTS or name == "market_shock":
        return fmt.pct(value)
    if name in MONEY_INPUTS:
        return fmt.money(value, scenario.currency)
    if name.endswith("months") or name.endswith("month"):
        return f"{value:.0f} months" if name.endswith("months") else f"month {value:.0f}"
    return f"{value:.2f}"


@dataclass(frozen=True)
class Recommendation:
    """The recommendation, split into the sections a reader expects."""

    route: str
    headline: str
    why: list[str]
    stops_winning: list[str]
    risks: list[str]
    next_steps: list[str]

    def to_markdown(self) -> str:
        """Render as Markdown for the README, the app and the terminal."""

        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        return "\n\n".join(
            [
                f"**{self.headline}**",
                "**Why it wins**\n\n" + bullets(self.why),
                "**When it stops winning**\n\n" + bullets(self.stops_winning),
                "**Top three risks**\n\n"
                + "\n".join(f"{i}. {r}" for i, r in enumerate(self.risks, 1)),
                "**Next steps**\n\n"
                + "\n".join(f"{i}. {s}" for i, s in enumerate(self.next_steps, 1)),
            ]
        )


def _why(scenario: Scenario, result: ScoreResult, fin: Financials, share: float) -> list[str]:
    win, run = result.winner, result.runner_up
    diff = {
        d: (result.dimension_scores.loc[win, d] - result.dimension_scores.loc[run, d])
        * result.weights[d]
        for d in DIMENSIONS
    }
    ahead = " and ".join(f"{diff[d]:.1f} on {d.title()}" for d in DIMENSIONS if diff[d] > 0)
    behind = " and ".join(f"{-diff[d]:.1f} on {d.title()}" for d in DIMENSIONS if diff[d] < 0)
    edge = f"Against the runner-up ({scenario.label(run)}) it gains {ahead}"
    edge += f", and gives up {behind} (weighted points)." if behind else " (weighted points)."
    base = fin.base[win]
    numbers = (
        f"It breaks even at {fmt.month(base.projected_breakeven_month, scenario.horizon_months)}, "
        f"needs peak funding of {fmt.money(base.peak_funding, scenario.currency)} and runs at a "
        f"{fmt.pct(base.run_rate_margin)} cash margin by month {scenario.horizon_months}."
    )
    breadth = (
        f"It ranks first in {fmt.pct(share)} of all possible Velocity/Margin/Robustness "
        "weightings, so the result does not hinge on the exact default weights."
    )
    return [edge, numbers, breadth]


def _threshold_moves(thresholds: pd.DataFrame) -> list[dict]:
    """Every (input, direction) pair whose change flips the ranking, as flat records."""
    moves = []
    for t in thresholds.to_dict("records"):
        for direction in ("down", "up"):
            change = t[f"change_{direction}"]
            if change is not None and not pd.isna(change):
                moves.append({**t, "change": change, "new_winner": t[f"winner_{direction}"]})
    return moves


def _conditions(
    scenario: Scenario,
    result: ScoreResult,
    flips: pd.DataFrame,
    thresholds: pd.DataFrame,
    downside: pd.DataFrame,
) -> list[tuple[str, str | None]]:
    """Each condition under which the ranking changes, with the route that then wins."""
    out: list[tuple[str, str | None]] = []
    for f in flips.itertuples():
        side = "above" if f.direction == "up" else "below"
        out.append(
            (
                f"{f.dimension.title()} weight {side} {fmt.pct(f.flip_weight)} "
                f"(default {fmt.pct(f.default_weight)})",
                f.new_winner,
            )
        )
    moves = _threshold_moves(thresholds)
    for m in moves:
        word = "higher" if m["change"] > 0 else "lower"
        new_value = m["base_value"] * (1 + m["change"])
        out.append(
            (
                f"{_phrase(m['label'])} is {abs(m['change']):.0%} {word} than assumed "
                f"({format_input(scenario, m['input'], m['base_value'])} to "
                f"{format_input(scenario, m['input'], new_value)})",
                m["new_winner"],
            )
        )
    limit = fmt.pct(scenario.sensitivity.flip_search_max)
    for label in thresholds.label[~thresholds.label.isin([m["label"] for m in moves])]:
        out.append((f"the ranking holds for any change in {_phrase(label)} up to +/-{limit}", None))
    row = downside.set_index("route").loc[result.winner]
    stress = f"Downside case ({_downside_description(scenario)})"
    if row.rank_downside == 1:
        out.append((f"{stress}: it still ranks first", None))
    else:
        top = downside.loc[downside.rank_downside == 1, "route"].iloc[0]
        out.append((f"{stress}, where it drops to #{row.rank_downside}", top))
    return out


def _stops_winning(conditions: list[tuple[str, str | None]], scenario: Scenario) -> list[str]:
    return [
        f"{text[0].upper()}{text[1:]}: {scenario.label(new)} ranks first."
        if new
        else f"{text[0].upper()}{text[1:]}."
        for text, new in conditions
    ]


def _phrase(label: str) -> str:
    """Input label for use mid-sentence: 'Reseller network: CAC' -> "the reseller network's CAC"."""
    if ": " in label:
        route, name = label.split(": ", 1)
        return f"the {route.lower()}'s {name}"
    return label[0].lower() + label[1:]


def _sentence_case(text: str) -> str:
    return text[0].lower() + text[1:]


def _downside_description(scenario: Scenario) -> str:
    d = scenario.downside
    return (
        f"market {fmt.signed_pct(d.market_shock)}, main intermediary lost in "
        f"month {d.partner_loss_month:.0f}"
    )


def _most_fragile(thresholds: pd.DataFrame) -> dict | None:
    """The input whose smallest move (either direction) flips the ranking."""
    moves = _threshold_moves(thresholds)
    return min(moves, key=lambda m: abs(m["change"])) if moves else None


def _robustness_risk(scenario: Scenario, result: ScoreResult, fin: Financials) -> str:
    """Describe the recommended route's lowest-scoring robustness metric."""
    w = result.workings
    rows = w[(w.route == result.winner) & (w.dimension == "robustness")]
    weakest = rows.loc[rows.score.idxmin()]
    route = scenario.route(result.winner)
    retention = revenue_retention(fin.base[route.key], fin.downside[route.key])
    text = {
        "downside_revenue_retention": (
            f"Downside exposure: in the downside case ({_downside_description(scenario)}) "
            f"it keeps only {fmt.pct(retention)} of base-case revenue."
        ),
        "concentration_risk": (
            f"Concentration: rated {fmt.pct(route.concentration_risk)} on the concentration "
            "scale; a few accounts or channels carry most of the volume."
        ),
        "partner_dependency": (
            f"Partner dependency: {fmt.pct(route.partner_dependency)} of revenue runs through "
            "the largest intermediary; losing it removes that revenue for "
            f"~{scenario.downside.partner_loss_recovery_months:.0f} months while it is replaced."
        ),
        "lock_in_months": (
            f"Lock-in: the channel contract binds for {route.lock_in_months:.0f} months, "
            "which limits a switch of route if it under-delivers."
        ),
    }
    return text[weakest.metric]


def build_recommendation(scenario: Scenario) -> Recommendation:
    """Score, stress and summarise the scenario in plain English."""
    fin = run_financials(scenario)
    result = score_routes(scenario, fin)
    flips = weight_flips(scenario, result)
    sensitivity = tornado(scenario)
    top = list(sensitivity.input.head(scenario.sensitivity.top_inputs))
    thresholds = flip_thresholds(scenario, top)
    downside = downside_table(scenario)
    share = weight_space_share(scenario, result)[result.winner]

    win, run = result.winner, result.runner_up
    headline = (
        f"Enter through the {scenario.label(win).lower()}. It scores "
        f"{result.totals[win]:.0f}/100 on VMR against {result.totals[run]:.0f} for the "
        f"{scenario.label(run).lower()}, the strongest balance of speed, profit and resilience."
    )

    risks = []
    fragile = _most_fragile(thresholds)
    if fragile is not None:
        word = "higher" if fragile["change"] > 0 else "lower"
        risks.append(
            f"Assumption risk: if {_phrase(fragile['label'])} is {abs(fragile['change']):.0%} "
            f"{word} than assumed, {scenario.label(fragile['new_winner'])} overtakes."
        )
    base, stressed = fin.base[win], fin.downside[win]
    horizon = scenario.horizon_months
    risks.append(
        f"Cash exposure: cumulative cash bottoms at "
        f"{fmt.money(-base.peak_funding, scenario.currency)} and breakeven comes at "
        f"{fmt.month(base.projected_breakeven_month, horizon)}; in the downside case it "
        f"moves to {fmt.month(stressed.projected_breakeven_month, horizon)}."
    )
    risks.append(_robustness_risk(scenario, result, fin))

    conditions = _conditions(scenario, result, flips, thresholds, downside)
    # The fallback trigger: prefer an input condition (testable) over a weight change.
    to_runner_up = [c for c, new in conditions if new == run]
    fallback = min(to_runner_up, key=lambda c: "weight" in c) if to_runner_up else None
    route = scenario.route(win)
    review_month = route.months_to_first_revenue + route.ramp_months / 2
    next_steps = [
        "Test the two assumptions that move the ranking most before committing: "
        + " and ".join(_phrase(label) for label in sensitivity.label.head(2))
        + ".",
        f"Set a go/no-go review at month {review_month:.0f}, half-way through the ramp, "
        "against the base-case revenue curve.",
        f"Keep the {scenario.label(run).lower()} as the fallback"
        + (f". It becomes the better route if {_sentence_case(fallback)}." if fallback else "."),
    ]
    return Recommendation(
        route=win,
        headline=headline,
        why=_why(scenario, result, fin, share),
        stops_winning=_stops_winning(conditions, scenario),
        risks=risks,
        next_steps=next_steps,
    )
