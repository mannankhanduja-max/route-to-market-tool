"""Print ranking, workings, sensitivity and recommendation: ``python -m rtm [scenario.yaml]``."""

from __future__ import annotations

import sys

import pandas as pd

from rtm.config import DEFAULT_SCENARIO, load_scenario
from rtm.ml import model_table, run_ml_check
from rtm.recommend import build_recommendation
from rtm.scoring import score_routes, summary_table
from rtm.sensitivity import downside_table, tornado, weight_flips


def main(path: str = str(DEFAULT_SCENARIO)) -> None:
    scenario = load_scenario(path)
    result = score_routes(scenario)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    weights = ", ".join(f"{k} {v:.0%}" for k, v in result.weights.items())
    print(f"Scenario: {scenario.name}\n")
    print(f"Ranking (weights: {weights})")
    print(summary_table(scenario, result).round(1).to_string(index=False))
    print("\nWorkings: raw value -> 0-100 score between anchors -> weighted into dimension")
    print(result.workings.round(2).to_string(index=False))
    print("\nWeight flips (nearest weight at which the top route changes)")
    print(weight_flips(scenario, result).round(3).to_string(index=False))
    print("\nTornado (top 10 inputs by movement in the winner's lead)")
    cols = ["label", "base_value", "lead_low", "lead_high", "winner_low", "winner_high"]
    print(tornado(scenario)[cols].head(10).round(2).to_string(index=False))
    print("\nDownside case")
    print(downside_table(scenario).round(2).to_string(index=False))
    ml = run_ml_check(scenario, result.winner)
    print(f"\nMachine-learning check ({len(ml.sample):,} scenarios, every input moved at once)")
    print("Win rates: " + ", ".join(f"{k} {v:.0%}" for k, v in ml.win_rates.items()))
    if ml.trained:
        print(model_table(scenario, ml).round(3).to_string(index=False))
        cols = ["label", "Logistic regression importance", "Random forest importance"]
        print(ml.importance[cols].head(scenario.ml.top_features).round(3).to_string(index=False))
    print("\n" + build_recommendation(scenario).to_markdown())


if __name__ == "__main__":
    main(*sys.argv[1:])
