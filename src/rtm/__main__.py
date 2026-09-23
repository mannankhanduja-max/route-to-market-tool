"""Print the ranking and the full scoring workings: ``python -m rtm [scenario.yaml]``."""

from __future__ import annotations

import sys

import pandas as pd

from rtm.config import DEFAULT_SCENARIO, load_scenario
from rtm.scoring import score_routes, summary_table


def main(path: str = str(DEFAULT_SCENARIO)) -> None:
    scenario = load_scenario(path)
    result = score_routes(scenario)
    pd.set_option("display.width", 120)
    print(f"Scenario: {scenario.name}\n")
    print("Ranking (weights: " + ", ".join(f"{k} {v:.0%}" for k, v in result.weights.items()) + ")")
    print(summary_table(scenario, result).round(1).to_string(index=False))
    print("\nWorkings: raw value -> 0-100 score between anchors -> weighted into dimension")
    print(result.workings.round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
