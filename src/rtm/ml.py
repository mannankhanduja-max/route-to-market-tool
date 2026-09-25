"""Machine-learning check: logistic regression and random forest surrogates of the VMR model.

There is no historical data on market entries to learn from, so the models
learn the VMR model itself:

1. Draw many scenarios, moving every input at once uniformly within
   +/- ``input_range`` of its base value (shares clipped to [0, 1]).
2. Label each draw with the route the VMR model ranks first.
3. Train a logistic regression and a random forest to predict that label
   from the inputs, and test them on held-out draws.

This answers three questions the one-at-a-time tornado cannot:

* how often each route wins when all assumptions are uncertain together;
* how confident each model is about the base case;
* which assumptions decide the winner when they move together
  (permutation importance, plus the logistic regression's direction).

The models describe this model's behaviour, not the real market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rtm.config import Scenario
from rtm.scoring import total_scores

LOGISTIC, FOREST = "Logistic regression", "Random forest"
MIN_PER_CLASS_TO_STRATIFY = 2


def varied_inputs(scenario: Scenario) -> list[str]:
    """Inputs that are sampled: every model input that is non-zero in the base case."""
    return [p for p in scenario.input_paths() if scenario.get(p) != 0]


def sample_scenarios(scenario: Scenario) -> pd.DataFrame:
    """Draw scenarios and label each with its winning route.

    Returns one row per draw: the value of every varied input, the winning
    route key, and the winner's lead over the next-best route.
    """
    settings = scenario.ml
    paths = varied_inputs(scenario)
    base = np.array([scenario.get(p) for p in paths])
    rng = np.random.default_rng(settings.seed)
    factors = rng.uniform(
        1 - settings.input_range, 1 + settings.input_range, (settings.n_samples, len(paths))
    )
    rows = []
    for draw in factors * base:
        s = scenario
        for path, value in zip(paths, draw, strict=True):
            s = s.with_value(path, float(value))
        totals = total_scores(s)
        ranked = sorted(totals.values(), reverse=True)
        rows.append(
            [s.get(p) for p in paths] + [max(totals, key=totals.get), ranked[0] - ranked[1]]
        )
    return pd.DataFrame(rows, columns=[*paths, "winner", "lead"])


@dataclass(frozen=True)
class MLResult:
    """Win rates under uncertainty, model accuracy, base-case probabilities, importances."""

    sample: pd.DataFrame
    win_rates: pd.Series
    trained: bool
    baseline_accuracy: float | None
    accuracy: dict[str, float]
    base_probability: pd.DataFrame
    importance: pd.DataFrame


def _winner_coefficients(model, winner: str) -> np.ndarray:
    """Standardised logistic-regression coefficients pushing towards ``winner``.

    Positive means a higher input value makes the winner more likely to stay on top.
    """
    lr = model[-1]
    classes = list(lr.classes_)
    if winner not in classes:
        return np.zeros(lr.coef_.shape[1])
    if len(classes) == 2:  # binary: one row of coefficients, for classes[1]
        return lr.coef_[0] if winner == classes[1] else -lr.coef_[0]
    return lr.coef_[classes.index(winner)]


def _probabilities(model, x: pd.DataFrame, keys: tuple[str, ...]) -> dict[str, float]:
    proba = dict(zip(model.classes_, model.predict_proba(x)[0], strict=True))
    return {k: float(proba.get(k, 0.0)) for k in keys}


def train_models(scenario: Scenario, sample: pd.DataFrame, winner: str) -> MLResult:
    """Fit both models on the sample and summarise what they learned.

    ``winner`` is the base-case winner; the logistic-regression direction is
    reported with respect to it. Inputs are sorted by the average permutation
    importance of the two models (the drop in test accuracy when that input
    is shuffled).
    """
    settings = scenario.ml
    paths = varied_inputs(scenario)
    x, y = sample[paths], sample["winner"]
    keys = scenario.route_keys
    win_rates = y.value_counts(normalize=True).reindex(keys, fill_value=0.0)
    labels = [scenario.input_label(p) for p in paths]

    stratify = y if y.value_counts().min() >= MIN_PER_CLASS_TO_STRATIFY else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=settings.test_share, random_state=settings.seed, stratify=stratify
    )
    if y_train.nunique() < 2:
        # Nothing to classify: one route wins every training draw.
        empty = pd.DataFrame({"input": paths, "label": labels})
        return MLResult(sample, win_rates, False, None, {}, pd.DataFrame(), empty)
    models = {
        LOGISTIC: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=settings.lr_regularisation, max_iter=5000),
        ),
        FOREST: RandomForestClassifier(
            n_estimators=settings.rf_trees,
            min_samples_leaf=settings.rf_min_leaf,
            random_state=settings.seed,
            n_jobs=-1,
        ),
    }
    base_x = pd.DataFrame([[scenario.get(p) for p in paths]], columns=paths)
    accuracy, probability, importance = {}, {}, {"input": paths, "label": labels}
    for name, model in models.items():
        model.fit(x_train, y_train)
        if name == FOREST:
            model.set_params(n_jobs=1)  # parallelise the shuffles below instead of the trees
        accuracy[name] = float((model.predict(x_test) == y_test).mean())
        probability[name] = _probabilities(model, base_x, keys)
        perm = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=settings.importance_repeats,
            random_state=settings.seed,
            n_jobs=-1,
        )
        importance[f"{name} importance"] = perm.importances_mean
    importance["Logistic regression direction"] = _winner_coefficients(models[LOGISTIC], winner)
    table = pd.DataFrame(importance)
    both = table[[f"{LOGISTIC} importance", f"{FOREST} importance"]].mean(axis=1)
    table = table.iloc[both.sort_values(ascending=False, kind="stable").index]
    baseline = float((y_test == y_train.mode().iloc[0]).mean())
    return MLResult(
        sample=sample,
        win_rates=win_rates,
        trained=True,
        baseline_accuracy=baseline,
        accuracy=accuracy,
        base_probability=pd.DataFrame(probability),
        importance=table.reset_index(drop=True),
    )


def run_ml_check(scenario: Scenario, winner: str) -> MLResult:
    """Sample scenarios, then train and evaluate both models."""
    return train_models(scenario, sample_scenarios(scenario), winner)


def model_table(scenario: Scenario, ml: MLResult) -> pd.DataFrame:
    """Accuracy on held-out draws and base-case win probabilities, one row per model.

    The first row is the benchmark a model has to beat: always predicting the
    route that won most often in training.
    """
    labels = {k: f"P({scenario.label(k)})" for k in scenario.route_keys}
    rows = [
        {
            "model": "Benchmark: always pick the most common winner",
            "test accuracy": ml.baseline_accuracy,
        }
    ]
    for name in (LOGISTIC, FOREST):
        probs = ml.base_probability[name]
        rows.append(
            {
                "model": name,
                "test accuracy": ml.accuracy[name],
                **{labels[k]: probs[k] for k in scenario.route_keys},
            }
        )
    return pd.DataFrame(rows)
