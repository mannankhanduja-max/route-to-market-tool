"""Load, validate and modify the scenario file.

Every input in ``scenario.yaml`` is a mapping with a ``value`` and a
``source``. The loader keeps the numbers in typed dataclasses for the model
and the sources in a flat ``{dotted.path: source}`` dict for the docs, so no
number can reach the model without a stated origin.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCENARIO = Path(__file__).resolve().parents[2] / "data" / "scenario.yaml"

DIMENSIONS = ("velocity", "margin", "robustness")

# Metrics the scoring model knows how to compute, by dimension.
METRICS = {
    "velocity": ("months_to_first_revenue", "breakeven_month", "onboarding_months"),
    "margin": ("run_rate_margin", "cumulative_margin"),
    "robustness": (
        "downside_revenue_retention",
        "concentration_risk",
        "partner_dependency",
        "lock_in_months",
    ),
}

# Inputs that are shares and must stay within [0, 1] when perturbed.
FRACTION_INPUTS = frozenset(
    {
        "target_share",
        "gross_margin",
        "annual_churn",
        "reach",
        "channel_take",
        "concentration_risk",
        "partner_dependency",
    }
)

# Plain-English names for inputs, used in charts and the recommendation.
INPUT_LABELS = {
    "annual_market_size": "market size",
    "target_share": "target share",
    "annual_price_per_customer": "price per customer",
    "gross_margin": "gross margin",
    "annual_churn": "churn",
    "onboarding_months": "onboarding time",
    "months_to_first_revenue": "time to first revenue",
    "ramp_months": "ramp time",
    "reach": "reach",
    "channel_take": "channel take",
    "cac": "CAC",
    "fixed_setup_cost": "set-up cost",
    "monthly_fixed_cost": "monthly fixed cost",
    "concentration_risk": "concentration risk",
    "partner_dependency": "partner dependency",
    "lock_in_months": "lock-in",
    "market_shock": "downside market shock",
    "partner_loss_month": "partner-loss month",
    "partner_loss_recovery_months": "partner-loss recovery time",
}


@dataclass(frozen=True)
class Market:
    """Market-level inputs shared by every route."""

    annual_market_size: float
    target_share: float
    annual_price_per_customer: float
    gross_margin: float
    annual_churn: float


@dataclass(frozen=True)
class Route:
    """Inputs that describe one route to market."""

    key: str
    label: str
    description: str
    onboarding_months: float
    months_to_first_revenue: float
    ramp_months: float
    reach: float
    channel_take: float
    cac: float
    fixed_setup_cost: float
    monthly_fixed_cost: float
    concentration_risk: float
    partner_dependency: float
    lock_in_months: float


@dataclass(frozen=True)
class Downside:
    """Stress case: a market-size shock plus the loss of the main intermediary."""

    market_shock: float
    partner_loss_month: float
    partner_loss_recovery_months: float


@dataclass(frozen=True)
class Anchor:
    """Scaling for one metric: ``worst`` maps to 0, ``best`` to 100."""

    best: float
    worst: float
    weight: float


@dataclass(frozen=True)
class SensitivitySettings:
    """Numerical settings for the sensitivity analysis."""

    tornado_swing: float
    weight_resolution: float
    weight_map_step: float
    flip_search_max: float
    flip_search_step: float
    top_inputs: int


@dataclass(frozen=True)
class Scenario:
    """A complete, validated scenario."""

    name: str
    currency: str
    horizon_months: int
    market: Market
    routes: tuple[Route, ...]
    downside: Downside
    weights: dict[str, float]
    anchors: dict[str, dict[str, Anchor]]
    sensitivity: SensitivitySettings
    sources: dict[str, str]

    @property
    def route_keys(self) -> tuple[str, ...]:
        return tuple(r.key for r in self.routes)

    def route(self, key: str) -> Route:
        """Return the route with this key."""
        for r in self.routes:
            if r.key == key:
                return r
        raise KeyError(f"Unknown route: {key}")

    def label(self, key: str) -> str:
        return self.route(key).label

    def get(self, path: str) -> float:
        """Read an input by dotted path, e.g. ``routes.reseller.cac``."""
        head, *rest = path.split(".")
        if head == "routes":
            return getattr(self.route(rest[0]), rest[1])
        if head == "weights":
            return self.weights[rest[0]]
        if head in ("market", "downside"):
            return getattr(getattr(self, head), rest[0])
        raise KeyError(f"Unknown input path: {path}")

    def with_value(self, path: str, value: float) -> Scenario:
        """Return a copy with one input changed. Shares are clipped to [0, 1]."""
        head, *rest = path.split(".")
        if rest[-1] in FRACTION_INPUTS:
            value = min(max(value, 0.0), 1.0)
        if head == "routes":
            key, name = rest
            self.route(key)  # raises on an unknown route
            routes = tuple(replace(r, **{name: value}) if r.key == key else r for r in self.routes)
            return replace(self, routes=routes)
        if head == "weights":
            return replace(self, weights={**self.weights, rest[0]: value})
        if head in ("market", "downside"):
            return replace(self, **{head: replace(getattr(self, head), **{rest[0]: value})})
        raise KeyError(f"Unknown input path: {path}")

    def with_weights(self, weights: dict[str, float]) -> Scenario:
        """Return a copy with new dimension weights."""
        return replace(self, weights=dict(weights))

    def input_paths(self) -> list[str]:
        """Dotted paths of every model input that a sensitivity test can move."""
        paths = [f"market.{f.name}" for f in fields(Market)]
        route_inputs = [f.name for f in fields(Route) if f.name not in _ROUTE_TEXT]
        paths += [f"routes.{r.key}.{name}" for r in self.routes for name in route_inputs]
        paths += [f"downside.{f.name}" for f in fields(Downside)]
        return paths

    def input_label(self, path: str) -> str:
        """Readable name for an input path, e.g. 'Reseller network: channel take'."""
        parts = path.split(".")
        name = INPUT_LABELS.get(parts[-1], parts[-1])
        if parts[0] == "routes":
            return f"{self.label(parts[1])}: {name}"
        return name[0].upper() + name[1:]


_ROUTE_TEXT = frozenset({"key", "label", "description"})


class ScenarioError(ValueError):
    """Raised when the scenario file is incomplete or inconsistent."""


def _value(node: Any, path: str, sources: dict[str, str]) -> float:
    """Unpack a ``{value, source}`` leaf and record its source."""
    if not isinstance(node, dict) or "value" not in node:
        raise ScenarioError(f"{path}: expected a mapping with 'value' and 'source'")
    source = str(node.get("source", "")).strip()
    if not source:
        raise ScenarioError(f"{path}: every assumption needs a source or rationale")
    sources[path] = source
    return node["value"]


def _section(raw: dict[str, Any], cls: type, prefix: str, sources: dict[str, str]) -> dict:
    """Read every numeric field of a dataclass from a YAML section."""
    out = {}
    for f in fields(cls):
        if f.name in _ROUTE_TEXT:
            continue
        if f.name not in raw:
            raise ScenarioError(f"{prefix}.{f.name}: missing")
        out[f.name] = _value(raw[f.name], f"{prefix}.{f.name}", sources)
    return out


def _anchors(raw: dict[str, Any], sources: dict[str, str]) -> dict[str, dict[str, Anchor]]:
    anchors: dict[str, dict[str, Anchor]] = {}
    for dim in DIMENSIONS:
        section = raw.get(dim) or {}
        unknown = set(section) - set(METRICS[dim])
        if unknown:
            raise ScenarioError(f"scoring.{dim}: unknown metrics {sorted(unknown)}")
        anchors[dim] = {}
        for metric, node in section.items():
            path = f"scoring.{dim}.{metric}"
            if not str(node.get("source", "")).strip():
                raise ScenarioError(f"{path}: every anchor needs a source or rationale")
            if node["best"] == node["worst"]:
                raise ScenarioError(f"{path}: best and worst anchors must differ")
            sources[path] = node["source"]
            anchors[dim][metric] = Anchor(
                best=node["best"], worst=node["worst"], weight=node["weight"]
            )
        if not anchors[dim]:
            raise ScenarioError(f"scoring.{dim}: needs at least one metric")
    return anchors


def _validate(s: Scenario) -> None:
    if s.horizon_months < 1:
        raise ScenarioError("horizon_months must be positive")
    if sum(s.weights.values()) <= 0 or min(s.weights.values()) < 0:
        raise ScenarioError("weights must be non-negative and not all zero")
    for r in s.routes:
        if r.ramp_months <= 0:
            raise ScenarioError(f"routes.{r.key}.ramp_months must be positive")
        if r.months_to_first_revenue < r.onboarding_months:
            raise ScenarioError(
                f"routes.{r.key}: first revenue cannot come before onboarding is complete"
            )
        for name in FRACTION_INPUTS & {f.name for f in fields(Route)}:
            if not 0 <= getattr(r, name) <= 1:
                raise ScenarioError(f"routes.{r.key}.{name} must be between 0 and 1")


def parse_scenario(raw: dict[str, Any]) -> Scenario:
    """Build a validated :class:`Scenario` from the parsed YAML mapping."""
    sources: dict[str, str] = {}
    meta = raw["scenario"]
    horizon = int(_value(meta["horizon_months"], "scenario.horizon_months", sources))
    market = Market(**_section(raw["market"], Market, "market", sources))
    routes = tuple(
        Route(
            key=key,
            label=node["label"],
            description=node.get("description", ""),
            **_section(node, Route, f"routes.{key}", sources),
        )
        for key, node in raw["routes"].items()
    )
    downside = Downside(**_section(raw["downside"], Downside, "downside", sources))
    weights = {d: float(_value(raw["weights"][d], f"weights.{d}", sources)) for d in DIMENSIONS}
    sens = _section(raw["sensitivity"], SensitivitySettings, "sensitivity", sources)
    scenario = Scenario(
        name=meta["name"],
        currency=meta["currency"],
        horizon_months=horizon,
        market=market,
        routes=routes,
        downside=downside,
        weights=weights,
        anchors=_anchors(raw["scoring"], sources),
        sensitivity=SensitivitySettings(**{**sens, "top_inputs": int(sens["top_inputs"])}),
        sources=sources,
    )
    _validate(scenario)
    return scenario


def load_scenario(path: str | Path = DEFAULT_SCENARIO) -> Scenario:
    """Load and validate a scenario YAML file."""
    with open(path, encoding="utf-8") as fh:
        return parse_scenario(yaml.safe_load(fh))
