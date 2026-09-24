"""Number formatting shared by the charts, the recommendation and the app."""

from __future__ import annotations

import math

CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}


def money(amount: float, currency: str) -> str:
    """Compact money figure, e.g. '€2.4m' or '€850k'."""
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e6:
        return f"{sign}{symbol}{amount / 1e6:.1f}m"
    if amount >= 1e3:
        return f"{sign}{symbol}{amount / 1e3:.0f}k"
    return f"{sign}{symbol}{amount:.0f}"


def month(value: float, horizon: int) -> str:
    """Breakeven month in words, flagging values projected past the horizon."""
    if math.isinf(value):
        return "not in sight"
    if value > horizon:
        return f"month {value:.0f} (projected)"
    return f"month {value:.0f}"


def pct(value: float) -> str:
    """Whole-number percentage, e.g. '35%'."""
    return f"{value:.0%}"


def signed_pct(value: float) -> str:
    """Signed percentage change, e.g. '+25%' or '-18%'."""
    return f"{value:+.0%}"
