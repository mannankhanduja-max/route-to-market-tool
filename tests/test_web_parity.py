"""The browser version (docs/index.html) must give the same answers as the Python model."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from rtm.finance import run_financials
from rtm.scoring import score_routes
from rtm.sensitivity import flip_thresholds, tornado, weight_flips, weight_space_share

PAGE = Path(__file__).resolve().parents[1] / "docs" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _run_page_model() -> dict:
    """Run the page's model script on its built-in example scenario."""
    html = PAGE.read_text(encoding="utf-8")
    model = re.search(r'<script id="model">(.*?)</script>', html, re.S).group(1)
    example = re.search(r"const EXAMPLE = (\{.*?\n\});", html, re.S).group(1)
    script = (
        model
        + f"\nconst EXAMPLE = {example};\nconst a = analyse(EXAMPLE);\n"
        + "console.log(JSON.stringify({totals: a.res.totals, stressed: a.stressed.ranking,"
        + " flips: a.flips.map(f => [f.dim, f.dir, f.w, f.win]), share: a.share,"
        + " torn: a.torn.slice(0, 5).map(t => [t.path.join('.'), t.leadLow, t.leadHigh]),"
        + " thr: a.thresholds.map(t => [t.down && t.down.change, t.up && t.up.change]),"
        + " breakeven: a.fin.base.map(f => f.projected)}));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _js_path(scenario, path: str) -> str:
    """'routes.reseller.cac' -> 'routes.1.cac' (the page indexes routes by position)."""
    parts = path.split(".")
    if parts[0] == "routes":
        parts[1] = str(scenario.route_keys.index(parts[1]))
    return ".".join(parts)


def test_page_example_matches_scenario_yaml(scenario):
    js = _run_page_model()
    fin = run_financials(scenario)
    result = score_routes(scenario, fin)
    keys = scenario.route_keys

    assert js["totals"] == pytest.approx([result.totals[k] for k in keys], abs=1e-9)
    stressed = score_routes(scenario, fin, stressed=True).ranking
    assert [keys[i] for i in js["stressed"]] == stressed
    assert js["share"] == pytest.approx(list(weight_space_share(scenario, result)), abs=1e-9)
    assert js["breakeven"] == pytest.approx(
        [fin.base[k].projected_breakeven_month for k in keys], abs=1e-9
    )

    flips = weight_flips(scenario, result)
    assert [(d, dr, keys[w]) for d, dr, _, w in js["flips"]] == list(
        zip(flips.dimension, flips.direction, flips.new_winner, strict=True)
    )
    assert [w for *_, w, _ in js["flips"]] == pytest.approx(list(flips.flip_weight), abs=1e-9)

    table = tornado(scenario).head(5)
    assert [p for p, *_ in js["torn"]] == [_js_path(scenario, p) for p in table.input]
    assert [lo for _, lo, _ in js["torn"]] == pytest.approx(list(table.lead_low), abs=1e-9)

    thresholds = flip_thresholds(scenario, list(tornado(scenario).input.head(3)))
    for (down, up), row in zip(js["thr"], thresholds.itertuples(), strict=True):
        for js_change, py_change in ((down, row.change_down), (up, row.change_up)):
            if pd.isna(py_change):
                assert js_change is None
            else:
                assert js_change == pytest.approx(py_change)
