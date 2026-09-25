# Methodology

This page documents every formula in the model and flags the ones that are
judgment calls. All numbers come from `data/scenario.yaml`; none are hard-coded.

## 1. Financial model (`src/rtm/finance.py`)

Each route is projected month by month over the horizon (36 months).

| Quantity | Formula |
|---|---|
| Steady-state revenue (annual) | market size x target share x route reach |
| Ramp share in month *t* | 0 before the first-revenue month, then a smooth S-curve `3x² - 2x³` with `x = (t - first revenue month + 1) / ramp months`, capped at 1 |
| Revenue | steady-state revenue / 12 x ramp share |
| Active customers | revenue x 12 / annual price per customer |
| Acquisition cost | (new customers + churn replacements) x CAC |
| Contribution | revenue x (gross margin - channel take) |
| Cash flow | contribution - acquisition cost - monthly fixed cost - set-up cost (month 1 only) |
| Breakeven month | first month in which cumulative cash is back at zero or above |

**Revenue definition.** "Revenue" is what end customers pay. The channel take
(reseller margin or partner revenue share) is a cost on top of the cost of
delivery, which is why contribution is revenue x (gross margin - take).

**Downside case.** The same projection with two shocks together:

- the market is smaller by the downside shock (default -30%);
- in the partner-loss month (default 18) the route loses its
  `partner_dependency` share of revenue and wins it back in a straight line
  over the recovery period (default 12 months). Direct sales has no
  dependency, so it only takes the market shock.

### Judgment calls

- **S-shaped ramp.** Real ramps are rarely linear; an S-curve (slow start, fast
  middle, plateau) is the standard simplification. The ramp length per route
  is the assumption that matters, and it is in the YAML.
- **Projected breakeven.** If a route has not paid back by month 36, its
  breakeven is projected forward at the final month's cash flow (a straight
  line). It is flagged "(projected)" everywhere it appears. A route still
  losing money in month 36 gets "not in sight" and a Velocity breakeven score
  of zero.
- **No discounting.** Cash is not discounted. Over three years at typical
  hurdle rates this changes the numbers slightly, not the ranking.
- **One churn rate for all routes.** In practice churn may differ by channel
  (a partner that owns the relationship can hold customers better or worse);
  the model does not assume either.

## 2. VMR scoring (`src/rtm/scoring.py`)

### Step 1: scale each metric to 0-100

```
score = 100 x clip((value - worst) / (best - worst), 0, 1)
```

`best` and `worst` are fixed anchors in the YAML. The same formula works for
"lower is better" metrics (months, risk) because `best < worst` flips the
sign.

**Judgment call: fixed anchors, not relative scaling.** A common shortcut is to
give the best route 100 and the worst route 0. That makes scores unstable: a
route's score changes when a different route's inputs change, and a trivially
small gap becomes a 100-point gap. Fixed anchors avoid both problems. The
price is that the anchors themselves are choices, so each one has a rationale
in the YAML.

### Step 2: average within each dimension

| Dimension | Metric | Weight in dimension | Anchors (worst -> best) |
|---|---|---|---|
| Velocity | Months to first revenue | 35% | 18 -> 3 |
| | Breakeven month (projected if past horizon) | 45% | 60 -> 12 |
| | Onboarding months | 20% | 12 -> 0 |
| Margin | Run-rate cash margin (final month) | 60% | 0% -> 50% |
| | Cumulative cash / cumulative revenue (36 months) | 40% | -50% -> 25% |
| Robustness | Downside revenue retention | 40% | 40% -> 100% |
| | Concentration risk (0-1 judgment scale) | 20% | 1 -> 0 |
| | Partner dependency (share of revenue via largest intermediary) | 20% | 1 -> 0 |
| | Contract lock-in months | 20% | 36 -> 0 |

**Judgment call: Margin blends run-rate and cumulative.** Run-rate margin alone
would ignore set-up and ramp costs; cumulative margin alone would punish a
route that is expensive to start but very profitable once running. The 60/40
split favours the steady state because the entry decision is long term.

**Judgment call: concentration risk is a rating, not a measurement.** It is a
0-1 score assigned per route with a written reason. Partner dependency, by
contrast, is a share of revenue.

### Step 3: combine with dimension weights

```
total = w_V x Velocity + w_M x Margin + w_R x Robustness     (weights sum to 1)
```

Default weights are 40 / 35 / 25 from the VMR framework. The dashboard lets
the reader change them, and the sensitivity analysis shows how far they can
move before the answer changes.

## 3. Sensitivity analysis (`src/rtm/sensitivity.py`)

| View | What it does |
|---|---|
| Weight sweep | Moves one dimension weight from 0% to 100% in 0.1 percentage-point steps; the other two share the rest in their default ratio. Reports the nearest weight above and below the default where the top route changes. |
| Weight map | Evaluates every weight combination on a 5% grid over the triangle of possible weights and reports the share of combinations each route wins. |
| Tornado | Moves each input -/+20% on its own and records the winner's lead over the best other route. Inputs that are zero in the base case are skipped (a relative change leaves them at zero). Shares are clipped to [0, 1]. |
| Flip thresholds | For the three inputs at the top of the tornado, searches outwards in 1% steps up to +/-100% for the smallest change that changes the top route. |
| Downside ranking | Re-scores Velocity and Margin on the downside financials and ranks the routes again. Robustness already compares base and downside, so it does not change. |

The model is deterministic: there is no random sampling, so results are
reproducible without a seed.

## 4. Recommendation (`src/rtm/recommend.py`)

The text is generated from the results, so it cannot drift from the numbers.

- **Why it wins:** the weighted points gained and lost against the runner-up
  by dimension, the winner's breakeven, peak funding and run-rate margin, and
  the share of weight combinations it wins.
- **When it stops winning:** every weight flip, the flip threshold of each top
  tornado input, and the downside ranking.
- **Top three risks (a rule, and a judgment call):**
  1. the input whose smallest change flips the ranking (the most fragile assumption);
  2. cash exposure: peak funding and breakeven in the base and downside cases;
  3. the winner's lowest-scoring Robustness metric, described in plain English.
- **Next steps:** test the two most sensitive assumptions; review half-way
  through the winner's ramp; keep the runner-up as the fallback with the
  condition under which it takes over.

## 5. Machine-learning check (`src/rtm/ml.py`)

There is no historical data on market entries in this project, so the models
are **surrogates**: they learn the VMR model's own decision rule. This shows
how the recommendation behaves when every assumption is uncertain at once,
which the one-at-a-time tornado cannot.

| Step | What happens |
|---|---|
| Sample | 2,000 draws. Every non-zero input is drawn independently and uniformly within ±25% of its base value; shares are clipped to [0, 1]. Seed 42. |
| Label | Each draw is scored with the same VMR arithmetic (`total_scores`, a DataFrame-free copy of `score_routes`, tested to match it) and labelled with the top route. |
| Split | 75% train, 25% test, stratified by winner. |
| Logistic regression | Multinomial, on standardised inputs, C = 1.0. |
| Random forest | 200 trees, at least 5 draws per leaf. |
| Accuracy | Share of test draws where the predicted winner is right, against the benchmark of always predicting the most common winner. |
| Importance | Permutation importance on the test draws: the drop in accuracy when one input is shuffled (5 repeats). Inputs are ranked by the average of the two models. |
| Direction | The logistic regression's standardised coefficient for the base-case winner. Positive means a higher value makes it more likely to stay on top. |

### Judgment calls

- **Independent uniform draws.** Real assumptions are correlated (a weak
  market usually means lower share *and* longer ramps). Independent draws are
  the neutral starting point; the downside case covers one correlated shock.
- **±25% for every input.** Some inputs are much less certain than others.
  A per-input range would be better once there is evidence to set it.
- **Surrogates, not forecasts.** The models learn how this tool decides, not
  how markets behave. High accuracy means the decision rule is simple to learn
  from the inputs; it says nothing about whether the inputs are right.

### The browser version

`docs/index.html` trains the same two models in JavaScript, in a background
worker, on whatever inputs the viewer enters. It follows the steps above with
three differences, made so it finishes in about five seconds in a browser:

- its own seeded random numbers (mulberry32, seed 42), so individual draws
  differ from Python's;
- the logistic regression is fitted by gradient descent with momentum (400
  steps) on the same objective, cross-entropy plus an L2 penalty with C = 1;
- the random forest searches splits over 64 quantile bins per input rather
  than every distinct value, as histogram-based forests such as LightGBM do.

`tests/test_web_parity.py` runs the page's code under node and checks that it
reaches the same conclusions as Python on the example:
- win rates within 3 points;
- both models beat the benchmark;
- both favour the same base-case winner;
- both channel takes are among the four most important inputs.

## 6. Limitations

- All inputs are synthetic. The tool shows the reasoning; the numbers need to be
  replaced with market research before a real decision.
- Routes are compared as alternatives. Hybrid models (for example resellers for
  the mid-market plus a small direct team for key accounts) are not modelled.
- One-at-a-time sensitivity does not capture inputs moving together; the
  downside case is the only joint stress.
- The concentration rating and the scoring anchors are judgment calls. Changing
  them changes scores, which is why they sit in the YAML next to their rationale.
