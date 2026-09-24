# Hypothesis tree

**Governing question.** Which route to market should a mid-market B2B software
company use to enter a new European country: direct sales, a reseller
network, or a strategic partner?

**Lead hypothesis.** A reseller network is the best entry route: it reaches
revenue nearly as fast as a partner, keeps more of each euro than a partner,
and does not stake the whole market on one relationship.

The hypothesis holds only if all three branches below hold. Each branch maps to
one VMR dimension, and each leaf maps to an input or output of the model.

```
Reseller network is the best entry route
│
├── 1. VELOCITY: it reaches revenue and payback fast enough
│   ├── 1.1 Resellers can be signed and certified within ~5 months      -> onboarding_months
│   ├── 1.2 First deals close within ~3 months of onboarding            -> months_to_first_revenue
│   └── 1.3 The route pays back within about three years                -> breakeven_month
│
├── 2. MARGIN: it keeps enough of each euro once running
│   ├── 2.1 Reseller margin stays near 30% of list price                -> channel_take
│   ├── 2.2 Channel support costs stay small next to revenue            -> monthly_fixed_cost, cac
│   └── 2.3 Resellers reach ~75% of the target segment                  -> reach
│
└── 3. ROBUSTNESS: it survives a bad year
    ├── 3.1 No single reseller carries more than ~35% of revenue        -> partner_dependency
    ├── 3.2 Agreements can be exited within a year                      -> lock_in_months
    └── 3.3 Revenue holds up if the market shrinks and a reseller leaves -> downside_revenue_retention
```

## What the model says about each branch

| Branch | Verdict (base case) | Evidence |
|---|---|---|
| 1. Velocity | **Partly holds.** Faster than direct, slower than a partner. Payback is projected at month 37, just outside the horizon. | Velocity 56 vs 68 (partner) and 32 (direct). |
| 2. Margin | **Holds.** The highest Margin score of the three routes. | Margin 61 vs 55 (partner) and 44 (direct). |
| 3. Robustness | **Holds against the partner, not against direct.** | Robustness 55 vs 16 (partner) and 77 (direct). |

The routes trade off against each other, so the answer depends on how the
dimensions are weighted. The sensitivity analysis shows the reseller network
wins across 54% of all possible weightings, including the default.

## What would disprove the hypothesis

| If this turns out true... | ...then | Test before committing |
|---|---|---|
| Resellers demand a margin above ~38% (vs 30% assumed) | The strategic partner overtakes | Term-sheet conversations with 3-5 candidate resellers |
| The partner accepts a revenue share below ~35% (vs 45%) | The strategic partner overtakes | Early negotiation with the leading partner candidate |
| Gross margin is ~18% lower than assumed (64% vs 78%) | Direct sales overtakes (the channel take eats a larger share of a thinner margin) | Cost-to-serve estimate for the new country (hosting, localisation, support) |
| Speed matters much more than profit (Velocity weight above ~64%) | The strategic partner overtakes | Agree the weights with the leadership team before scoring |
| Resilience matters much more (Robustness weight above ~48%) | Direct sales overtakes | Same |

Numbers in this page come from `docs/results.md`, which `make report`
regenerates from `data/scenario.yaml`.
