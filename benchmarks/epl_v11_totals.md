# EPL GAH v1.1 Totals Experiment

GAH v1.1 keeps the Asian-handicap core unchanged and changes only the totals engine.

The totals distribution is an online mixture of the GAH Dixon-Coles distribution and
a training-window league-average Poisson distribution. The mixture weight is fitted
only from earlier out-of-sample total-goal likelihoods.

## Protocol

- Seasons: 2021-22, 2022-23, 2023-24, 2024-25, 2025-26
- Initial training window: 380 chronological matches
- Expanding walk-forward evaluation
- Dixon-Coles refit every 20 evaluated matches or when a new team requires it
- Totals blend warm-up: 100 prior OOS predictions
- Totals blend lookback: 380 prior OOS predictions
- Asian handicap remains pure GAH v1

## Overall totals

| Metric | GAH v1.1 blend | GAH v1 | Naive | v1.1 vs v1 | v1.1 vs naive |
|---|---:|---:|---:|---:|---:|
| Total-goals MAE ↓ | 1.3191 | 1.3448 | 1.3140 | +1.92% | -0.38% |
| Total-goals NLL ↓ | 1.9027 | 1.9150 | 1.9047 | +0.64% | +0.10% |
| Over 2.5 Brier ↓ | 0.2425 | 0.2454 | 0.2449 | +1.15% | +0.99% |

Mean GAH weight in totals blend: **0.431**

Evaluated predictions: **1513**

## Handicap guardrail

The AH engine is unchanged from v1:

- Goal-margin MAE: **1.3465**
- Home -0.5 Brier: **0.2157**

## By season

| Season | N | Blend MAE | v1 MAE | Naive MAE | Blend O2.5 Brier | v1 O2.5 | Naive O2.5 | Mean GAH weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-23 | 377 | 1.4068 | 1.4195 | 1.4239 | 0.2429 | 0.2427 | 0.2492 | 0.577 |
| 2023-24 | 378 | 1.3458 | 1.3616 | 1.3464 | 0.2351 | 0.2384 | 0.2369 | 0.403 |
| 2024-25 | 379 | 1.2900 | 1.3235 | 1.2641 | 0.2429 | 0.2440 | 0.2456 | 0.479 |
| 2025-26 | 379 | 1.2342 | 1.2753 | 1.2223 | 0.2492 | 0.2563 | 0.2481 | 0.265 |

## Decision

**Promote GAH v1.1 totals calibration.**

It passes the primary promotion criteria:
- total-goal likelihood improves vs both v1 and naive;
- O2.5 Brier improves vs both v1 and naive;
- Asian-handicap metrics remain unchanged.

Total-goal MAE remains 0.38% worse than the naive league baseline, so that remains an explicit v1.2 target.
