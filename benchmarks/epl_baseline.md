# EPL Baseline Benchmark

This is the first code-first benchmark for Football GAH.

## Protocol

- Source: OpenFootball public-domain EPL results
- Seasons loaded: 2021-22, 2022-23, 2023-24, 2024-25, 2025-26
- First 380 chronological matches used before evaluation begins
- Expanding walk-forward training only; no future matches enter a prediction
- Dixon-Coles model refitted every 20 evaluated matches and immediately when a newly observed team requires it
- Exponential time decay half-life: 180 days
- Fixed market checks: Over 2.5 and Home -0.5
- Naive comparator: training-window league-average home/away goal rates

## Overall

| Metric | GAH v1 | Naive | Improvement |
|---|---:|---:|---:|
| Exact-score NLL ↓ | 3.0288 | 3.0960 | +2.17% |
| Total-goals MAE ↓ | 1.3448 | 1.3140 | -2.35% |
| Goal-margin MAE ↓ | 1.3465 | 1.4553 | +7.47% |
| Over 2.5 Brier ↓ | 0.2454 | 0.2449 | -0.16% |
| Home -0.5 Brier ↓ | 0.2157 | 0.2470 | +12.67% |

Evaluated predictions: **1513**

## By season

| Season | N | Score NLL | Total MAE | Margin MAE | O2.5 Brier | Home -0.5 Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2022-23 | 377 | 3.1001 | 1.4195 | 1.4209 | 0.2427 | 0.2297 |
| 2023-24 | 378 | 3.1037 | 1.3616 | 1.3966 | 0.2384 | 0.2034 |
| 2024-25 | 379 | 2.9869 | 1.3235 | 1.3371 | 0.2440 | 0.2068 |
| 2025-26 | 379 | 2.9250 | 1.2753 | 1.2319 | 0.2563 | 0.2229 |

## Interpretation

This report is a baseline, not a betting strategy. No bookmaker prices, xG, lineups, injuries, shots, or ML residual features are used yet.
A feature should only be kept in later versions if it improves genuinely out-of-sample metrics under the same walk-forward discipline.

