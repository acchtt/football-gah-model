# GAH v1.4 Cross-Competition Validation

GAH v1.4 is a validation and data-coverage release. The production prediction
logic remains the promoted GAH v1.3 architecture.

## Scope

Five completed seasons were evaluated independently for:

- English Premier League
- German Bundesliga
- Spanish La Liga
- Italian Serie A
- French Ligue 1

Each league used the same leakage-safe protocol:

- initial 380 chronological matches before evaluation;
- expanding walk-forward training;
- Dixon-Coles refit every 20 evaluated matches or when a new team required it;
- 11 Asian totals lines from 1.50 through 4.00;
- 17 home Asian-handicap lines from -2.00 through +2.00;
- five-outcome settlement Brier as the primary Asian-line metric;
- totals use the promoted v1.3 OOS blend + total-goal tilt;
- Asian handicap remains pure Dixon-Coles.

Total out-of-sample predictions across the five competitions: **6,951**.

## Cross-competition results

| Competition | OOS N | Totals Brier | Naive | Totals vs naive | Totals lines better | AH Brier | AH naive | AH vs naive | AH lines better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EPL | 1,513 | 0.5385 | 0.5424 | **+0.73%** | **11/11** | 0.4357 | 0.4742 | **+8.12%** | **17/17** |
| Bundesliga | 1,145 | 0.5357 | 0.5382 | **+0.46%** | **10/11** | 0.4384 | 0.4808 | **+8.82%** | **17/17** |
| La Liga | 1,504 | 0.5220 | 0.5298 | **+1.48%** | **11/11** | 0.4090 | 0.4453 | **+8.16%** | **17/17** |
| Serie A | 1,503 | 0.5290 | 0.5300 | **+0.20%** | **7/11** | 0.4085 | 0.4524 | **+9.71%** | **17/17** |
| Ligue 1 | 1,286 | 0.5447 | 0.5421 | **-0.47%** | **0/11** | 0.4440 | 0.4747 | **+6.47%** | **17/17** |

## Point metrics and calibration parameters

| Competition | Total NLL | Median total MAE | Margin MAE | Mean DC totals weight | Mean tilt beta |
|---|---:|---:|---:|---:|---:|
| EPL | 1.9009 | 1.2987 | 1.3465 | 0.431 | +0.0138 |
| Bundesliga | 1.9587 | 1.3825 | 1.4339 | 0.457 | +0.0113 |
| La Liga | 1.8379 | 1.2394 | 1.1390 | 0.597 | +0.0247 |
| Serie A | 1.8206 | 1.2136 | 1.1659 | 0.377 | -0.0472 |
| Ligue 1 | 1.9186 | 1.3468 | 1.3385 | 0.354 | +0.0026 |

The variation in totals weights and tilt beta is itself evidence that totals
behavior is competition-sensitive.

## Asian handicap conclusion

The AH result is the strongest v1.4 finding.

The pure Dixon-Coles handicap engine beat the naive comparator on **all 17
tested lines in all five competitions**: 85/85 league-line comparisons.

Aggregate AH improvement by league ranged from **+6.47% to +9.71%**.

No new AH calibration layer is justified by this test.

## Totals conclusion

Totals do not generalize uniformly.

- **EPL:** validated; all 11 lines beat naive.
- **La Liga:** strongest totals validation; all 11 lines beat naive.
- **Bundesliga:** broadly validated; 10/11 lines beat naive.
- **Serie A:** marginal/line-dependent; only 7/11 lines beat naive.
- **Ligue 1:** not validated; 0/11 lines beat naive and aggregate Brier is 0.47% worse.

Therefore v1.4 does **not** claim universal totals superiority.

## Rejected safety-blend experiment

A second leakage-safe layer was tested that mixed v1.3 totals with the naive
league distribution using a Brier-optimal historical weight.

It failed the promotion test:

| Competition | v1.3 Brier | Safety Brier | Change |
|---|---:|---:|---:|
| EPL | 0.5385 | 0.5383 | +0.04% |
| Bundesliga | 0.5357 | 0.5364 | -0.12% |
| La Liga | 0.5220 | 0.5231 | -0.21% |
| Serie A | 0.5290 | 0.5293 | -0.06% |
| Ligue 1 | 0.5447 | 0.5462 | -0.28% |

The safety blend slightly helped EPL but degraded every other league, including
Ligue 1. It is rejected and is not part of production GAH.

## Decision

**Promote GAH v1.4 as a validation/coverage release with unchanged v1.3
prediction logic.**

Production interpretation:

- Asian handicap: validated across all five tested major leagues.
- Totals: use competition-specific evidence; do not assume EPL performance
  transfers automatically.
- Ligue 1 totals should remain experimental until a future totals architecture
  beats the naive comparator out of sample.
