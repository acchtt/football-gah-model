# GAH v1.8 XI / availability — final report

## Decision

**No production promotion.**

Production remains GAH v1.4 with validated v1.3 prediction logic.

Two predeclared confirmed-XI hypotheses were tested on full EPL 2015/16 and Serie A 2015/16 StatsBomb seasons.

### 1. Directional XI continuity

The current XI was compared with each team's prior 10 starting XIs. A higher recent-core coverage than the opponent produced a small OOS-fitted goal-margin tilt.

| Competition | AH Brier change | Margin MAE change | AH lines improved |
|---|---:|---:|---:|
| EPL 2015/16 | -0.43% | -0.40% | 0/17 |
| Serie A 2015/16 | -0.21% | -0.22% | 3/17 |

Disrupted-lineup subsets were also worse: EPL -0.85% AH Brier and Serie A -0.17%.

### 2. XI disruption as uncertainty

The second experiment did not move the expected goal margin. Confirmed-XI disruption only widened or narrowed the goal-margin distribution, with the dispersion scale fitted on prior OOS rows.

| Competition | AH Brier change | Exact-margin NLL change | AH lines improved |
|---|---:|---:|---:|
| EPL 2015/16 | +0.04% | -0.10% | 11/17 |
| Serie A 2015/16 | -0.18% | -0.14% | 0/17 |

The disrupted-lineup subsets were not improved:
- EPL: AH Brier -0.03%, margin NLL -0.43%
- Serie A: AH Brier -0.28%, margin NLL -0.27%

## Conclusion

Confirmed-XI continuity/disruption alone is not a robust enough structured signal to alter the production probability distribution. A changed XI does not encode player quality, role, tactical intent or replacement strength, and the tested transformations fail the cross-competition promotion guardrail.

No further threshold/window tuning is justified from this dataset.

## Next

Proceed to GAH v1.9: a separate market-aware layer using de-vigged bookmaker probabilities. GAH Pure remains independently reproducible and unchanged.
