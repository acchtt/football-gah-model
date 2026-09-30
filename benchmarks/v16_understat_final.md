# GAH v1.6 Understat enrichment — final experiment report

## Decision

**Rejected for global production promotion.**

Production remains GAH v1.4 with validated v1.3 prediction logic.

## Experiments

### Raw xG matrix blend

| League | Multi-line Brier change vs v1.3 |
|---|---:|
| EPL | +0.06% |
| Bundesliga | +0.11% |
| La Liga | -0.20% |
| Serie A | -0.08% |
| Ligue 1 | -0.21% |

Raw xG contained independent information but did not generalise as a safe production blend.

### Non-penalty xG matrix blend

| League | Multi-line Brier change vs v1.3 |
|---|---:|
| EPL | +0.05% |
| Bundesliga | +0.09% |
| La Liga | -0.20% |
| Serie A | -0.18% |
| Ligue 1 | -0.13% |

Separating penalty xG did not solve the cross-competition instability.

### Tactical xG residual

Features:
- recent non-penalty xG for/against
- PPDA
- deep completions
- short/medium form windows
- leakage-safe OOS fitting
- capped match-specific total-goal tilt

| League | v1.3 Brier | Tactical Brier | Change | Lines improved |
|---|---:|---:|---:|---:|
| EPL | 0.5385 | 0.5399 | -0.27% | 0/11 |
| Bundesliga | 0.5356 | 0.5340 | +0.31% | 11/11 |
| La Liga | 0.5214 | 0.5222 | -0.14% | 0/11 |
| Serie A | 0.5293 | 0.5310 | -0.32% | 0/11 |
| Ligue 1 | 0.5438 | 0.5418 | +0.37% | 10/11 |

Additional tactical results:
- EPL: NLL -0.24%, median MAE +0.66%
- Bundesliga: NLL +0.04%, median MAE +0.06%
- La Liga: NLL -0.11%, median MAE +0.11%
- Serie A: NLL -0.07%, median MAE -1.41%
- Ligue 1: NLL +0.16%, median MAE +2.06%

## Interpretation

Understat features do contain useful extra signal, particularly in Bundesliga and Ligue 1, but the same residual architecture degrades three of five competitions and fails the cross-competition promotion guardrail.

No league-specific hard-coded deployment is promoted from this experiment. The observed competition dependence is retained as research evidence rather than converted into production logic after repeated validation exposure.

## Next

Proceed to GAH v1.7: leakage-safe Elo/team-strength priors, focused on promoted teams, early-season sparse data and structural team-strength changes. The Asian-handicap production core remains frozen unless v1.7 clears a materially higher validation bar.
