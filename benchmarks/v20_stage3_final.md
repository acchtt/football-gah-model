# GAH v2.0 Stage 3 — Fresh Validation Final Decision

## Frozen fresh-validation result

Football-Data 2016/17-2020/21, frozen Stage-3 reliability specification:

| Metric | Result | Required | Pass? |
|---|---:|---:|:---:|
| Selected bets | 403 | >= 30 | YES |
| Realized ROI | -0.1% | > 0% | **NO** |
| Mean probability CLV | +0.18pp | > 0 | YES |
| Positive-CLV rate | 52.7% | > 50% | YES |
| Largest competition/market share | EPL AH 110/403 = 27.3% | <= 60% | YES |

**Decision: Stage 3 fails the predeclared promotion rule.**

The rule required all five criteria to pass. Four passed; realized ROI did not.

## Fresh block detail

Season split:
- 2019/20: 382 bets, +1.4% ROI, +0.25pp CLV, 54.0% positive CLV.
- 2020/21: 21 bets, -27.3% ROI, -1.94pp CLV, 14.3% positive CLV.

Competition / AH split:
- Bundesliga: 95 bets, -8.1% ROI, +0.37pp CLV.
- EPL: 110 bets, -3.5% ROI, +0.04pp CLV.
- La Liga: 65 bets, +7.7% ROI, -0.06pp CLV.
- Ligue 1: 44 bets, -16.8% ROI, +0.42pp CLV.
- Serie A: 89 bets, +15.1% ROI, +0.29pp CLV.

All selected rows were AH in this fresh validation.

## Interpretation

The reliability layer appears to extract some information about subsequent market movement, but it does not demonstrate a robust positive realized return under the frozen fresh test.

Do not:
- relax the positive-ROI criterion;
- remove 2020/21 after seeing its poor result;
- hardcode profitable leagues;
- tune regularization, refit cadence or EV gates against this fresh block;
- promote the selector to production.

## v2.0 conclusion so far

Three increasingly selective hypotheses failed to establish a robust betting selector:
1. raw per-league disagreement;
2. raw global max-8 disagreement;
3. rolling market-residual reliability.

The recurring finding is narrower:
**GAH disagreement can contain information about later market movement, especially AH, without reliably converting that information into opening-price realized profit.**

Production remains GAH v1.4 / v1.3 unchanged.
