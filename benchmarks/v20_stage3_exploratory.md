# GAH v2.0 selection — stage 3 exploratory reliability result

## Status

**Promising, not promotable yet.**

The rolling market-residual reliability model produced:

| Phase | Bets | ROI | Mean residual edge | Mean predicted EV | Mean probability CLV | Positive CLV rate |
|---|---:|---:|---:|---:|---:|---:|
| 2021/22-2023/24 development | 62 | -1.0% | +2.62pp | +1.3% | +0.59pp | 61.8% |
| 2024/25-2025/26 evaluation | 15 | +22.2% | +2.03pp | +4.7% | +5.34pp | 88.9% |

The rule is substantially more selective than raw disagreement and shows much stronger closing-market movement.

## Important experimental-design limitation

The 2024/25-2025/26 outcomes were already inspected during v2.0 stages 1 and 2 before the stage-3 reliability specification was created. Therefore this period is no longer an untouched confirmatory holdout for stage 3.

The +22.2% ROI and +5.34pp CLV are treated as exploratory evidence only.

## Frozen stage-3 specification

Before any fresh validation, freeze:
- de-vigged opening market probability as the baseline logit offset;
- residual features: GAH edge, pure confidence, market type, AH line magnitude;
- L2 = 10;
- minimum 500 prior active-settlement rows;
- refit every 250 new historical candidate rows;
- candidate must have positive reliability residual edge;
- candidate must have positive reliability-implied opening-price EV;
- at most one market per match;
- at most 8 matches per date;
- rank by reliability-implied EV, then residual edge.

No parameter above may be changed in response to the fresh validation result.

## Fresh confirmation block

Use Football-Data seasons 2016/17 through 2020/21, which were not used in the v1.9/v2.0 selection screens.

This is temporally older and includes the COVID-era seasons, so it is not a perfect proxy for current football. Its purpose is narrower: determine whether the frozen reliability mechanism has independent signal outside the already-inspected 2021/22-2025/26 sample.
