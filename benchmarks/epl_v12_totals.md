# EPL GAH v1.2 Totals Experiment

GAH v1.2 investigated two distribution changes and one point-forecast correction while keeping the Asian-handicap model frozen.

## Five-season walk-forward protocol

- EPL seasons: 2021-22 through 2025-26
- Initial training window: 380 chronological matches
- 1,513 evaluated predictions
- Dixon-Coles refit every 20 evaluated matches or when a new team requires it
- All totals blend weights learned only from prior out-of-sample predictions
- Asian handicap remains pure Dixon-Coles

## Distribution experiments

| Metric | v1.1 | Regime-Poisson | Regime-NB | RP vs v1.1 | NB vs v1.1 |
|---|---:|---:|---:|---:|---:|
| Total MAE using mean ↓ | 1.3190 | 1.3213 | 1.3213 | -0.18% | -0.17% |
| Total NLL ↓ | 1.9027 | 1.9007 | 1.9011 | +0.10% | +0.08% |
| O2.5 Brier ↓ | 0.2425 | 0.2423 | 0.2424 | +0.07% | +0.06% |

The scoring-regime variants marginally improve probability metrics but fail the promotion rule because they worsen total MAE.

Negative-binomial overdispersion is not useful in this EPL sample. The weighted scoring environment has mean total **2.960** and variance **2.786**, so the data are not globally overdispersed relative to Poisson.

## Promoted point-forecast correction

Absolute-error loss is minimized by the predictive median rather than the predictive mean.

| Point forecast | Total MAE |
|---|---:|
| v1.1 blended predictive mean | 1.3190 |
| **v1.2 blended predictive median** | **1.2987** |
| Old naive league-average mean benchmark | 1.3140 |

The predictive median improves MAE by **1.53% versus v1.1** and is approximately **1.16% better than the old naive benchmark**.

Because the underlying v1.1 probability distribution is unchanged, its validated probability metrics are preserved:

- Total NLL: **1.9027**
- O2.5 Brier: **0.2425**
- Goal-margin MAE: **1.3465**
- Home -0.5 Brier: **0.2157**

## Seasonal median MAE

| Season | v1.1 mean MAE | v1.2 median MAE |
|---|---:|---:|
| 2022-23 | 1.4064 | **1.3687** |
| 2023-24 | 1.3458 | **1.3360** |
| 2024-25 | 1.2900 | **1.2665** |
| 2025-26 | 1.2342 | **1.2243** |

The median improvement appears in every evaluated season.

## Operational correction

During v1.2 development an implementation gap was identified: the prediction CLI was still pricing totals directly from the raw Dixon-Coles matrix even though v1.1's online totals blend had been promoted by benchmark.

v1.2 fixes this. The prediction path now:

1. fits the Dixon-Coles team model;
2. derives a leakage-safe next-match totals blend weight from historical OOS predictions;
3. blends the Dixon-Coles score distribution with the historical league scoring distribution for totals only;
4. prices O/U markets from the blended distribution;
5. reports the predictive median as the MAE-optimal point total;
6. continues pricing Asian handicap from the unmodified Dixon-Coles score distribution.

## Decision

**Promote the predictive-median correction and production wiring as GAH v1.2.**

Do **not** promote the recency-regime Poisson or negative-binomial distribution changes.
