# EPL GAH v1.3 Full Asian-Line Calibration

GAH v1.3 validates the promoted v1.2 model across the full standard Asian totals and handicap line grids, then applies one leakage-safe totals probability tilt.

## Protocol

- EPL 2021-22 through 2025-26
- 1,513 out-of-sample match predictions
- 11 totals lines from 1.50 through 4.00
- 17 home Asian-handicap lines from -2.00 through +2.00
- Quarter lines scored as full win / half win / push / half loss / full loss
- Primary line metric: multiclass settlement Brier score
- Totals use the v1.2 OOS blend plus v1.3 total-goal tilt
- Asian handicap remains pure Dixon-Coles

## Full-line validation before tilt

| Market | Model Brier | Naive Brier | Improvement |
|---|---:|---:|---:|
| Totals | 0.5399 | 0.5424 | +0.47% |
| Asian handicap | 0.4357 | 0.4742 | +8.12% |

- Totals lines worse than naive: **1/11**
- AH lines worse than naive: **0/17**
- Home -0.50 improvement vs naive: **+12.67%**
- Home +0.50 improvement vs naive: **+10.76%**
- Home -1.50 improvement vs naive: **+9.43%**

The totals grid showed a consistent negative settlement-value bias, strongest on lower lines.

## Promoted totals tilt

One exponential tilt is learned from prior out-of-sample total-goal distributions:

`p'(h,a) ∝ p(h,a) × exp(beta × (h+a))`

This moves the entire totals distribution coherently rather than hand-adjusting individual lines.

| Metric | v1.2 | v1.3 tilt | Improvement |
|---|---:|---:|---:|
| Multi-line settlement Brier | 0.5399 | **0.5385** | **+0.26%** |
| Exact-total NLL | 1.9027 | **1.9009** | **+0.09%** |
| Median point MAE | **1.2987** | **1.2987** | unchanged |

Additional checks:

- Mean learned beta: **+0.0138**
- Median learned beta: **+0.0027**
- Lines with worse Brier after tilt: **0/11**
- Naive multi-line totals Brier: **0.5424**

After tilt, every tested totals line beats the naive comparator.

## Selected total lines after tilt

| Over line | Tilt Brier | Improvement vs v1.2 | Improvement vs naive |
|---:|---:|---:|---:|
| 1.50 | 0.3139 | +0.47% | +0.27% |
| 2.25 | 0.5765 | +0.19% | +0.66% |
| 2.50 | 0.4839 | +0.24% | +1.23% |
| 2.75 | 0.6422 | +0.25% | +0.78% |
| 3.50 | 0.4486 | +0.49% | +1.03% |
| 4.00 | 0.5104 | +0.23% | +0.51% |

## Decision

**Promote GAH v1.3.**

Production totals path:
1. Dixon-Coles score distribution.
2. Leakage-safe OOS blend with league baseline.
3. Leakage-safe exponential total-goal tilt.
4. Asian O/U pricing from calibrated totals matrix.
5. Predictive median as the MAE-optimal total point.

Asian handicap remains pure Dixon-Coles.
