# GAH v2.0 Stage 3 — Fresh Validation Decision Rule

This decision rule is written before the fresh 2016/17-2020/21 validation result is inspected.

## Frozen candidate

The Stage-3 reliability specification is fixed in
`benchmarks/v20_stage3_exploratory.md`.

No model feature, regularization setting, refit cadence, EV gate, residual-edge
gate, ranking rule, or board-size rule may be changed in response to the fresh
validation result.

## Minimum evidence required to retain Stage 3 as a v2.0 candidate

All of the following must hold on the fresh five-season block:

1. At least **30 selected bets** overall.
2. Overall realized opening-price ROI is **positive**.
3. Overall mean probability CLV is **positive**.
4. Overall positive-CLV rate is **above 50%**.
5. No single competition/market cell accounts for **more than 60%** of all
   selections.

These are screening criteria, not proof of a profitable production strategy.

## Interpretation

- If any criterion fails: do not promote the reliability selector. Archive it
  as exploratory evidence and continue v2.0 with a different hypothesis.
- If all criteria pass: retain the frozen selector as a **candidate** and move
  to a stricter temporal/stability audit before any production use.
- Even if all criteria pass, the result does not authorize automatic betting,
  staking, or replacement of the current production forecast.

The purpose of this rule is to prevent post-result threshold selection or
narrative fitting.
