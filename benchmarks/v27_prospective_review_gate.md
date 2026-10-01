# GAH v2.7 — Prospective Review Gate

## Purpose

v2.7 enforces the frozen v2.1 rule that no formal prospective review is allowed
until both conditions are true:

- at least 100 settled AH snapshots;
- at least 30 calendar days of prospective collection.

The gate is descriptive infrastructure. It does not promote a betting rule,
retune GAH, or create selection thresholds.

## Expected normalized input

One row per prospective AH snapshot with:
- snapshot_id;
- captured_at;
- outcome_status;
- settlement;
- net_return;
- captured_line;
- closing_line;
- market_probability;
- closing_market_probability.

Additional columns are preserved but not required for readiness.

## Readiness outputs

- settled snapshot count;
- observation span in days;
- remaining snapshots to 100;
- remaining days to 30;
- READY or WAIT.

## Descriptive report after readiness

Once READY, report:
- settled snapshot count;
- mean captured-price return per unit;
- same-line CLV coverage;
- mean same-line probability CLV;
- positive same-line CLV rate;
- line-move coverage and mean line movement.

These statistics are observational. They do not authorize automatic betting,
staking, threshold changes, league exclusions, or production-model changes.

## Guardrails

1. Do not review early because the first results look strong.
2. Do not exclude losing observations after the gate opens.
3. Do not treat missing same-line CLV as zero.
4. Do not combine same-line price CLV with line movement as if they were the same metric.
5. Do not call the descriptive report proof of profitability.
