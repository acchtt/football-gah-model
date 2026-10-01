# GAH v2.5 — Board Capacity / Followability Protocol

## Purpose

v2.5 prevents operational overload after matches have already been approved for monitoring.

It is **not** a betting selector and must not rank candidates using GAH edge, model EV,
historical ROI, or post-hoc league profitability.

## Inputs

Each candidate contains:
- match_id;
- kickoff;
- analyst-assigned operational priority;
- monitoring window before kickoff;
- monitoring window after kickoff.

Priority is an explicit upstream decision. v2.5 does not infer it.

## Capacity controls

A board has:
- maximum total matches;
- maximum simultaneous monitored matches.

Candidates are considered deterministically by:
1. higher explicit priority;
2. earlier kickoff;
3. match_id as a stable tie-break.

A candidate is accepted only if adding its monitoring interval does not violate
the simultaneous-monitoring cap and total board capacity has not been reached.

## Outputs

For every candidate:
- SELECTED, or
- REJECTED_TOTAL_CAPACITY, or
- REJECTED_CONCURRENCY.

The output must preserve rejected candidates so missed opportunities can be
audited later rather than disappearing from the record.

## Guardrails

- Do not use betting edge as a hidden priority score.
- Do not expand the board because several attractive matches appear at once.
- Do not retrospectively change priority after match results are known.
- Do not remove rejected candidates from audit history.
- Capacity is an operations constraint, not evidence that rejected matches were
  weaker predictions.

## Default operational shape

The code exposes capacity as configuration rather than hardcoding one permanent
number. Current workflows can use a small board such as 8 total matches with a
2-3 match simultaneous-monitoring cap.
