# GAH v2.4 — Automated Audit Protocol

## Purpose

v2.4 reconciles prospective market observations, execution decisions, actual entries,
closing-market information and settlement into one repeatable audit.

It must keep separate:
- forecast/model evidence;
- execution evidence;
- operational followability.

## Minimum row fields

Each audited observation should contain:
- snapshot_id;
- match_id;
- competition;
- kickoff;
- side;
- captured_line;
- captured_odds;
- market_probability;
- gah_probability;
- execution_state;
- entry_type;
- entered;
- closing_line;
- closing_market_probability;
- settlement;
- net_return.

## Derived metrics

- GAH edge = gah_probability - market_probability
- realized ROI over entered bets
- same-line probability CLV only when closing_line == captured_line
- line move otherwise
- entry conversion = entered / eligible observations
- protected-entry share
- decay-entry share
- pass/wait/no-entry counts
- win/push/loss settlement mix

## Diagnostic buckets

v2.4 does not claim a single losing bet proves a forecast failure.

Rows are placed into descriptive buckets:
- NOT_ENTERED
- ENTERED_WIN
- ENTERED_PUSH
- ENTERED_LOSS
- UNSETTLED

Forecast and execution summaries are reported separately.

## Review discipline

Do not tune thresholds from one audit.
Do not remove losing leagues or periods after seeing results.
Do not mix unavailable same-line CLV with line-move observations.
