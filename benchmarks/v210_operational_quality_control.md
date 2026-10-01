# GAH v2.10 — forward monitoring and operational quality control

Status: operational QA only. This layer does not select matches, alter
probabilities, recommend bets, change thresholds, or modify production GAH.

\`gah/quality_control.py\` produces a concise machine-checkable health report
with ERROR / WARNING / INFO issues.

## Checks

- missing required prospective snapshot fields;
- duplicate Snapshot IDs;
- duplicate Settlement IDs;
- entered decisions missing entry time/line/odds/type;
- PASS records that conflict with entry metadata;
- settled snapshots missing settlement or return;
- missing closing line/probability (reported as coverage warnings, never zero);
- stale pending prospective rows;
- invalid lifecycle transitions;
- exact duplicate lifecycle events;
- duplicate ledgers;
- lifecycle progression without a corresponding snapshot;
- lifecycle SETTLED without a settled prospective outcome;
- stale BOARD_SELECTED records missing capture;
- v2.5 total/concurrent board-capacity violations;
- frozen v2.1 readiness progress through the existing v2.7 gate.

The reader accepts both the repository's snake_case review columns and the
live Airtable field names used by \`Prospective AH\`.

## Health status

- \`ERROR\`: at least one integrity/correctness error;
- \`WARN\`: no errors, but one or more operational warnings;
- \`OK\`: no errors or warnings.

Readiness \`WAIT\` is informational, not a warning. The formal gate remains
unchanged: at least 100 settled AH snapshots AND at least 30 calendar days.

After this version, the roadmap pauses new predictive/selection hypotheses
and prioritizes real forward collection and workflow reliability.
