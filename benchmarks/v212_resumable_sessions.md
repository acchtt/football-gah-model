# GAH v2.12 — resumable incremental sessions

Status: operational continuity only. No predictive or selection logic changes.

v2.9 can execute a complete board/session in one process. v2.11 can reload the
persisted Airtable state. v2.12 combines those capabilities so an operator can
continue one existing snapshot after interruption without recreating it.

## Supported incremental actions

- execution re-check (WAIT / protected entry / decay entry / PASS);
- optional live AH reprice;
- prospective outcome recording;
- entered-decision lifecycle settlement;
- explicit AUDITED lifecycle completion.

Every update begins from the reconstructed v2.11 DecisionLedger and is checked
against the frozen v2.6 transition rules.

## Retry safety

A deterministic lifecycle event that already exists is recognized.

- If it is the current latest event and the snapshot link is missing, v2.12
  can re-emit only the Prospective AH update to repair a partial write.
- If it is already historical and newer lifecycle events exist, the retry is a
  no-op so an old execution/live event cannot regress the canonical snapshot.
- Exact prospective outcomes are idempotent through deterministic Settlement
  IDs.

This is a continuity layer, not a selector. It never chooses a match, side,
line, threshold, or priority.
