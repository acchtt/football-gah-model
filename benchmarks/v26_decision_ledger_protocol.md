# GAH v2.6 — Decision Ledger / Lifecycle Protocol

## Purpose

v2.6 connects the existing operational layers without changing prediction logic.

It records the lifecycle of an already-identified match from board admission through
prospective capture, execution, optional live repricing, settlement and audit.

## Lifecycle states

- IDENTIFIED
- BOARD_SELECTED
- SNAPSHOT_CAPTURED
- WAITING_ENTRY
- ENTERED_PROTECTED
- ENTERED_DECAY
- PASSED
- LIVE_REPRICED
- SETTLED
- AUDITED

## Core rules

1. Lifecycle events are append-only.
2. Event timestamps must be timezone-aware.
3. A match may not be settled before it has either been entered or explicitly passed.
4. A passed match may still be audited, but it may not later become an entry.
5. Live repricing is only valid after a snapshot exists.
6. An audited match is terminal.
7. The ledger never chooses a match, line or threshold. It only records decisions produced upstream.
8. Every transition carries a reason/source note so later audits can reconstruct why it happened.

## Relationship to existing versions

- v2.1 supplies prospective snapshots.
- v2.2 supplies WAIT / ENTER_PROTECTED / ENTER_DECAY / PASS execution decisions.
- v2.3 supplies live re-pricing observations.
- v2.4 audits outcomes.
- v2.5 supplies board admission/rejection.
- v2.6 records the complete chain in one deterministic lifecycle.

Production GAH v1.4/v1.3 remains unchanged.
