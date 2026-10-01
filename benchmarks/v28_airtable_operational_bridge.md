# GAH v2.8 — Airtable operational bridge

Status: operational infrastructure only. No forecasting or selection logic is changed.

## Datastore

- Base: \`Football GAH\` — \`app2wBjpvj69bCoMS\`
- Prospective snapshots/outcomes: \`Prospective AH\` — \`tblk9kLKRqrnBOQ2k\`
- Append-only decision events: \`Lifecycle Events\` — \`tblF4rvY4jQPgb3an\`

\`Prospective AH\` remains forward-only. v2.8 does not backfill historical selections.

## Guarantees

- deterministic canonical JSON serialization;
- deterministic Event IDs and Settlement IDs;
- snapshot/event linkage through \`Match ID\` and \`Snapshot ID\`;
- append-only lifecycle history;
- duplicate identical writes collapse to no-ops;
- conflicting duplicate snapshots/events raise instead of silently overwriting;
- settlement and entry facts are protected from conflicting rewrites;
- \`UPDATE_EXISTING\` refuses to create a missing prospective snapshot;
- Airtable credentials are runtime-only via \`AIRTABLE_TOKEN\`.

## Prospective AH operational fields added

- Bridge Schema Version
- Execution State
- Entry Type
- Entry At
- Entry Line
- Entry Odds
- Settlement ID
- Last Lifecycle Event ID

## Lifecycle Events fields

- Event ID
- Match ID
- Snapshot ID
- State
- Event At
- Reason
- Source
- Payload JSON
- Schema Version

The bridge is intentionally operational. It does not rank matches, alter GAH probabilities, or authorize betting changes.
