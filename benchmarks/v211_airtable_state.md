# GAH v2.11 — durable Airtable state loader

Status: operational persistence only. No prediction, ranking, threshold, or
selection logic is changed.

v2.8 made writes deterministic; v2.11 completes the other half of that
contract by reconstructing the current operational state from Airtable after
a process/chat interruption.

## Capabilities

- paginated reads from the canonical \`Prospective AH\` table;
- paginated reads from append-only \`Lifecycle Events\`;
- deterministic prospective snapshot DataFrame reconstruction;
- duplicate Snapshot ID rejection;
- duplicate lifecycle Event ID rejection;
- lifecycle event chronological sorting and v2.6 transition validation;
- per-match DecisionLedger reconstruction;
- deterministic v2.11 JSON state export/import for handoff or debugging;
- counts for snapshots, matches, lifecycle events, pending and settled rows.

The loader intentionally does not infer missing transitions or repair data.
Invalid persisted state fails loudly so v2.10 quality control and the operator
can investigate it.

Runtime Airtable reads use \`AIRTABLE_TOKEN\`; credentials are never persisted
in the export.
