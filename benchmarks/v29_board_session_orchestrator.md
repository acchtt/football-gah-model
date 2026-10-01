# GAH v2.9 — board/session orchestrator

Status: operational orchestration only. It does not introduce a selector.

\`gah/session.py\` coordinates one board/session across the already-approved
v2.5–v2.8 components:

1. v2.5 board capacity using caller-supplied priority;
2. v2.1 prospective snapshot capture for selected board matches;
3. v2.2 execution decision;
4. optional v2.3 live repricing;
5. prospective outcome capture and, when actually entered, v2.6 settlement;
6. audit-ready rows for v2.4;
7. ordered v2.8 Airtable bridge operations.

## Selection boundary

The orchestrator never derives priority from GAH edge, model EV, historical
ROI, league, closing-line behavior, or any other profitability signal.
Priority is an explicit upstream input. Capacity-rejected matches remain in
the board result and receive no snapshot write from this session.

## Ordered writes

One match can mutate the same \`Prospective AH\` record several times:
snapshot -> execution -> live -> outcome. Therefore v2.9 intentionally emits
an ordered operation sequence and applies it sequentially. It does not collapse
those operations into one v2.8 de-duplication batch.

## Outcome semantics

All forward-only snapshots can eventually receive their prospective outcome.
Only actually-entered decisions receive lifecycle \`SETTLED\`. A non-entered
snapshot must have an explicit execution \`PASS\` before an outcome is attached
in the same session.

No production GAH forecast formula is modified.
