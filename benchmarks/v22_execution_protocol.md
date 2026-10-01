# GAH v2.2 — Execution / Decay Protocol

## Scope

v2.2 does **not** select matches.

Its input is an already-chosen AH direction plus an execution plan. Its job is to
turn changing pre-match lines/prices into one of four states:

- `WAIT` — preferred entry has not arrived and the plan is still live.
- `ENTER_PROTECTED` — protected line/price is available.
- `ENTER_DECAY` — a predeclared later fallback is available.
- `PASS` — no acceptable entry remains before expiry.

This keeps forecast quality, match selection and execution separate.

## Line convention

All AH lines are expressed from the selected side's perspective.

For the selected side, a numerically larger AH line is never worse:
- -0.50 is better than -0.75;
- +0.75 is better than +0.50.

Therefore an offered line meets a target when:

`current_line >= target_line`

Higher decimal odds are also better.

## Execution plan

A plan contains:

- selected side;
- protected target line;
- protected minimum odds;
- zero or more predeclared decay steps;
- expiry in minutes-to-kickoff.

Each decay step contains:
- an activation threshold in minutes-to-kickoff;
- the worst acceptable selected-side AH line from that point;
- the minimum acceptable odds.

Decay steps may only relax execution requirements as kickoff approaches.
They must be fixed before observing subsequent market movement.

## Guardrails

1. Never change a target because a goal, lineup announcement, or price move made
   the original target inconvenient.
2. Never reinterpret a missed protected line as a successful entry.
3. If the line is better but odds are below the active minimum, keep waiting.
4. If odds are sufficient but the line is worse than the active target, keep waiting.
5. Once the plan expires, return `PASS`; do not chase.
6. Log the line/odds actually available at every decision point.
7. v2.2 execution data can be studied prospectively, but no execution rule is
   promoted from isolated wins/losses.

## Relationship to v2.1

v2.1 captures the market information stream.
v2.2 consumes that stream for predeclared execution decisions.

Production GAH v1.4/v1.3 remains unchanged.
