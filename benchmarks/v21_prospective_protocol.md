# GAH v2.1 — Prospective Entry / Market-Movement Protocol

## Why v2.1 exists

v2.0 did not establish a robust retrospective betting selector.

The repeatable signal was narrower:
GAH-vs-market disagreement sometimes anticipated subsequent AH market movement,
but that did not reliably translate into positive realized opening-price ROI.

v2.1 therefore tests **entry timing and information value prospectively**.
It is not another retrospective threshold search.

## Frozen scope

- Market: Asian Handicap only.
- Production forecast: existing GAH Pure v1.4 / v1.3 logic.
- Market baseline: de-vigged currently available AH odds.
- No closing odds may enter any pre-match decision.
- No league hardcoding.
- No retrospective exclusion of losing leagues or seasons.
- No automatic staking.

## Prospective capture

For every eligible monitored match with usable AH odds, record before kickoff:
- fixture key;
- snapshot timestamp;
- competition;
- home and away teams;
- AH side and line;
- decimal odds;
- de-vigged market probability;
- GAH Pure effective-win probability;
- signed GAH-minus-market edge;
- absolute edge;
- GAH-implied EV at the captured price.

No minimum edge is required for logging. We preserve the full eligible distribution.

## After the market closes / match settles

Append:
- closing AH line;
- closing de-vigged probability for the originally captured side where comparable;
- probability CLV = closing selected-side probability - captured selected-side probability;
- exact Asian settlement category;
- realized net return per unit at the captured price.

If the closing AH line changes, record the line change separately and do not pretend
same-line probability CLV exists.

## Primary prospective questions

1. Does signed GAH disagreement predict the direction of later AH market movement?
2. Is larger absolute disagreement associated with larger positive CLV without
   post-hoc league filters?
3. Does taking the captured price outperform waiting until close for the same
   direction/line where comparison is possible?
4. Does any information-value signal remain stable enough to support an entry
   rule before we revisit bet selection?

## Review cadence

Do not alter the rule after individual wins or losses.

First formal review only after:
- at least 100 settled AH snapshots, and
- at least 30 calendar days of prospective collection.

Until then, results are descriptive only.

## Production guardrail

GAH v1.4 / v1.3 remains production.
v2.1 observations do not replace or modify production forecasts.
