# GAH v1.8 confirmed-XI continuity — directional screen

## Decision

**Directional continuity adjustment rejected.**

Production remains GAH v1.4 / validated v1.3 logic.

The first v1.8 hypothesis was that a team whose confirmed XI contained more of its recent core starters than its opponent should receive a small goal-margin adjustment.

The adjustment was leakage-safe:
- confirmed XI compared with the team's prior 10 starting XIs;
- minimum five prior lineups before coverage was considered valid;
- adjustment scale fitted only on earlier OOS rows;
- Dixon-Coles base distribution otherwise unchanged.

## Full-season screen

| Competition | Evaluable matches | AH Brier change | Margin MAE change | AH lines improved |
|---|---:|---:|---:|---:|
| EPL 2015/16 | 299 | -0.43% | -0.40% | 0/17 |
| Serie A 2015/16 | 300 | -0.21% | -0.22% | 3/17 |

## Disrupted-lineup subset

Disrupted means at least one confirmed XI contained less than 80% of its recent core-starter mass.

| Competition | Disrupted matches | AH Brier change |
|---|---:|---:|
| EPL 2015/16 | 142 | -0.85% |
| Serie A 2015/16 | 174 | -0.17% |

The effect is not rescued in the matches where lineup continuity should matter most.

## Interpretation

Recent-starter continuity is not a robust directional strength signal on top of the existing Dixon-Coles estimate. A changed XI does not automatically imply a weaker XI: rotation can replace starters with comparable or stronger players, tactical changes are not inherently negative, and raw continuity ignores player quality.

This exact directional mechanism will not be tuned further.

## Remaining v1.8 question

One predeclared alternative remains: test lineup disruption as an **uncertainty** signal rather than a directional strength signal. That experiment will preserve the base expected goal margin and ask only whether disrupted lineups warrant a wider/narrower margin distribution.

If that also fails, v1.8 closes without production promotion.
