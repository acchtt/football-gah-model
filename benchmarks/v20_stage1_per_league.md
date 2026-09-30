# GAH v2.0 selection — stage 1 per-league screen

## Status

No selection rule is promoted from the first per-league screen.

The tested rule was deliberately simple and predeclared:
- select the side favored by GAH relative to the de-vigged opening market;
- rank by absolute probability disagreement;
- fixed edge bands: <2pp, 2-4pp, 4-6pp, 6-10pp, 10pp+;
- at most one market per match;
- per-league boards capped at 8 matches per day.

Development = 2021/22-2023/24.
Holdout = 2024/25-2025/26.

## Main conclusion

**Raw disagreement magnitude is not a robust realized-return selector.**

At the 10pp+ holdout threshold, capped-board ROI by league was:
- EPL: +9.2%
- Bundesliga: -10.2%
- La Liga: -3.3%
- Serie A: -2.3%
- Ligue 1: -17.2%

The cross-league inconsistency is too large to promote a simple edge threshold.

## Information-value observation

Large AH disagreements show a different pattern in the holdout: the closing market tends to move in the GAH-selected direction.

10pp+ AH probability CLV:
- EPL: +0.0092
- Bundesliga: +0.0074
- La Liga: +0.0043
- Serie A: +0.0105
- Ligue 1: +0.0093

Weighted across the five leagues, the 10pp+ AH group has approximately:
- 877 observations;
- mean realized ROI: -4.4%;
- mean probability CLV: +0.0084.

This is evidence of information about subsequent market movement, **not** evidence of a profitable betting rule.

For 10pp+ totals, holdout evidence is materially weaker:
- 257 observations;
- weighted ROI about -7.5%;
- weighted probability CLV about +0.0011.

## Guardrail

Do not create a new betting threshold from these holdout results.

The next valid test is the already-predeclared global cross-league board: at most 8 total matches per date across all five leagues. If that also fails realized-return robustness, v2.0 should shift from raw disagreement ranking toward a separately specified reliability/market-movement stage, with any new parameters learned only from development data and evaluated on a new untouched slice or rolling OOS procedure.
