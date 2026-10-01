# GAH v2.3 — Live Re-pricing Protocol

## Scope

v2.3 re-prices an already-approved AH direction after kickoff or after a
material pre/live-match update.

It does **not** reopen v2.0 match selection and it does not replace v2.2
execution discipline.

## Core model

The frozen baseline is intentionally simple:

1. Start from the production pre-match expected goals for home and away.
2. Condition on the observed score.
3. Scale the remaining expected goals by the fraction of regulation time left.
4. Apply only explicitly supplied rate multipliers for material events.
5. Build a conditional final-score distribution.
6. Re-use the existing Asian-handicap settlement/pricing engine.

No automatic red-card, substitution or lineup coefficient is hardcoded in v2.3.
If a material event is believed to alter scoring rates, the adjustment must be
explicit and accompanied by a reason/source.

## Inputs

Required:
- pre-match home xG;
- pre-match away xG;
- current minute (0-90 regulation);
- current home score;
- current away score;
- selected AH side and candidate live line.

Optional structured adjustment:
- home remaining-goal rate multiplier;
- away remaining-goal rate multiplier;
- material-event labels;
- adjustment reason/source.

Examples of material-event labels:
- `red_card_home`;
- `red_card_away`;
- `key_sub_home`;
- `key_sub_away`;
- `lineup_change`.

The labels themselves do not alter probabilities.

## Outputs

For each candidate AH line:
- exact Asian settlement probabilities;
- effective win probability;
- fair decimal odds;
- expected value at the quoted live price when supplied.

The output can feed v2.2 execution. v2.3 does not choose an EV threshold.

## Information discipline

A market move by itself does not alter GAH's football-state probability.
It changes the price/line at which the existing live probability is evaluated.

A model re-price is justified by:
- elapsed match time;
- score change;
- explicitly documented material event / rate adjustment.

Do not manufacture a football-state adjustment merely because the market moved
against the preferred entry.

## Regulation-time limitation

v2.3 currently models 0-90 minutes only. Extra time is out of scope.

## Validation status

This is operational infrastructure, not a promoted live betting model.
Prospective observations should be logged and audited before any calibrated
event multipliers are introduced.
