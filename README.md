# Football GAH v1

A code-first baseline for forecasting:

- **Goal totals** (Asian O/U lines)
- **Asian handicap** (home/away, including quarter lines)

The model first predicts a **full score distribution** using a time-decayed Dixon-Coles model. All market probabilities are derived from that same score matrix.

## Why this exists

GAH is deliberately separate from text-heavy analyst workflows.

The prediction core is deterministic:

> same data + same model version = same probability output

LLM/web research can be added later as explicit structured features, but it does not replace the statistical engine.

## v1 architecture

```text
Historical completed matches
        ↓
Time-decayed attack / defence strengths
        ↓
Dixon-Coles expected goals
   λ_home       λ_away
        ↓
Home × away score matrix
        ↓
┌───────────────────┬─────────────────────┐
│ Asian totals      │ Asian handicap      │
│ 2.0 / 2.25 / 2.5 │ 0 / ±.25 / ±.5 ... │
└───────────────────┴─────────────────────┘
        ↓
Settlement probabilities + fair odds
```

## Data schema

Required columns:

```csv
match_date,home_team,away_team,home_goals,away_goals
```

Optional:

```csv
competition
```

Use one competition at a time for the first baseline. Pooling unrelated leagues without a competition-strength layer can distort team strengths.

## Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Predict a match

```bash
python predict.py matches.csv --home "Arsenal" --away "Chelsea"
```

Optional competition filter:

```bash
python predict.py matches.csv \
  --competition "Premier League" \
  --home "Arsenal" \
  --away "Chelsea"
```

The output includes expected home goals, expected away goals, expected total, expected goal margin, and fair prices for standard totals/AH lines.

## Walk-forward backtest

```bash
python backtest_cli.py matches.csv \
  --competition "Premier League" \
  --min-train 250 \
  --refit-every 50 \
  --out epl_backtest.csv
```

The expanding window ensures every forecast is generated only from matches that occurred earlier.

Initial metrics:

- exact-score negative log likelihood
- total-goal absolute error
- goal-margin absolute error

Market calibration and ROI/CLV evaluation should be layered on after a bookmaker-odds dataset is connected.

## Important modelling rules

1. Do not use random train/test splits for time-series match data.
2. Do not feed post-kickoff information into a pre-match feature.
3. Keep bookmaker inputs out of the **pure** model.
4. Compare pure predictions with a separate market-aware model later.
5. Add features only when walk-forward results improve.
6. Keep Asian quarter-line settlement exact; never convert 2.25/2.75 or ±0.25/±0.75 into binary labels.

## Planned progression

### GAH v1
- Time-decayed Dixon-Coles
- Score probability matrix
- Asian totals pricing
- Asian handicap pricing
- Walk-forward backtest

### GAH v1.1
- Better calibration diagnostics
- Probability reliability plots
- Competition-specific home advantage
- Tail/overdispersion testing

### GAH v2
- xG
- shots / shots on target
- Elo/team-strength prior
- CatBoost residual correction
- lineup/absence structured inputs

### GAH v3
- bookmaker opening/closing lines
- de-vigging
- pure-vs-market comparison
- CLV / expected-value testing
