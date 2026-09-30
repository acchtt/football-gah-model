# GAH v2.0 selection — stage 2 global board

## Decision

**Reject absolute GAH-vs-market disagreement as the global board ranking rule.**

The predeclared global board combined EPL, Bundesliga, La Liga, Serie A and Ligue 1, allowed at most one market per match, and capped the slate at 8 total matches per date.

Development = 2021/22-2023/24.
Holdout = 2024/25-2025/26.

## Global capped-board results

| Phase | Min edge | Bets | ROI | Mean probability CLV | Positive CLV rate |
|---|---:|---:|---:|---:|---:|
| development | 0pp | 1946 | -6.3% | -0.18pp | 47.2% |
| development | 2pp | 1910 | -6.9% | -0.19pp | 47.1% |
| development | 4pp | 1822 | -6.7% | -0.20pp | 46.8% |
| development | 6pp | 1647 | -6.9% | -0.23pp | 46.8% |
| development | 10pp | 1162 | -7.4% | -0.14pp | 49.2% |
| holdout | 0pp | 1894 | -8.6% | +0.17pp | 49.6% |
| holdout | 2pp | 1867 | -8.9% | +0.18pp | 49.6% |
| holdout | 4pp | 1746 | -8.4% | +0.19pp | 49.5% |
| holdout | 6pp | 1577 | -8.8% | +0.24pp | 50.1% |
| holdout | 10pp | 960 | -6.5% | +0.51pp | 51.9% |

No fixed edge threshold produces a profitable or development-consistent board.

## Holdout 10pp+ composition

| Competition | Market | N | ROI | Probability CLV |
|---|---|---:|---:|---:|
| Bundesliga | AH | 136 | -12.0% | +0.58pp |
| Bundesliga | Total 2.5 | 21 | -29.2% | +0.85pp |
| EPL | AH | 180 | +2.1% | +0.56pp |
| EPL | Total 2.5 | 38 | +23.1% | +0.67pp |
| La Liga | AH | 135 | -0.1% | +0.17pp |
| La Liga | Total 2.5 | 43 | -3.0% | +0.32pp |
| Ligue 1 | AH | 149 | -16.7% | +1.00pp |
| Ligue 1 | Total 2.5 | 30 | -21.5% | -0.90pp |
| Serie A | AH | 202 | -10.5% | +0.75pp |
| Serie A | Total 2.5 | 26 | +4.2% | +0.04pp |

## Interpretation

Absolute disagreement contains some information about later market movement, especially for AH, but does not identify a robust profitable board by itself.

The next stage must therefore estimate whether a disagreement is reliable rather than assuming larger disagreement is better.

The reliability model is restricted to predeclared opening-time information:
- de-vigged market probability;
- GAH-vs-market edge;
- pure-model confidence;
- market type;
- AH line magnitude.

It must be trained only on prior rows in chronological order. No threshold will be selected from the already-inspected holdout.
