# GAH v1.9 market-aware probabilities — final report

## Decision

**Reject probability blending as a production forecast.**

Keep the market ingestion, de-vigging and evaluation infrastructure as the foundation for v2.0 selection.

GAH Pure remains unchanged and independently reproducible.

## Five-league OOS results

Opening market averages are the only bookmaker input to the tested actionable blend.
Closing market averages are reference only.

| Competition | Market | GAH Pure | Opening market | GAH + market | Mean market weight |
|---|---|---:|---:|---:|---:|
| EPL | AH | 0.2610 | 0.2505 | 0.2547 | 0.834 |
| EPL | O2.5 | 0.2421 | 0.2389 | 0.2400 | 0.802 |
| Bundesliga | AH | 0.2664 | 0.2498 | 0.2516 | 0.910 |
| Bundesliga | O2.5 | 0.2336 | 0.2268 | 0.2279 | 0.912 |
| La Liga | AH | 0.2604 | 0.2494 | 0.2523 | 0.912 |
| La Liga | O2.5 | 0.2454 | 0.2376 | 0.2396 | 0.928 |
| Serie A | AH | 0.2594 | 0.2493 | 0.2500 | 0.911 |
| Serie A | O2.5 | 0.2494 | 0.2458 | 0.2463 | 0.806 |
| Ligue 1 | AH | 0.2616 | 0.2496 | 0.2512 | 0.906 |
| Ligue 1 | O2.5 | 0.2499 | 0.2415 | 0.2432 | 0.908 |

Across all 10 league-market comparisons:
- the opening market beats GAH Pure;
- the OOS blend improves on GAH Pure;
- the opening market still beats the blend;
- fitted weights are strongly market-dominant.

## Closing-market reference

Closing market averages are generally at least as strong as opening prices, reinforcing the conclusion that bookmaker consensus contains information not captured by GAH Pure.

Closing prices were never used as an actionable model input.

## Interpretation

The market should not be treated as another feature to average into the GAH probability and call the result a superior forecast. Once opening market information is available, GAH Pure contributes too little average probability accuracy to improve the market consensus.

That does **not** make GAH Pure useless.

The next question is conditional rather than global:

> When GAH and the market disagree materially, are some disagreement patterns systematically profitable or predictive?

That is the v2.0 selection problem.

## Production / architecture decision

- Production GAH Pure: unchanged v1.4 / validated v1.3 logic.
- No blended probability promoted.
- Preserve market data loader, de-vigging, settlement exposure and benchmark utilities.
- Use market consensus as the reference probability baseline for selection.
- Use GAH Pure as an independent disagreement/edge signal.

## Next: v2.0 selection

Initial predeclared selection dimensions:
1. signed model-vs-market probability edge;
2. absolute disagreement magnitude;
3. market type (AH vs totals);
4. line magnitude / line band;
5. pure-model confidence;
6. opening-to-closing movement for retrospective validation only;
7. minimum sample and board-size controls.

The first v2.0 benchmark must test whether selecting by predeclared disagreement bands improves outcome/CLV metrics OOS. It must not tune thresholds on the same evaluation slice.
