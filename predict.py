from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gah.backtest import (
    _independent_poisson_matrix,
    fit_next_total_blend_weight,
)
from gah.markets import price_handicap, price_total
from gah.model import DixonColesModel
from gah.totals import absolute_error_optimal_point, matrix_to_total_pmf


TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
AH_LINES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def build_market_table(raw_matrix: np.ndarray, totals_matrix: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []

    # Promoted totals distribution: v1.1 online blend, with v1.2 median point output.
    for line in TOTAL_LINES:
        rows.append(price_total(totals_matrix, line, "over"))
        rows.append(price_total(totals_matrix, line, "under"))

    # Handicap remains the validated pure Dixon-Coles distribution.
    for line in AH_LINES:
        rows.append(price_handicap(raw_matrix, line, "home"))
        rows.append(price_handicap(raw_matrix, -line, "away"))

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Football GAH v1.2 prediction")
    ap.add_argument("csv", help="Historical completed matches CSV")
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument(
        "--competition",
        default=None,
        help="Optional competition value to filter before fitting",
    )
    ap.add_argument("--half-life", type=float, default=180.0)
    ap.add_argument(
        "--total-weight",
        type=float,
        default=None,
        help="Override totals blend weight. Default: leakage-safe auto calibration.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.competition is not None:
        if "competition" not in df.columns:
            raise SystemExit("CSV has no 'competition' column.")
        df = df[df["competition"] == args.competition].copy()

    model = DixonColesModel(half_life_days=args.half_life, max_goals=10)
    summary = model.fit(df)
    pred = model.predict(args.home, args.away)
    raw_matrix = pred["score_matrix"]

    if args.total_weight is None:
        total_weight = fit_next_total_blend_weight(
            df,
            half_life_days=args.half_life,
            min_train_matches=380,
            refit_every=20,
            min_history=100,
            window=380,
        )
        weight_source = "auto OOS calibration"
    else:
        total_weight = float(np.clip(args.total_weight, 0.0, 1.0))
        weight_source = "manual override"

    league_home_rate = float(df["home_goals"].mean())
    league_away_rate = float(df["away_goals"].mean())
    league_matrix = _independent_poisson_matrix(
        league_home_rate,
        league_away_rate,
        max_goals=10,
    )

    totals_matrix = (
        total_weight * raw_matrix
        + (1.0 - total_weight) * league_matrix
    )
    totals_matrix = totals_matrix / totals_matrix.sum()

    total_pmf = matrix_to_total_pmf(totals_matrix)
    total_point = absolute_error_optimal_point(total_pmf)
    blended_mean = float(
        total_weight * pred["xg_total"]
        + (1.0 - total_weight) * (league_home_rate + league_away_rate)
    )

    print(f"\n{args.home} vs {args.away}")
    print("=" * (len(args.home) + len(args.away) + 4))
    print(f"DC xG home:          {pred['xg_home']:.3f}")
    print(f"DC xG away:          {pred['xg_away']:.3f}")
    print(f"DC xG total:         {pred['xg_total']:.3f}")
    print(f"Expected margin:     {pred['expected_margin']:+.3f}")
    print(f"Totals blend weight: {total_weight:.3f} ({weight_source})")
    print(f"Totals blended mean: {blended_mean:.3f}")
    print(f"Totals point (MAE):  {total_point:.2f}")
    print(f"rho:                 {summary.rho:+.4f}")
    print()

    markets = build_market_table(raw_matrix, totals_matrix)
    display_cols = [
        "market",
        "side",
        "line",
        "full_win",
        "half_win",
        "push",
        "half_loss",
        "full_loss",
        "fair_decimal_odds",
    ]
    print(markets[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
