from __future__ import annotations

import argparse
import pandas as pd

from gah.model import DixonColesModel
from gah.markets import standard_market_table


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description="Football GAH v1 prediction")
    ap.add_argument("csv", help="Historical matches CSV")
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--competition", default=None,
                    help="Optional competition value to filter before fitting")
    ap.add_argument("--half-life", type=float, default=180.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.competition is not None:
        if "competition" not in df.columns:
            raise SystemExit("CSV has no 'competition' column.")
        df = df[df["competition"] == args.competition].copy()

    model = DixonColesModel(half_life_days=args.half_life)
    summary = model.fit(df)
    pred = model.predict(args.home, args.away)

    print(f"\n{args.home} vs {args.away}")
    print("=" * (len(args.home) + len(args.away) + 4))
    print(f"xG home:        {pred['xg_home']:.3f}")
    print(f"xG away:        {pred['xg_away']:.3f}")
    print(f"xG total:       {pred['xg_total']:.3f}")
    print(f"Expected margin:{pred['expected_margin']:+.3f}")
    print(f"rho:            {summary.rho:+.4f}")
    print()

    markets = pd.DataFrame(standard_market_table(pred["score_matrix"]))
    display_cols = [
        "market", "side", "line",
        "full_win", "half_win", "push", "half_loss", "full_loss",
        "fair_decimal_odds",
    ]
    print(markets[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
