from __future__ import annotations

import argparse
import pandas as pd

from gah.backtest import walk_forward_backtest, summarize_backtest


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest for Football GAH v1")
    ap.add_argument("csv")
    ap.add_argument("--competition", default=None)
    ap.add_argument("--half-life", type=float, default=180.0)
    ap.add_argument("--min-train", type=int, default=250)
    ap.add_argument("--refit-every", type=int, default=50)
    ap.add_argument("--out", default="backtest_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.competition is not None:
        if "competition" not in df.columns:
            raise SystemExit("CSV has no 'competition' column.")
        df = df[df["competition"] == args.competition].copy()

    results = walk_forward_backtest(
        df,
        half_life_days=args.half_life,
        min_train_matches=args.min_train,
        refit_every=args.refit_every,
    )
    results.to_csv(args.out, index=False)
    print(summarize_backtest(results))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
