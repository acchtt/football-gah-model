from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gah.backtest import walk_forward_backtest, summarize_backtest
from gah.data import load_openfootball_epl_seasons


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def improvement(model_value: float, naive_value: float) -> float:
    if naive_value == 0:
        return 0.0
    return 100.0 * (naive_value - model_value) / naive_value


def build_report(results: pd.DataFrame, seasons: list[str], half_life: float) -> str:
    s = summarize_backtest(results)

    lines = [
        "# EPL Baseline Benchmark",
        "",
        "This is the first code-first benchmark for Football GAH.",
        "",
        "## Protocol",
        "",
        f"- Source: OpenFootball public-domain EPL results",
        f"- Seasons loaded: {', '.join(seasons)}",
        "- First 380 chronological matches used before evaluation begins",
        "- Expanding walk-forward training only; no future matches enter a prediction",
        "- Dixon-Coles model refitted every 20 evaluated matches and immediately when a newly observed team requires it",
        f"- Exponential time decay half-life: {half_life:.0f} days",
        "- Fixed market checks: Over 2.5 and Home -0.5",
        "- Naive comparator: training-window league-average home/away goal rates",
        "",
        "## Overall",
        "",
        "| Metric | GAH v1 | Naive | Improvement |",
        "|---|---:|---:|---:|",
        f"| Exact-score NLL ↓ | {s['mean_score_nll']:.4f} | {s['mean_naive_score_nll']:.4f} | {improvement(s['mean_score_nll'], s['mean_naive_score_nll']):+.2f}% |",
        f"| Total-goals MAE ↓ | {s['mean_total_abs_error']:.4f} | {s['mean_naive_total_abs_error']:.4f} | {improvement(s['mean_total_abs_error'], s['mean_naive_total_abs_error']):+.2f}% |",
        f"| Goal-margin MAE ↓ | {s['mean_margin_abs_error']:.4f} | {s['mean_naive_margin_abs_error']:.4f} | {improvement(s['mean_margin_abs_error'], s['mean_naive_margin_abs_error']):+.2f}% |",
        f"| Over 2.5 Brier ↓ | {s['mean_over_25_brier']:.4f} | {s['mean_naive_over_25_brier']:.4f} | {improvement(s['mean_over_25_brier'], s['mean_naive_over_25_brier']):+.2f}% |",
        f"| Home -0.5 Brier ↓ | {s['mean_home_m05_brier']:.4f} | {s['mean_naive_home_m05_brier']:.4f} | {improvement(s['mean_home_m05_brier'], s['mean_naive_home_m05_brier']):+.2f}% |",
        "",
        f"Evaluated predictions: **{s['n_predictions']}**",
        "",
        "## By season",
        "",
        "| Season | N | Score NLL | Total MAE | Margin MAE | O2.5 Brier | Home -0.5 Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    if "season" in results.columns:
        for season, g in results.groupby("season", sort=True):
            lines.append(
                f"| {season} | {len(g)} | "
                f"{g['score_nll'].mean():.4f} | "
                f"{g['total_abs_error'].mean():.4f} | "
                f"{g['margin_abs_error'].mean():.4f} | "
                f"{g['over_2.5_brier'].mean():.4f} | "
                f"{g['home_-0.5_brier'].mean():.4f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "This report is a baseline, not a betting strategy. No bookmaker prices, xG, lineups, injuries, shots, or ML residual features are used yet.",
        "A feature should only be kept in later versions if it improves genuinely out-of-sample metrics under the same walk-forward discipline.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--half-life", type=float, default=180.0)
    ap.add_argument("--min-train", type=int, default=380)
    ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--output", default="benchmarks/epl_baseline.md")
    args = ap.parse_args()

    data = load_openfootball_epl_seasons(args.seasons)
    results = walk_forward_backtest(
        data,
        half_life_days=args.half_life,
        min_train_matches=args.min_train,
        refit_every=args.refit_every,
        total_line=2.5,
        handicap_line=-0.5,
    )

    report = build_report(results, args.seasons, args.half_life)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
