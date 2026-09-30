from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gah.backtest import walk_forward_backtest, summarize_backtest
from gah.data import load_openfootball_epl_seasons


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return 100.0 * (old - new) / old


def build_report(results: pd.DataFrame, seasons: list[str]) -> str:
    s = summarize_backtest(results)

    lines = [
        "# EPL GAH v1.1 Totals Experiment",
        "",
        "GAH v1.1 keeps the Asian-handicap core unchanged and changes only the totals engine.",
        "The totals distribution is an online mixture of the GAH Dixon-Coles distribution and",
        "a training-window league-average Poisson distribution. The mixture weight is fitted",
        "only from earlier out-of-sample total-goal likelihoods.",
        "",
        "## Protocol",
        "",
        f"- Seasons: {', '.join(seasons)}",
        "- Initial training window: 380 chronological matches",
        "- Expanding walk-forward evaluation",
        "- Dixon-Coles refit every 20 evaluated matches or when a new team requires it",
        "- Totals blend warm-up: 100 prior OOS predictions",
        "- Totals blend lookback: 380 prior OOS predictions",
        "- Asian handicap remains pure GAH v1",
        "",
        "## Overall totals",
        "",
        "| Metric | GAH v1.1 blend | GAH v1 | Naive | v1.1 vs v1 | v1.1 vs naive |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Total-goals MAE ↓ | {s['mean_blended_total_abs_error']:.4f} | {s['mean_total_abs_error']:.4f} | {s['mean_naive_total_abs_error']:.4f} | {pct_change(s['mean_blended_total_abs_error'], s['mean_total_abs_error']):+.2f}% | {pct_change(s['mean_blended_total_abs_error'], s['mean_naive_total_abs_error']):+.2f}% |",
        f"| Total-goals NLL ↓ | {s['mean_blended_total_nll']:.4f} | {s['mean_total_nll']:.4f} | {s['mean_naive_total_nll']:.4f} | {pct_change(s['mean_blended_total_nll'], s['mean_total_nll']):+.2f}% | {pct_change(s['mean_blended_total_nll'], s['mean_naive_total_nll']):+.2f}% |",
        f"| Over 2.5 Brier ↓ | {s['mean_blended_over_25_brier']:.4f} | {s['mean_over_25_brier']:.4f} | {s['mean_naive_over_25_brier']:.4f} | {pct_change(s['mean_blended_over_25_brier'], s['mean_over_25_brier']):+.2f}% | {pct_change(s['mean_blended_over_25_brier'], s['mean_naive_over_25_brier']):+.2f}% |",
        "",
        f"Mean GAH weight in totals blend: **{s['mean_total_blend_weight']:.3f}**",
        f"Evaluated predictions: **{s['n_predictions']}**",
        "",
        "## Handicap guardrail",
        "",
        "The AH engine is intentionally unchanged. Its benchmark values should match v1:",
        "",
        f"- Goal-margin MAE: **{s['mean_margin_abs_error']:.4f}**",
        f"- Home -0.5 Brier: **{s['mean_home_m05_brier']:.4f}**",
        "",
        "## By season",
        "",
        "| Season | N | Blend MAE | v1 MAE | Naive MAE | Blend O2.5 Brier | v1 O2.5 | Naive O2.5 | Mean GAH weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for season, g in results.groupby("season", sort=True):
        lines.append(
            f"| {season} | {len(g)} | "
            f"{g['blended_total_abs_error'].mean():.4f} | "
            f"{g['total_abs_error'].mean():.4f} | "
            f"{g['naive_total_abs_error'].mean():.4f} | "
            f"{g['blended_over_2.5_brier'].mean():.4f} | "
            f"{g['over_2.5_brier'].mean():.4f} | "
            f"{g['naive_over_2.5_brier'].mean():.4f} | "
            f"{g['total_blend_weight'].mean():.3f} |"
        )

    lines += [
        "",
        "## Promotion rule",
        "",
        "v1.1 should be promoted only if it improves total-goal likelihood and O2.5 Brier",
        "without changing the v1 Asian-handicap results. Total MAE is a secondary guardrail.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--output", default="benchmarks/epl_v11_totals.md")
    args = ap.parse_args()

    data = load_openfootball_epl_seasons(args.seasons)
    results = walk_forward_backtest(
        data,
        half_life_days=180.0,
        min_train_matches=380,
        refit_every=20,
        total_line=2.5,
        handicap_line=-0.5,
        total_blend=True,
        total_blend_min_history=100,
        total_blend_window=380,
    )

    report = build_report(results, args.seasons)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
