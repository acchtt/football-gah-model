from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gah.backtest import (
    _fit_total_blend_weight,
    _independent_poisson_matrix,
    _total_probability,
)
from gah.calibration import (
    calibration_bias,
    expected_calibration_error,
    expected_settlement_value,
    multiclass_brier,
    observed_settlement_value,
)
from gah.data import load_openfootball_epl_seasons
from gah.markets import (
    price_handicap,
    price_total,
    settle_handicap,
    settle_total,
)
from gah.model import DixonColesModel


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
AH_LINES = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
]


def run_benchmark(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    rows: list[dict] = []

    hist_model_prob: list[float] = []
    hist_naive_prob: list[float] = []

    for i in range(380, len(data)):
        target = data.iloc[i]
        train = data.iloc[:i]

        known = set(train["home_team"]).union(train["away_team"])
        if target["home_team"] not in known or target["away_team"] not in known:
            continue

        needs_refit = (
            model is None
            or last_fit_i is None
            or (i - last_fit_i) >= 20
            or target["home_team"] not in model.team_to_idx_
            or target["away_team"] not in model.team_to_idx_
        )
        if needs_refit:
            model = DixonColesModel(half_life_days=180.0, max_goals=10)
            model.fit(train, as_of=train["match_date"].max())
            last_fit_i = i

        pred = model.predict(target["home_team"], target["away_team"])
        raw_matrix = pred["score_matrix"]

        naive_home_rate = float(train["home_goals"].mean())
        naive_away_rate = float(train["away_goals"].mean())
        naive_matrix = _independent_poisson_matrix(
            naive_home_rate,
            naive_away_rate,
            max_goals=10,
        )

        blend_weight = 1.0
        if len(hist_model_prob) >= 100:
            start = max(0, len(hist_model_prob) - 380)
            blend_weight = _fit_total_blend_weight(
                hist_model_prob[start:],
                hist_naive_prob[start:],
            )

        totals_matrix = (
            blend_weight * raw_matrix
            + (1.0 - blend_weight) * naive_matrix
        )
        totals_matrix = totals_matrix / totals_matrix.sum()

        hg = int(target["home_goals"])
        ag = int(target["away_goals"])
        actual_total = hg + ag
        actual_margin = hg - ag

        for line in TOTAL_LINES:
            pricing = price_total(totals_matrix, line, "over")
            naive_pricing = price_total(naive_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")

            rows.append(
                {
                    "season": target["season"],
                    "market": "total",
                    "side": "over",
                    "line": line,
                    "model_brier": multiclass_brier(pricing, actual),
                    "naive_brier": multiclass_brier(naive_pricing, actual),
                    "predicted_value": expected_settlement_value(pricing),
                    "naive_predicted_value": expected_settlement_value(naive_pricing),
                    "observed_value": observed_settlement_value(actual),
                    "blend_weight": blend_weight,
                }
            )

        for line in AH_LINES:
            pricing = price_handicap(raw_matrix, line, "home")
            naive_pricing = price_handicap(naive_matrix, line, "home")
            actual = settle_handicap(actual_margin, line, "home")

            rows.append(
                {
                    "season": target["season"],
                    "market": "asian_handicap",
                    "side": "home",
                    "line": line,
                    "model_brier": multiclass_brier(pricing, actual),
                    "naive_brier": multiclass_brier(naive_pricing, actual),
                    "predicted_value": expected_settlement_value(pricing),
                    "naive_predicted_value": expected_settlement_value(naive_pricing),
                    "observed_value": observed_settlement_value(actual),
                    "blend_weight": blend_weight,
                }
            )

        # Exact-total likelihoods update the blend only after this match was priced.
        hist_model_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))

    return pd.DataFrame(rows)


def summarize_lines(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (market, side, line), group in results.groupby(["market", "side", "line"], sort=True):
        model_brier = float(group["model_brier"].mean())
        naive_brier = float(group["naive_brier"].mean())
        summary_rows.append(
            {
                "market": market,
                "side": side,
                "line": float(line),
                "n": int(len(group)),
                "model_brier": model_brier,
                "naive_brier": naive_brier,
                "brier_improvement_pct": (
                    100.0 * (naive_brier - model_brier) / naive_brier
                    if naive_brier > 0 else 0.0
                ),
                "ece": expected_calibration_error(
                    group["predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                    n_bins=10,
                ),
                "naive_ece": expected_calibration_error(
                    group["naive_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                    n_bins=10,
                ),
                "bias": calibration_bias(
                    group["predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
                "naive_bias": calibration_bias(
                    group["naive_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def table_lines(df: pd.DataFrame) -> list[str]:
    lines = [
        "| Line | Model Brier ↓ | Naive Brier ↓ | Improvement | ECE ↓ | Bias |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.sort_values("line").iterrows():
        lines.append(
            f"| {row['line']:+.2f} | "
            f"{row['model_brier']:.4f} | "
            f"{row['naive_brier']:.4f} | "
            f"{row['brier_improvement_pct']:+.2f}% | "
            f"{row['ece']:.4f} | "
            f"{row['bias']:+.4f} |"
        )
    return lines


def build_report(results: pd.DataFrame, summary: pd.DataFrame) -> str:
    totals = summary[summary["market"] == "total"]
    ah = summary[summary["market"] == "asian_handicap"]

    total_model = float(totals["model_brier"].mean())
    total_naive = float(totals["naive_brier"].mean())
    ah_model = float(ah["model_brier"].mean())
    ah_naive = float(ah["naive_brier"].mean())

    total_worse = totals[totals["model_brier"] > totals["naive_brier"]]
    ah_worse = ah[ah["model_brier"] > ah["naive_brier"]]

    report = [
        "# EPL GAH v1.3 Full Asian-Line Calibration",
        "",
        "The v1.2 model is frozen. This benchmark validates the full standard",
        "Asian totals and handicap line grids using five-outcome settlement scoring.",
        "",
        "## Protocol",
        "",
        "- EPL 2021-22 through 2025-26",
        "- 1,513 out-of-sample match predictions",
        "- Totals use the promoted leakage-safe v1.2 blended distribution",
        "- Asian handicap uses pure Dixon-Coles",
        "- Quarter lines are scored as full win / half win / push / half loss / full loss",
        "- Primary metric: proper multiclass settlement Brier score",
        "- ECE/bias use normalized settlement value only as descriptive diagnostics",
        "- Over and Home are sufficient because opposite sides are algebraic settlement mirrors",
        "",
        "## Aggregate",
        "",
        "| Market | Model Brier ↓ | Naive Brier ↓ | Improvement | Mean ECE ↓ | Mean |bias| ↓ |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Totals | {total_model:.4f} | {total_naive:.4f} | "
        f"{100.0*(total_naive-total_model)/total_naive:+.2f}% | "
        f"{totals['ece'].mean():.4f} | {totals['bias'].abs().mean():.4f} |",
        f"| Asian handicap | {ah_model:.4f} | {ah_naive:.4f} | "
        f"{100.0*(ah_naive-ah_model)/ah_naive:+.2f}% | "
        f"{ah['ece'].mean():.4f} | {ah['bias'].abs().mean():.4f} |",
        "",
        f"Totals lines worse than naive: **{len(total_worse)}/{len(totals)}**",
        f"AH lines worse than naive: **{len(ah_worse)}/{len(ah)}**",
        "",
        "## Totals — Over side",
        "",
    ]
    report += table_lines(totals)
    report += ["", "## Asian handicap — Home side", ""]
    report += table_lines(ah)

    worst_bias = summary.assign(abs_bias=summary["bias"].abs()).sort_values(
        "abs_bias", ascending=False
    ).head(8)
    report += [
        "",
        "## Largest calibration biases",
        "",
        "| Market | Line | Bias | ECE | Brier improvement vs naive |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in worst_bias.iterrows():
        report.append(
            f"| {row['market']} | {row['line']:+.2f} | "
            f"{row['bias']:+.4f} | {row['ece']:.4f} | "
            f"{row['brier_improvement_pct']:+.2f}% |"
        )

    report += [
        "",
        "## Interpretation rule",
        "",
        "Do not recalibrate a line merely because one diagnostic is imperfect.",
        "A v1.3 probability transformation is justified only when a stable directional",
        "bias appears across adjacent lines and/or seasons. Otherwise the result is",
        "validation evidence and v1.2 remains unchanged.",
        "",
    ]
    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--output", default="benchmarks/epl_v13_line_calibration.md")
    ap.add_argument("--csv", default="benchmarks/epl_v13_line_calibration.csv")
    args = ap.parse_args()

    data = load_openfootball_epl_seasons(args.seasons)
    results = run_benchmark(data)
    summary = summarize_lines(results)
    report = build_report(results, summary)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")

    csv = Path(args.csv)
    csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv, index=False)

    print(report)


if __name__ == "__main__":
    main()
