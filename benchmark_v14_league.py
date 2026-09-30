from __future__ import annotations

import argparse
import math
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
from gah.data import load_openfootball_league_seasons
from gah.markets import (
    price_handicap,
    price_total,
    settle_handicap,
    settle_total,
)
from gah.model import DixonColesModel
from gah.totals import (
    absolute_error_optimal_point,
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    pmf_probability,
    tilt_score_matrix_by_total,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
AH_LINES = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
]


def _nll(p: float) -> float:
    return -math.log(max(float(p), 1e-12))


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None

    hist_model_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []

    line_rows: list[dict] = []
    match_rows: list[dict] = []

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

        base_totals_matrix = (
            blend_weight * raw_matrix
            + (1.0 - blend_weight) * naive_matrix
        )
        base_totals_matrix = base_totals_matrix / base_totals_matrix.sum()
        base_pmf = matrix_to_total_pmf(base_totals_matrix)

        beta = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            beta = fit_total_tilt_beta(
                hist_base_pmfs[start:],
                hist_actual_totals[start:],
            )

        totals_matrix = tilt_score_matrix_by_total(base_totals_matrix, beta)
        total_pmf = matrix_to_total_pmf(totals_matrix)

        hg = int(target["home_goals"])
        ag = int(target["away_goals"])
        actual_total = hg + ag
        actual_margin = hg - ag

        match_rows.append(
            {
                "season": target["season"],
                "blend_weight": blend_weight,
                "beta": beta,
                "total_nll": _nll(pmf_probability(total_pmf, actual_total)),
                "point_mae": abs(
                    absolute_error_optimal_point(total_pmf) - actual_total
                ),
                "margin_mae": abs(pred["expected_margin"] - actual_margin),
            }
        )

        for line in TOTAL_LINES:
            pricing = price_total(totals_matrix, line, "over")
            naive_pricing = price_total(naive_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")
            line_rows.append(
                {
                    "season": target["season"],
                    "market": "total",
                    "line": line,
                    "model_brier": multiclass_brier(pricing, actual),
                    "naive_brier": multiclass_brier(naive_pricing, actual),
                    "predicted_value": expected_settlement_value(pricing),
                    "naive_predicted_value": expected_settlement_value(naive_pricing),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        for line in AH_LINES:
            pricing = price_handicap(raw_matrix, line, "home")
            naive_pricing = price_handicap(naive_matrix, line, "home")
            actual = settle_handicap(actual_margin, line, "home")
            line_rows.append(
                {
                    "season": target["season"],
                    "market": "asian_handicap",
                    "line": line,
                    "model_brier": multiclass_brier(pricing, actual),
                    "naive_brier": multiclass_brier(naive_pricing, actual),
                    "predicted_value": expected_settlement_value(pricing),
                    "naive_predicted_value": expected_settlement_value(naive_pricing),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        hist_model_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, line), g in lines.groupby(["market", "line"], sort=True):
        model_brier = float(g["model_brier"].mean())
        naive_brier = float(g["naive_brier"].mean())
        rows.append(
            {
                "market": market,
                "line": float(line),
                "n": int(len(g)),
                "model_brier": model_brier,
                "naive_brier": naive_brier,
                "improvement_pct": (
                    100.0 * (naive_brier - model_brier) / naive_brier
                    if naive_brier > 0 else 0.0
                ),
                "ece": expected_calibration_error(
                    g["predicted_value"].to_numpy(),
                    g["observed_value"].to_numpy(),
                    n_bins=10,
                ),
                "bias": calibration_bias(
                    g["predicted_value"].to_numpy(),
                    g["observed_value"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    competition: str,
    data: pd.DataFrame,
    line_results: pd.DataFrame,
    match_results: pd.DataFrame,
) -> str:
    summary = summarize_lines(line_results)
    totals = summary[summary["market"] == "total"]
    ah = summary[summary["market"] == "asian_handicap"]

    total_model = float(totals["model_brier"].mean())
    total_naive = float(totals["naive_brier"].mean())
    ah_model = float(ah["model_brier"].mean())
    ah_naive = float(ah["naive_brier"].mean())

    total_better = int((totals["model_brier"] < totals["naive_brier"]).sum())
    ah_better = int((ah["model_brier"] < ah["naive_brier"]).sum())

    seasons = ", ".join(sorted(data["season"].unique()))
    lines = [
        f"# GAH v1.4 Cross-Competition Validation — {competition}",
        "",
        "Frozen GAH v1.3 architecture; no league-specific feature changes.",
        "",
        "## Protocol",
        "",
        f"- Seasons: {seasons}",
        "- Initial training window: 380 chronological matches",
        f"- Completed matches loaded: {len(data)}",
        f"- Out-of-sample predictions: {len(match_results)}",
        "- Dixon-Coles refit every 20 evaluated matches or for a newly observed team",
        "- Totals: OOS DC/league blend + OOS total-goal tilt",
        "- Asian handicap: pure Dixon-Coles",
        "- 11 totals lines and 17 AH lines",
        "- Five-outcome Asian settlement Brier is the primary line metric",
        "",
        "## Aggregate",
        "",
        "| Metric | GAH v1.3 | Naive | Improvement |",
        "|---|---:|---:|---:|",
        f"| Totals multi-line Brier ↓ | {total_model:.4f} | {total_naive:.4f} | "
        f"{100.0*(total_naive-total_model)/total_naive:+.2f}% |",
        f"| AH multi-line Brier ↓ | {ah_model:.4f} | {ah_naive:.4f} | "
        f"{100.0*(ah_naive-ah_model)/ah_naive:+.2f}% |",
        "",
        f"Totals lines beating naive: **{total_better}/{len(totals)}**",
        f"AH lines beating naive: **{ah_better}/{len(ah)}**",
        f"Exact-total NLL: **{match_results['total_nll'].mean():.4f}**",
        f"Median total point MAE: **{match_results['point_mae'].mean():.4f}**",
        f"Goal-margin MAE: **{match_results['margin_mae'].mean():.4f}**",
        f"Mean blend weight on Dixon-Coles: **{match_results['blend_weight'].mean():.3f}**",
        f"Mean totals tilt beta: **{match_results['beta'].mean():+.4f}**",
        "",
        "## Totals by line",
        "",
        "| Line | Model Brier | Naive Brier | Improvement | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in totals.sort_values("line").iterrows():
        lines.append(
            f"| {row['line']:+.2f} | {row['model_brier']:.4f} | "
            f"{row['naive_brier']:.4f} | {row['improvement_pct']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    lines += [
        "",
        "## Asian handicap by line",
        "",
        "| Line | Model Brier | Naive Brier | Improvement | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ah.sort_values("line").iterrows():
        lines.append(
            f"| {row['line']:+.2f} | {row['model_brier']:.4f} | "
            f"{row['naive_brier']:.4f} | {row['improvement_pct']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_openfootball_league_seasons(args.seasons, competition)
    line_results, match_results = run_benchmark(data)
    report = build_report(competition, data, line_results, match_results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
