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
from gah.data import load_openfootball_epl_seasons
from gah.markets import price_total, settle_total
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


def _nll(p: float) -> float:
    return -math.log(max(float(p), 1e-12))


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    line_rows: list[dict] = []
    match_rows: list[dict] = []

    hist_model_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []

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

        base_matrix = (
            blend_weight * raw_matrix
            + (1.0 - blend_weight) * naive_matrix
        )
        base_matrix = base_matrix / base_matrix.sum()
        base_pmf = matrix_to_total_pmf(base_matrix)

        beta = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            beta = fit_total_tilt_beta(
                hist_base_pmfs[start:],
                hist_actual_totals[start:],
            )

        tilted_matrix = tilt_score_matrix_by_total(base_matrix, beta)
        tilted_pmf = matrix_to_total_pmf(tilted_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        base_nll = _nll(pmf_probability(base_pmf, actual_total))
        tilted_nll = _nll(pmf_probability(tilted_pmf, actual_total))

        base_point = absolute_error_optimal_point(base_pmf)
        tilted_point = absolute_error_optimal_point(tilted_pmf)

        match_rows.append(
            {
                "season": target["season"],
                "beta": beta,
                "blend_weight": blend_weight,
                "base_nll": base_nll,
                "tilted_nll": tilted_nll,
                "base_point_mae": abs(base_point - actual_total),
                "tilted_point_mae": abs(tilted_point - actual_total),
            }
        )

        for line in TOTAL_LINES:
            base_pricing = price_total(base_matrix, line, "over")
            tilted_pricing = price_total(tilted_matrix, line, "over")
            naive_pricing = price_total(naive_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")
            observed = observed_settlement_value(actual)

            line_rows.append(
                {
                    "season": target["season"],
                    "line": line,
                    "base_brier": multiclass_brier(base_pricing, actual),
                    "tilted_brier": multiclass_brier(tilted_pricing, actual),
                    "naive_brier": multiclass_brier(naive_pricing, actual),
                    "base_predicted_value": expected_settlement_value(base_pricing),
                    "tilted_predicted_value": expected_settlement_value(tilted_pricing),
                    "naive_predicted_value": expected_settlement_value(naive_pricing),
                    "observed_value": observed,
                    "beta": beta,
                }
            )

        # Update every calibrator only after this match has been predicted.
        hist_model_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, g in results.groupby("line", sort=True):
        base_brier = float(g["base_brier"].mean())
        tilted_brier = float(g["tilted_brier"].mean())
        naive_brier = float(g["naive_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "n": int(len(g)),
                "base_brier": base_brier,
                "tilted_brier": tilted_brier,
                "naive_brier": naive_brier,
                "tilt_vs_base_pct": 100.0 * (base_brier - tilted_brier) / base_brier,
                "tilt_vs_naive_pct": 100.0 * (naive_brier - tilted_brier) / naive_brier,
                "base_ece": expected_calibration_error(
                    g["base_predicted_value"].to_numpy(),
                    g["observed_value"].to_numpy(),
                ),
                "tilted_ece": expected_calibration_error(
                    g["tilted_predicted_value"].to_numpy(),
                    g["observed_value"].to_numpy(),
                ),
                "base_bias": calibration_bias(
                    g["base_predicted_value"].to_numpy(),
                    g["observed_value"].to_numpy(),
                ),
                "tilted_bias": calibration_bias(
                    g["tilted_predicted_value"].to_numpy(),
                    g["observed_value"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(lines: pd.DataFrame, matches: pd.DataFrame) -> str:
    summary = summarize_lines(lines)

    base_brier = float(lines["base_brier"].mean())
    tilted_brier = float(lines["tilted_brier"].mean())
    naive_brier = float(lines["naive_brier"].mean())

    base_nll = float(matches["base_nll"].mean())
    tilted_nll = float(matches["tilted_nll"].mean())
    base_mae = float(matches["base_point_mae"].mean())
    tilted_mae = float(matches["tilted_point_mae"].mean())

    worse_lines = summary[summary["tilted_brier"] > summary["base_brier"]]

    report = [
        "# EPL GAH v1.3 Totals Probability-Tilt Experiment",
        "",
        "This experiment responds to the v1.3 line-calibration finding that all",
        "totals lines showed negative settlement-value bias. It applies one",
        "leakage-safe exponential tilt to the whole totals distribution.",
        "",
        "## Overall",
        "",
        "| Metric | v1.2 base | v1.3 tilt | Change |",
        "|---|---:|---:|---:|",
        f"| Multi-line settlement Brier ↓ | {base_brier:.4f} | {tilted_brier:.4f} | "
        f"{100.0*(base_brier-tilted_brier)/base_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {base_nll:.4f} | {tilted_nll:.4f} | "
        f"{100.0*(base_nll-tilted_nll)/base_nll:+.2f}% |",
        f"| Median point MAE ↓ | {base_mae:.4f} | {tilted_mae:.4f} | "
        f"{100.0*(base_mae-tilted_mae)/base_mae:+.2f}% |",
        "",
        f"Naive multi-line settlement Brier: **{naive_brier:.4f}**",
        f"Mean learned beta: **{matches['beta'].mean():+.4f}**",
        f"Median learned beta: **{matches['beta'].median():+.4f}**",
        f"Lines with worse Brier after tilt: **{len(worse_lines)}/{len(summary)}**",
        "",
        "## By total line",
        "",
        "| Line | Base Brier | Tilt Brier | Tilt vs base | Tilt vs naive | Base ECE | Tilt ECE | Base bias | Tilt bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['base_brier']:.4f} | "
            f"{row['tilted_brier']:.4f} | {row['tilt_vs_base_pct']:+.2f}% | "
            f"{row['tilt_vs_naive_pct']:+.2f}% | {row['base_ece']:.4f} | "
            f"{row['tilted_ece']:.4f} | {row['base_bias']:+.4f} | "
            f"{row['tilted_bias']:+.4f} |"
        )

    report += [
        "",
        "## Promotion rule",
        "",
        "Promote only if the tilt improves aggregate multi-line Brier and exact-total",
        "NLL, materially reduces the directional totals bias, does not materially",
        "damage median-point MAE, and avoids broad line-specific regressions.",
        "",
    ]
    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--output", default="benchmarks/epl_v13_tilt.md")
    ap.add_argument("--csv", default="benchmarks/epl_v13_tilt.csv")
    args = ap.parse_args()

    data = load_openfootball_epl_seasons(args.seasons)
    line_results, match_results = run_benchmark(data)
    report = build_report(line_results, match_results)
    summary = summarize_lines(line_results)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")

    csv = Path(args.csv)
    csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv, index=False)

    print(report)


if __name__ == "__main__":
    main()
