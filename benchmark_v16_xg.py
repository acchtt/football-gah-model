from __future__ import annotations

import argparse
import math
from collections import Counter

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
from gah.markets import price_total, settle_total
from gah.model import DixonColesModel
from gah.totals import (
    absolute_error_optimal_point,
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    pmf_probability,
    tilt_score_matrix_by_total,
)
from gah.xg import XGTeamState, load_understat_xg_seasons


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]


def _nll(prob: float) -> float:
    return -math.log(max(float(prob), 1e-12))


def _prepare_xg_state(data: pd.DataFrame, end: int) -> XGTeamState:
    state = XGTeamState(max_history=30)
    for _, row in data.iloc[:end].iterrows():
        state.update(
            row["home_team"],
            row["away_team"],
            row["home_xg"],
            row["away_xg"],
        )
    return state


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    xg_state = _prepare_xg_state(data, 380)

    hist_dc_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []

    hist_v13_prob: list[float] = []
    hist_xg_prob: list[float] = []

    line_rows: list[dict] = []
    match_rows: list[dict] = []

    for i in range(380, len(data)):
        target = data.iloc[i]
        train = data.iloc[:i]

        known = set(train["home_team"]).union(train["away_team"])
        can_predict = (
            target["home_team"] in known
            and target["away_team"] in known
        )

        if not can_predict:
            xg_state.update(
                target["home_team"],
                target["away_team"],
                target["home_xg"],
                target["away_xg"],
            )
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

        dc_blend_weight = 1.0
        if len(hist_dc_prob) >= 100:
            start = max(0, len(hist_dc_prob) - 380)
            dc_blend_weight = _fit_total_blend_weight(
                hist_dc_prob[start:],
                hist_naive_prob[start:],
            )

        base_matrix = (
            dc_blend_weight * raw_matrix
            + (1.0 - dc_blend_weight) * naive_matrix
        )
        base_matrix = base_matrix / base_matrix.sum()
        base_pmf = matrix_to_total_pmf(base_matrix)

        global_beta = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            global_beta = fit_total_tilt_beta(
                hist_base_pmfs[start:],
                hist_actual_totals[start:],
            )

        v13_matrix = tilt_score_matrix_by_total(base_matrix, global_beta)
        v13_pmf = matrix_to_total_pmf(v13_matrix)

        xg_home, xg_away = xg_state.expected_xg(
            target["home_team"],
            target["away_team"],
            window=10,
            prior_matches=5.0,
        )
        xg_matrix = _independent_poisson_matrix(
            xg_home,
            xg_away,
            max_goals=10,
        )
        xg_pmf = matrix_to_total_pmf(xg_matrix)

        v13_weight = 1.0
        if len(hist_v13_prob) >= 100:
            start = max(0, len(hist_v13_prob) - 380)
            v13_weight = _fit_total_blend_weight(
                hist_v13_prob[start:],
                hist_xg_prob[start:],
            )

        v16_matrix = (
            v13_weight * v13_matrix
            + (1.0 - v13_weight) * xg_matrix
        )
        v16_matrix = v16_matrix / v16_matrix.sum()
        v16_pmf = matrix_to_total_pmf(v16_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        match_rows.append(
            {
                "season": target["season"],
                "v13_weight": v13_weight,
                "xg_weight": 1.0 - v13_weight,
                "dc_blend_weight": dc_blend_weight,
                "global_beta": global_beta,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "actual_home_xg": float(target["home_xg"]),
                "actual_away_xg": float(target["away_xg"]),
                "v13_nll": _nll(pmf_probability(v13_pmf, actual_total)),
                "xg_nll": _nll(pmf_probability(xg_pmf, actual_total)),
                "v16_nll": _nll(pmf_probability(v16_pmf, actual_total)),
                "v13_point_mae": abs(
                    absolute_error_optimal_point(v13_pmf) - actual_total
                ),
                "xg_point_mae": abs(
                    absolute_error_optimal_point(xg_pmf) - actual_total
                ),
                "v16_point_mae": abs(
                    absolute_error_optimal_point(v16_pmf) - actual_total
                ),
            }
        )

        for line in TOTAL_LINES:
            v13_pricing = price_total(v13_matrix, line, "over")
            xg_pricing = price_total(xg_matrix, line, "over")
            v16_pricing = price_total(v16_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")

            line_rows.append(
                {
                    "season": target["season"],
                    "line": line,
                    "v13_brier": multiclass_brier(v13_pricing, actual),
                    "xg_brier": multiclass_brier(xg_pricing, actual),
                    "v16_brier": multiclass_brier(v16_pricing, actual),
                    "v16_predicted_value": expected_settlement_value(v16_pricing),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        dc_actual_prob = _total_probability(raw_matrix, actual_total)
        naive_actual_prob = _total_probability(naive_matrix, actual_total)
        v13_actual_prob = _total_probability(v13_matrix, actual_total)
        xg_actual_prob = _total_probability(xg_matrix, actual_total)

        hist_dc_prob.append(dc_actual_prob)
        hist_naive_prob.append(naive_actual_prob)
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)
        hist_v13_prob.append(v13_actual_prob)
        hist_xg_prob.append(xg_actual_prob)

        xg_state.update(
            target["home_team"],
            target["away_team"],
            target["home_xg"],
            target["away_xg"],
        )

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, group in lines.groupby("line", sort=True):
        v13 = float(group["v13_brier"].mean())
        xg = float(group["xg_brier"].mean())
        v16 = float(group["v16_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "v13_brier": v13,
                "xg_brier": xg,
                "v16_brier": v16,
                "v16_vs_v13_pct": 100.0 * (v13 - v16) / v13,
                "ece": expected_calibration_error(
                    group["v16_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
                "bias": calibration_bias(
                    group["v16_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    competition: str,
    data: pd.DataFrame,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    summary = summarize_lines(lines)

    v13_brier = float(lines["v13_brier"].mean())
    xg_brier = float(lines["xg_brier"].mean())
    v16_brier = float(lines["v16_brier"].mean())

    report = [
        f"# GAH v1.6 Understat xG Enrichment — {competition}",
        "",
        "The production v1.3 totals distribution is frozen as one component.",
        "A second matrix is built only from historical Understat xG information;",
        "an online OOS likelihood blend learns how much xG deserves to contribute.",
        "",
        "## Data",
        "",
        f"- Completed Understat matches loaded: **{len(data)}**",
        f"- Out-of-sample predictions: **{len(matches)}**",
        "- Five seasons: 2021-22 through 2025-26",
        "- xG state uses only matches completed before the prediction",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 base | xG-only | v1.6 blend | v1.6 vs v1.3 |",
        "|---|---:|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13_brier:.4f} | {xg_brier:.4f} | "
        f"{v16_brier:.4f} | {100.0*(v13_brier-v16_brier)/v13_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['xg_nll'].mean():.4f} | {matches['v16_nll'].mean():.4f} | "
        f"{100.0*(matches['v13_nll'].mean()-matches['v16_nll'].mean())/matches['v13_nll'].mean():+.2f}% |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['xg_point_mae'].mean():.4f} | {matches['v16_point_mae'].mean():.4f} | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['v16_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% |",
        "",
        f"Mean xG component weight: **{matches['xg_weight'].mean():.3f}**",
        f"Median xG component weight: **{matches['xg_weight'].median():.3f}**",
        f"Lines improved vs v1.3: **{int((summary['v16_brier'] < summary['v13_brier']).sum())}/{len(summary)}**",
        "",
        "## By line",
        "",
        "| Line | v1.3 | xG-only | v1.6 | v1.6 vs v1.3 | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['v13_brier']:.4f} | "
            f"{row['xg_brier']:.4f} | {row['v16_brier']:.4f} | "
            f"{row['v16_vs_v13_pct']:+.2f}% | {row['ece']:.4f} | "
            f"{row['bias']:+.4f} |"
        )

    report += [
        "",
        "## By season",
        "",
        "| Season | N | v1.3 Brier | v1.6 Brier | Change | Mean xG weight |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for season, group in lines.groupby("season", sort=True):
        match_group = matches[matches["season"] == season]
        a = float(group["v13_brier"].mean())
        b = float(group["v16_brier"].mean())
        report.append(
            f"| {season} | {len(match_group)} | {a:.4f} | {b:.4f} | "
            f"{100.0*(a-b)/a:+.2f}% | {match_group['xg_weight'].mean():.3f} |"
        )

    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_understat_xg_seasons(args.seasons, competition)
    lines, matches = run_benchmark(data)
    print(build_report(competition, data, lines, matches))


if __name__ == "__main__":
    main()
