from __future__ import annotations

import argparse
import math

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
from gah.xg import NPXGTeamState, load_understat_team_stats_seasons


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]


def _nll(prob: float) -> float:
    return -math.log(max(float(prob), 1e-12))


def _prepare_state(data: pd.DataFrame, end: int) -> NPXGTeamState:
    state = NPXGTeamState(max_history=30)
    for _, row in data.iloc[:end].iterrows():
        state.update(
            row["home_team"],
            row["away_team"],
            row["home_xg"],
            row["away_xg"],
            row["home_np_xg"],
            row["away_np_xg"],
        )
    return state


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    state = _prepare_state(data, 380)

    hist_dc_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs = []
    hist_actual_totals: list[int] = []
    hist_v13_prob: list[float] = []
    hist_npxg_prob: list[float] = []

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
            state.update(
                target["home_team"],
                target["away_team"],
                target["home_xg"],
                target["away_xg"],
                target["home_np_xg"],
                target["away_np_xg"],
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

        naive_matrix = _independent_poisson_matrix(
            float(train["home_goals"].mean()),
            float(train["away_goals"].mean()),
            max_goals=10,
        )

        dc_weight = 1.0
        if len(hist_dc_prob) >= 100:
            start = max(0, len(hist_dc_prob) - 380)
            dc_weight = _fit_total_blend_weight(
                hist_dc_prob[start:],
                hist_naive_prob[start:],
            )

        base_matrix = (
            dc_weight * raw_matrix
            + (1.0 - dc_weight) * naive_matrix
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

        np_home, np_away = state.expected_xg(
            target["home_team"],
            target["away_team"],
            window=10,
            prior_matches=5.0,
        )
        npxg_matrix = _independent_poisson_matrix(
            np_home,
            np_away,
            max_goals=10,
        )
        npxg_pmf = matrix_to_total_pmf(npxg_matrix)

        v13_weight = 1.0
        if len(hist_v13_prob) >= 100:
            start = max(0, len(hist_v13_prob) - 380)
            v13_weight = _fit_total_blend_weight(
                hist_v13_prob[start:],
                hist_npxg_prob[start:],
            )

        v16_matrix = (
            v13_weight * v13_matrix
            + (1.0 - v13_weight) * npxg_matrix
        )
        v16_matrix = v16_matrix / v16_matrix.sum()
        v16_pmf = matrix_to_total_pmf(v16_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        match_rows.append(
            {
                "season": target["season"],
                "npxg_weight": 1.0 - v13_weight,
                "npxg_home": np_home,
                "npxg_away": np_away,
                "v13_nll": _nll(pmf_probability(v13_pmf, actual_total)),
                "npxg_nll": _nll(pmf_probability(npxg_pmf, actual_total)),
                "v16_nll": _nll(pmf_probability(v16_pmf, actual_total)),
                "v13_point_mae": abs(
                    absolute_error_optimal_point(v13_pmf) - actual_total
                ),
                "npxg_point_mae": abs(
                    absolute_error_optimal_point(npxg_pmf) - actual_total
                ),
                "v16_point_mae": abs(
                    absolute_error_optimal_point(v16_pmf) - actual_total
                ),
            }
        )

        for line in TOTAL_LINES:
            v13_price = price_total(v13_matrix, line, "over")
            npxg_price = price_total(npxg_matrix, line, "over")
            v16_price = price_total(v16_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")

            line_rows.append(
                {
                    "season": target["season"],
                    "line": line,
                    "v13_brier": multiclass_brier(v13_price, actual),
                    "npxg_brier": multiclass_brier(npxg_price, actual),
                    "v16_brier": multiclass_brier(v16_price, actual),
                    "v16_predicted_value": expected_settlement_value(v16_price),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        hist_dc_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)
        hist_v13_prob.append(_total_probability(v13_matrix, actual_total))
        hist_npxg_prob.append(_total_probability(npxg_matrix, actual_total))

        state.update(
            target["home_team"],
            target["away_team"],
            target["home_xg"],
            target["away_xg"],
            target["home_np_xg"],
            target["away_np_xg"],
        )

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def build_report(
    competition: str,
    data: pd.DataFrame,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    groups = []
    for line, group in lines.groupby("line", sort=True):
        v13 = float(group["v13_brier"].mean())
        npb = float(group["npxg_brier"].mean())
        v16 = float(group["v16_brier"].mean())
        groups.append(
            {
                "line": float(line),
                "v13": v13,
                "npxg": npb,
                "v16": v16,
                "change": 100.0 * (v13 - v16) / v13,
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
    summary = pd.DataFrame(groups)

    v13_brier = float(lines["v13_brier"].mean())
    npxg_brier = float(lines["npxg_brier"].mean())
    v16_brier = float(lines["v16_brier"].mean())

    report = [
        f"# GAH v1.6 Non-Penalty xG — {competition}",
        "",
        "This experiment separates repeatable non-penalty shot quality from the",
        "sparse penalty-xG component, then lets prior OOS likelihood choose how",
        "much weight the non-penalty xG matrix receives.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 base | npXG-only | v1.6 npXG blend | Change vs v1.3 |",
        "|---|---:|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13_brier:.4f} | {npxg_brier:.4f} | "
        f"{v16_brier:.4f} | {100.0*(v13_brier-v16_brier)/v13_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['npxg_nll'].mean():.4f} | {matches['v16_nll'].mean():.4f} | "
        f"{100.0*(matches['v13_nll'].mean()-matches['v16_nll'].mean())/matches['v13_nll'].mean():+.2f}% |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['npxg_point_mae'].mean():.4f} | {matches['v16_point_mae'].mean():.4f} | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['v16_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% |",
        "",
        f"Completed matches loaded: **{len(data)}**",
        f"OOS predictions: **{len(matches)}**",
        f"Mean npXG component weight: **{matches['npxg_weight'].mean():.3f}**",
        f"Median npXG component weight: **{matches['npxg_weight'].median():.3f}**",
        f"Lines improved vs v1.3: **{int((summary['v16'] < summary['v13']).sum())}/{len(summary)}**",
        "",
        "## By line",
        "",
        "| Line | v1.3 | npXG-only | v1.6 | Change | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['v13']:.4f} | "
            f"{row['npxg']:.4f} | {row['v16']:.4f} | "
            f"{row['change']:+.2f}% | {row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_understat_team_stats_seasons(args.seasons, competition)
    lines, matches = run_benchmark(data)
    print(build_report(competition, data, lines, matches))


if __name__ == "__main__":
    main()
