from __future__ import annotations

import argparse
import math

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
from gah.markets import price_total, settle_total
from gah.model import DixonColesModel
from gah.residual import ConditionalTotalTiltModel, TeamResidualState
from gah.totals import (
    absolute_error_optimal_point,
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    pmf_mean,
    pmf_probability,
    tilt_score_matrix_by_total,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]


def _nll(prob: float) -> float:
    return -math.log(max(float(prob), 1e-12))


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

    residual_state = TeamResidualState(max_history=30)
    residual_features: list[np.ndarray] = []
    residual_pmfs: list[np.ndarray] = []
    residual_actual_totals: list[int] = []
    residual_model: ConditionalTotalTiltModel | None = None
    last_residual_fit_n = 0

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

        base_matrix = (
            blend_weight * raw_matrix
            + (1.0 - blend_weight) * naive_matrix
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

        features = residual_state.features(
            target["home_team"],
            target["away_team"],
            target["match_date"],
        )

        if (
            len(residual_features) >= 200
            and (
                residual_model is None
                or len(residual_features) - last_residual_fit_n >= 20
            )
        ):
            start = max(0, len(residual_features) - 760)
            candidate = ConditionalTotalTiltModel(
                l2=1.0,
                beta_clip=0.15,
            )
            candidate.fit(
                np.stack(residual_features[start:]),
                residual_pmfs[start:],
                residual_actual_totals[start:],
            )
            residual_model = candidate
            last_residual_fit_n = len(residual_features)

        residual_beta = (
            residual_model.predict_beta(features)
            if residual_model is not None
            else 0.0
        )
        v15_matrix = tilt_score_matrix_by_total(v13_matrix, residual_beta)
        v15_pmf = matrix_to_total_pmf(v15_matrix)

        hg = int(target["home_goals"])
        ag = int(target["away_goals"])
        actual_total = hg + ag

        match_rows.append(
            {
                "season": target["season"],
                "global_beta": global_beta,
                "residual_beta": residual_beta,
                "blend_weight": blend_weight,
                "v13_nll": _nll(pmf_probability(v13_pmf, actual_total)),
                "v15_nll": _nll(pmf_probability(v15_pmf, actual_total)),
                "v13_point_mae": abs(
                    absolute_error_optimal_point(v13_pmf) - actual_total
                ),
                "v15_point_mae": abs(
                    absolute_error_optimal_point(v15_pmf) - actual_total
                ),
            }
        )

        for line in TOTAL_LINES:
            v13_pricing = price_total(v13_matrix, line, "over")
            v15_pricing = price_total(v15_matrix, line, "over")
            naive_pricing = price_total(naive_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")

            line_rows.append(
                {
                    "season": target["season"],
                    "line": line,
                    "v13_brier": multiclass_brier(v13_pricing, actual),
                    "v15_brier": multiclass_brier(v15_pricing, actual),
                    "naive_brier": multiclass_brier(naive_pricing, actual),
                    "v15_predicted_value": expected_settlement_value(v15_pricing),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        # Update all state strictly after evaluating the current prediction.
        hist_model_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

        residual_features.append(features)
        residual_pmfs.append(v13_pmf)
        residual_actual_totals.append(actual_total)

        residual_state.update(
            target["home_team"],
            target["away_team"],
            target["match_date"],
            home_goals=hg,
            away_goals=ag,
            dc_xg_home=pred["xg_home"],
            dc_xg_away=pred["xg_away"],
            base_total_mean=pmf_mean(v13_pmf),
        )

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, group in lines.groupby("line", sort=True):
        v13 = float(group["v13_brier"].mean())
        v15 = float(group["v15_brier"].mean())
        naive = float(group["naive_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "n": int(len(group)),
                "v13_brier": v13,
                "v15_brier": v15,
                "naive_brier": naive,
                "v15_vs_v13_pct": 100.0 * (v13 - v15) / v13,
                "v15_vs_naive_pct": 100.0 * (naive - v15) / naive,
                "ece": expected_calibration_error(
                    group["v15_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
                "bias": calibration_bias(
                    group["v15_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    competition: str,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    summary = summarize_lines(lines)
    v13_brier = float(lines["v13_brier"].mean())
    v15_brier = float(lines["v15_brier"].mean())
    naive_brier = float(lines["naive_brier"].mean())

    active = matches[np.abs(matches["residual_beta"]) > 1e-12]

    report = [
        f"# GAH v1.5 Team/Match Residual Totals — {competition}",
        "",
        "Candidate experiment. v1.3 remains the base totals distribution; the new",
        "layer applies a match-specific conditional exponential tilt learned only",
        "from prior OOS team residual features.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 | v1.5 residual | Naive | v1.5 vs v1.3 | v1.5 vs naive |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13_brier:.4f} | {v15_brier:.4f} | "
        f"{naive_brier:.4f} | {100.0*(v13_brier-v15_brier)/v13_brier:+.2f}% | "
        f"{100.0*(naive_brier-v15_brier)/naive_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['v15_nll'].mean():.4f} | — | "
        f"{100.0*(matches['v13_nll'].mean()-matches['v15_nll'].mean())/matches['v13_nll'].mean():+.2f}% | — |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['v15_point_mae'].mean():.4f} | — | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['v15_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% | — |",
        "",
        f"Residual-active predictions: **{len(active)}/{len(matches)}**",
        f"Mean |residual beta| when active: **{active['residual_beta'].abs().mean() if len(active) else 0.0:.4f}**",
        f"Lines improved vs v1.3: **{int((summary['v15_brier'] < summary['v13_brier']).sum())}/{len(summary)}**",
        f"Lines beating naive: **{int((summary['v15_brier'] < summary['naive_brier']).sum())}/{len(summary)}**",
        "",
        "## By line",
        "",
        "| Line | v1.3 | v1.5 | Naive | v1.5 vs v1.3 | v1.5 vs naive | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['v13_brier']:.4f} | "
            f"{row['v15_brier']:.4f} | {row['naive_brier']:.4f} | "
            f"{row['v15_vs_v13_pct']:+.2f}% | {row['v15_vs_naive_pct']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    report += [
        "",
        "## By season",
        "",
        "| Season | N | v1.3 Brier | v1.5 Brier | Change | Mean |beta| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for season, group in lines.groupby("season", sort=True):
        mg = matches[matches["season"] == season]
        a = float(group["v13_brier"].mean())
        b = float(group["v15_brier"].mean())
        report.append(
            f"| {season} | {len(mg)} | {a:.4f} | {b:.4f} | "
            f"{100.0*(a-b)/a:+.2f}% | {mg['residual_beta'].abs().mean():.4f} |"
        )

    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    data = load_openfootball_league_seasons(
        args.seasons,
        args.competition.upper(),
    )
    lines, matches = run_benchmark(data)
    print(build_report(args.competition.upper(), lines, matches))


if __name__ == "__main__":
    main()
