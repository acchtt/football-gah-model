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
from gah.markets import price_total, settle_total
from gah.model import DixonColesModel
from gah.tactical import ConditionalTacticalTilt, TacticalTeamState
from gah.totals import (
    absolute_error_optimal_point,
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    pmf_probability,
    tilt_score_matrix_by_total,
)
from gah.xg import load_understat_team_stats_seasons


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]


def _nll(p: float) -> float:
    return -math.log(max(float(p), 1e-12))


def _update_state(state: TacticalTeamState, row: pd.Series) -> None:
    state.update(
        row["home_team"],
        row["away_team"],
        float(row["home_np_xg"]),
        float(row["away_np_xg"]),
        float(row["home_ppda"]) if pd.notna(row["home_ppda"]) else np.nan,
        float(row["away_ppda"]) if pd.notna(row["away_ppda"]) else np.nan,
        float(row["home_deep_completions"]) if pd.notna(row["home_deep_completions"]) else np.nan,
        float(row["away_deep_completions"]) if pd.notna(row["away_deep_completions"]) else np.nan,
    )


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    tactical_state = TacticalTeamState(max_history=30)

    for _, row in data.iloc[:380].iterrows():
        _update_state(tactical_state, row)

    hist_dc_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []

    tactical_features: list[np.ndarray] = []
    tactical_pmfs: list[np.ndarray] = []
    tactical_actuals: list[int] = []
    tactical_model: ConditionalTacticalTilt | None = None
    last_tactical_fit_n = 0

    line_rows: list[dict] = []
    match_rows: list[dict] = []

    for i in range(380, len(data)):
        target = data.iloc[i]
        train = data.iloc[:i]

        known = set(train["home_team"]).union(train["away_team"])
        if target["home_team"] not in known or target["away_team"] not in known:
            _update_state(tactical_state, target)
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

        blend_weight = 1.0
        if len(hist_dc_prob) >= 100:
            start = max(0, len(hist_dc_prob) - 380)
            blend_weight = _fit_total_blend_weight(
                hist_dc_prob[start:],
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

        features = tactical_state.features(
            target["home_team"],
            target["away_team"],
        )

        if (
            len(tactical_features) >= 200
            and (
                tactical_model is None
                or len(tactical_features) - last_tactical_fit_n >= 20
            )
        ):
            start = max(0, len(tactical_features) - 760)
            candidate = ConditionalTacticalTilt(
                l2=2.0,
                beta_clip=0.08,
            )
            candidate.fit(
                np.stack(tactical_features[start:]),
                tactical_pmfs[start:],
                tactical_actuals[start:],
            )
            tactical_model = candidate
            last_tactical_fit_n = len(tactical_features)

        tactical_beta = (
            tactical_model.predict_beta(features)
            if tactical_model is not None
            else 0.0
        )
        v16_matrix = tilt_score_matrix_by_total(v13_matrix, tactical_beta)
        v16_pmf = matrix_to_total_pmf(v16_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        match_rows.append(
            {
                "season": target["season"],
                "tactical_beta": tactical_beta,
                "v13_nll": _nll(pmf_probability(v13_pmf, actual_total)),
                "v16_nll": _nll(pmf_probability(v16_pmf, actual_total)),
                "v13_point_mae": abs(
                    absolute_error_optimal_point(v13_pmf) - actual_total
                ),
                "v16_point_mae": abs(
                    absolute_error_optimal_point(v16_pmf) - actual_total
                ),
            }
        )

        for line in TOTAL_LINES:
            v13_price = price_total(v13_matrix, line, "over")
            v16_price = price_total(v16_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")
            line_rows.append(
                {
                    "season": target["season"],
                    "line": line,
                    "v13_brier": multiclass_brier(v13_price, actual),
                    "v16_brier": multiclass_brier(v16_price, actual),
                    "v16_predicted_value": expected_settlement_value(v16_price),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        hist_dc_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

        tactical_features.append(features)
        tactical_pmfs.append(v13_pmf)
        tactical_actuals.append(actual_total)

        _update_state(tactical_state, target)

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def build_report(
    competition: str,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    rows = []
    for line, group in lines.groupby("line", sort=True):
        v13 = float(group["v13_brier"].mean())
        v16 = float(group["v16_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "v13": v13,
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
    summary = pd.DataFrame(rows)

    v13_brier = float(lines["v13_brier"].mean())
    v16_brier = float(lines["v16_brier"].mean())
    active = matches[np.abs(matches["tactical_beta"]) > 1e-12]

    report = [
        f"# GAH v1.6 Tactical xG Residual — {competition}",
        "",
        "Production v1.3 totals remain the base distribution. The new layer applies",
        "a small conditional tilt learned only from prior OOS npXG, PPDA, and deep",
        "completion team context.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 | v1.6 tactical | Change |",
        "|---|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13_brier:.4f} | {v16_brier:.4f} | "
        f"{100.0*(v13_brier-v16_brier)/v13_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['v16_nll'].mean():.4f} | "
        f"{100.0*(matches['v13_nll'].mean()-matches['v16_nll'].mean())/matches['v13_nll'].mean():+.2f}% |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['v16_point_mae'].mean():.4f} | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['v16_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% |",
        "",
        f"Residual-active predictions: **{len(active)}/{len(matches)}**",
        f"Mean |tactical beta| when active: **{active['tactical_beta'].abs().mean() if len(active) else 0.0:.4f}**",
        f"Lines improved vs v1.3: **{int((summary['v16'] < summary['v13']).sum())}/{len(summary)}**",
        "",
        "## By line",
        "",
        "| Line | v1.3 | v1.6 tactical | Change | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['v13']:.4f} | "
            f"{row['v16']:.4f} | {row['change']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
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
    print(build_report(competition, lines, matches))


if __name__ == "__main__":
    main()
