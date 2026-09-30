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
from gah.totals import (
    absolute_error_optimal_point,
    fit_total_tilt_beta,
    fit_total_tilt_beta_brier,
    matrix_to_total_pmf,
    pmf_probability,
    tilt_score_matrix_by_total,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0)


def _nll(prob: float) -> float:
    return -math.log(max(float(prob), 1e-12))


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None

    hist_dc_prob: list[float] = []
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

        naive_matrix = _independent_poisson_matrix(
            float(train["home_goals"].mean()),
            float(train["away_goals"].mean()),
            max_goals=10,
        )

        weight = 1.0
        if len(hist_dc_prob) >= 100:
            start = max(0, len(hist_dc_prob) - 380)
            weight = _fit_total_blend_weight(
                hist_dc_prob[start:],
                hist_naive_prob[start:],
            )

        base_matrix = weight * raw_matrix + (1.0 - weight) * naive_matrix
        base_matrix = base_matrix / base_matrix.sum()
        base_pmf = matrix_to_total_pmf(base_matrix)

        beta_v13 = 0.0
        beta_v15 = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            recent_pmfs = hist_base_pmfs[start:]
            recent_totals = hist_actual_totals[start:]
            beta_v13 = fit_total_tilt_beta(recent_pmfs, recent_totals)
            beta_v15 = fit_total_tilt_beta_brier(
                recent_pmfs,
                recent_totals,
                lines=TOTAL_LINES,
            )

        v13_matrix = tilt_score_matrix_by_total(base_matrix, beta_v13)
        v15_matrix = tilt_score_matrix_by_total(base_matrix, beta_v15)
        v13_pmf = matrix_to_total_pmf(v13_matrix)
        v15_pmf = matrix_to_total_pmf(v15_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        match_rows.append(
            {
                "season": target["season"],
                "blend_weight": weight,
                "v13_beta": beta_v13,
                "v15_beta": beta_v15,
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

        # Update only after the prediction is scored.
        hist_dc_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, group in lines.groupby("line", sort=True):
        a = float(group["v13_brier"].mean())
        b = float(group["v15_brier"].mean())
        n = float(group["naive_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "v13_brier": a,
                "v15_brier": b,
                "naive_brier": n,
                "vs_v13_pct": 100.0 * (a - b) / a,
                "vs_naive_pct": 100.0 * (n - b) / n,
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
    v13 = float(lines["v13_brier"].mean())
    v15 = float(lines["v15_brier"].mean())
    naive = float(lines["naive_brier"].mean())

    out = [
        f"# GAH v1.5 Asian-Brier Tilt — {competition}",
        "",
        "Candidate experiment. The v1.3 DC↔league blend is unchanged. Only the",
        "total-goal exponential tilt objective changes from exact-total NLL to",
        "full-grid Asian settlement Brier.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 | v1.5 Brier tilt | Naive | v1.5 vs v1.3 | v1.5 vs naive |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13:.4f} | {v15:.4f} | {naive:.4f} | "
        f"{100.0*(v13-v15)/v13:+.2f}% | {100.0*(naive-v15)/naive:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['v15_nll'].mean():.4f} | — | "
        f"{100.0*(matches['v13_nll'].mean()-matches['v15_nll'].mean())/matches['v13_nll'].mean():+.2f}% | — |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['v15_point_mae'].mean():.4f} | — | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['v15_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% | — |",
        "",
        f"Mean blend weight: **{matches['blend_weight'].mean():.3f}**",
        f"Mean v1.3 beta: **{matches['v13_beta'].mean():+.4f}**",
        f"Mean v1.5 beta: **{matches['v15_beta'].mean():+.4f}**",
        f"Lines improved vs v1.3: **{int((summary['v15_brier'] < summary['v13_brier']).sum())}/{len(summary)}**",
        f"Lines beating naive: **{int((summary['v15_brier'] < summary['naive_brier']).sum())}/{len(summary)}**",
        "",
        "## By line",
        "",
        "| Line | v1.3 | v1.5 | Naive | vs v1.3 | vs naive | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        out.append(
            f"| {row['line']:+.2f} | {row['v13_brier']:.4f} | "
            f"{row['v15_brier']:.4f} | {row['naive_brier']:.4f} | "
            f"{row['vs_v13_pct']:+.2f}% | {row['vs_naive_pct']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    out += [
        "",
        "## By season",
        "",
        "| Season | N | v1.3 Brier | v1.5 Brier | Naive | Change | v1.3 beta | v1.5 beta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for season, group in lines.groupby("season", sort=True):
        mg = matches[matches["season"] == season]
        a = float(group["v13_brier"].mean())
        b = float(group["v15_brier"].mean())
        out.append(
            f"| {season} | {len(mg)} | {a:.4f} | {b:.4f} | "
            f"{group['naive_brier'].mean():.4f} | {100.0*(a-b)/a:+.2f}% | "
            f"{mg['v13_beta'].mean():+.4f} | {mg['v15_beta'].mean():+.4f} |"
        )

    return "\n".join(out)


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
