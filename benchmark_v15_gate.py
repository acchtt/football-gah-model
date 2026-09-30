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
    use_model_by_paired_loss,
)
from gah.data import load_openfootball_league_seasons
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

    hist_model_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []

    hist_v13_match_brier: list[float] = []
    hist_naive_match_brier: list[float] = []

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

        beta = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            beta = fit_total_tilt_beta(
                hist_base_pmfs[start:],
                hist_actual_totals[start:],
            )

        v13_matrix = tilt_score_matrix_by_total(base_matrix, beta)
        v13_pmf = matrix_to_total_pmf(v13_matrix)

        use_v13 = use_model_by_paired_loss(
            hist_v13_match_brier,
            hist_naive_match_brier,
            min_history=100,
            z_threshold=1.0,
            window=380,
        )
        gate_matrix = v13_matrix if use_v13 else naive_matrix
        gate_pmf = matrix_to_total_pmf(gate_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        match_rows.append(
            {
                "season": target["season"],
                "use_v13": bool(use_v13),
                "blend_weight": blend_weight,
                "beta": beta,
                "v13_nll": _nll(pmf_probability(v13_pmf, actual_total)),
                "gate_nll": _nll(pmf_probability(gate_pmf, actual_total)),
                "naive_nll": _nll(
                    pmf_probability(matrix_to_total_pmf(naive_matrix), actual_total)
                ),
                "v13_point_mae": abs(
                    absolute_error_optimal_point(v13_pmf) - actual_total
                ),
                "gate_point_mae": abs(
                    absolute_error_optimal_point(gate_pmf) - actual_total
                ),
                "naive_point_mae": abs(
                    absolute_error_optimal_point(
                        matrix_to_total_pmf(naive_matrix)
                    ) - actual_total
                ),
            }
        )

        current_v13_briers: list[float] = []
        current_naive_briers: list[float] = []

        for line in TOTAL_LINES:
            v13_pricing = price_total(v13_matrix, line, "over")
            gate_pricing = price_total(gate_matrix, line, "over")
            naive_pricing = price_total(naive_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")

            v13_brier = multiclass_brier(v13_pricing, actual)
            naive_brier = multiclass_brier(naive_pricing, actual)
            current_v13_briers.append(v13_brier)
            current_naive_briers.append(naive_brier)

            line_rows.append(
                {
                    "season": target["season"],
                    "line": line,
                    "use_v13": bool(use_v13),
                    "v13_brier": v13_brier,
                    "gate_brier": multiclass_brier(gate_pricing, actual),
                    "naive_brier": naive_brier,
                    "gate_predicted_value": expected_settlement_value(gate_pricing),
                    "observed_value": observed_settlement_value(actual),
                }
            )

        # Update all sequential calibrators only after the current prediction.
        hist_model_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

        hist_v13_match_brier.append(float(np.mean(current_v13_briers)))
        hist_naive_match_brier.append(float(np.mean(current_naive_briers)))

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, group in lines.groupby("line", sort=True):
        v13 = float(group["v13_brier"].mean())
        gate = float(group["gate_brier"].mean())
        naive = float(group["naive_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "n": int(len(group)),
                "v13_brier": v13,
                "gate_brier": gate,
                "naive_brier": naive,
                "gate_vs_v13_pct": 100.0 * (v13 - gate) / v13,
                "gate_vs_naive_pct": 100.0 * (naive - gate) / naive,
                "ece": expected_calibration_error(
                    group["gate_predicted_value"].to_numpy(),
                    group["observed_value"].to_numpy(),
                ),
                "bias": calibration_bias(
                    group["gate_predicted_value"].to_numpy(),
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
    gate_brier = float(lines["gate_brier"].mean())
    naive_brier = float(lines["naive_brier"].mean())
    fallback_share = float((~matches["use_v13"]).mean())

    report = [
        f"# GAH v1.5 Reliability-Gate Totals — {competition}",
        "",
        "Candidate experiment. The gate uses only prior per-match OOS multi-line",
        "Brier losses. It falls back to the league baseline only when v1.3 is",
        "worse by more than one paired standard error over the recent window.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 | Reliability gate | Naive | Gate vs v1.3 | Gate vs naive |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13_brier:.4f} | {gate_brier:.4f} | "
        f"{naive_brier:.4f} | {100.0*(v13_brier-gate_brier)/v13_brier:+.2f}% | "
        f"{100.0*(naive_brier-gate_brier)/naive_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['gate_nll'].mean():.4f} | {matches['naive_nll'].mean():.4f} | "
        f"{100.0*(matches['v13_nll'].mean()-matches['gate_nll'].mean())/matches['v13_nll'].mean():+.2f}% | "
        f"{100.0*(matches['naive_nll'].mean()-matches['gate_nll'].mean())/matches['naive_nll'].mean():+.2f}% |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['gate_point_mae'].mean():.4f} | {matches['naive_point_mae'].mean():.4f} | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['gate_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% | "
        f"{100.0*(matches['naive_point_mae'].mean()-matches['gate_point_mae'].mean())/matches['naive_point_mae'].mean():+.2f}% |",
        "",
        f"Fallback-to-baseline share: **{fallback_share:.1%}**",
        f"Lines improved vs v1.3: **{int((summary['gate_brier'] < summary['v13_brier']).sum())}/{len(summary)}**",
        f"Lines beating naive: **{int((summary['gate_brier'] < summary['naive_brier']).sum())}/{len(summary)}**",
        "",
        "## By line",
        "",
        "| Line | v1.3 | Gate | Naive | Gate vs v1.3 | Gate vs naive | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['v13_brier']:.4f} | "
            f"{row['gate_brier']:.4f} | {row['naive_brier']:.4f} | "
            f"{row['gate_vs_v13_pct']:+.2f}% | {row['gate_vs_naive_pct']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    report += [
        "",
        "## By season",
        "",
        "| Season | N | v1.3 Brier | Gate Brier | Naive Brier | Fallback share |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for season, group in lines.groupby("season", sort=True):
        mg = matches[matches["season"] == season]
        report.append(
            f"| {season} | {len(mg)} | {group['v13_brier'].mean():.4f} | "
            f"{group['gate_brier'].mean():.4f} | {group['naive_brier'].mean():.4f} | "
            f"{float((~mg['use_v13']).mean()):.1%} |"
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
