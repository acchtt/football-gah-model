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
    fit_brier_blend_weight,
    multiclass_brier,
    observed_settlement_value,
)
from gah.data import load_openfootball_league_seasons
from gah.markets import (
    SETTLEMENT_CATEGORIES,
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


def _pricing_vector(pricing: dict) -> np.ndarray:
    return np.array(
        [float(pricing.get(category, 0.0)) for category in SETTLEMENT_CATEGORIES],
        dtype=float,
    )


def _observed_vector(actual_category: str) -> np.ndarray:
    out = np.zeros(len(SETTLEMENT_CATEGORIES), dtype=float)
    out[SETTLEMENT_CATEGORIES.index(actual_category)] = 1.0
    return out


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

    # Historical full-line settlement distributions for the v1.4 safety blend.
    hist_v13_vectors: list[np.ndarray] = []
    hist_naive_vectors: list[np.ndarray] = []
    hist_observed_vectors: list[np.ndarray] = []

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

        v13_totals_matrix = tilt_score_matrix_by_total(base_totals_matrix, beta)

        safety_alpha = 1.0
        if len(hist_v13_vectors) >= 100:
            start = max(0, len(hist_v13_vectors) - 380)
            safety_alpha = fit_brier_blend_weight(
                np.stack(hist_v13_vectors[start:]),
                np.stack(hist_naive_vectors[start:]),
                np.stack(hist_observed_vectors[start:]),
            )

        totals_matrix = (
            safety_alpha * v13_totals_matrix
            + (1.0 - safety_alpha) * naive_matrix
        )
        totals_matrix = totals_matrix / totals_matrix.sum()

        v13_pmf = matrix_to_total_pmf(v13_totals_matrix)
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
                "safety_alpha": safety_alpha,
                "v13_total_nll": _nll(pmf_probability(v13_pmf, actual_total)),
                "total_nll": _nll(pmf_probability(total_pmf, actual_total)),
                "v13_point_mae": abs(
                    absolute_error_optimal_point(v13_pmf) - actual_total
                ),
                "point_mae": abs(
                    absolute_error_optimal_point(total_pmf) - actual_total
                ),
                "margin_mae": abs(pred["expected_margin"] - actual_margin),
            }
        )

        current_v13_vectors: list[np.ndarray] = []
        current_naive_vectors: list[np.ndarray] = []
        current_observed_vectors: list[np.ndarray] = []

        for line in TOTAL_LINES:
            v13_pricing = price_total(v13_totals_matrix, line, "over")
            pricing = price_total(totals_matrix, line, "over")
            naive_pricing = price_total(naive_matrix, line, "over")
            actual = settle_total(actual_total, line, "over")

            v13_vec = _pricing_vector(v13_pricing)
            naive_vec = _pricing_vector(naive_pricing)
            observed_vec = _observed_vector(actual)
            current_v13_vectors.append(v13_vec)
            current_naive_vectors.append(naive_vec)
            current_observed_vectors.append(observed_vec)

            line_rows.append(
                {
                    "season": target["season"],
                    "market": "total",
                    "line": line,
                    "v13_brier": multiclass_brier(v13_pricing, actual),
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
            brier = multiclass_brier(pricing, actual)
            line_rows.append(
                {
                    "season": target["season"],
                    "market": "asian_handicap",
                    "line": line,
                    "v13_brier": brier,
                    "model_brier": brier,
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

        # Update the safety optimizer only AFTER the current outcome is known.
        hist_v13_vectors.append(np.stack(current_v13_vectors))
        hist_naive_vectors.append(np.stack(current_naive_vectors))
        hist_observed_vectors.append(np.stack(current_observed_vectors))

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def summarize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, line), g in lines.groupby(["market", "line"], sort=True):
        v13_brier = float(g["v13_brier"].mean())
        model_brier = float(g["model_brier"].mean())
        naive_brier = float(g["naive_brier"].mean())
        rows.append(
            {
                "market": market,
                "line": float(line),
                "n": int(len(g)),
                "v13_brier": v13_brier,
                "model_brier": model_brier,
                "naive_brier": naive_brier,
                "vs_v13_pct": (
                    100.0 * (v13_brier - model_brier) / v13_brier
                    if v13_brier > 0 else 0.0
                ),
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

    total_v13 = float(totals["v13_brier"].mean())
    total_model = float(totals["model_brier"].mean())
    total_naive = float(totals["naive_brier"].mean())
    ah_model = float(ah["model_brier"].mean())
    ah_naive = float(ah["naive_brier"].mean())

    total_better = int((totals["model_brier"] < totals["naive_brier"]).sum())
    total_vs_v13 = int((totals["model_brier"] < totals["v13_brier"]).sum())
    ah_better = int((ah["model_brier"] < ah["naive_brier"]).sum())

    seasons = ", ".join(sorted(data["season"].unique()))
    lines = [
        f"# GAH v1.4 Cross-Competition Validation — {competition}",
        "",
        "GAH v1.4 keeps Asian handicap unchanged and adds a leakage-safe totals",
        "safety blend. The safety weight is the closed-form mixture that minimizes",
        "historical out-of-sample multi-line settlement Brier between v1.3 totals",
        "and the league-average baseline.",
        "",
        "## Protocol",
        "",
        f"- Seasons: {seasons}",
        "- Initial training window: 380 chronological matches",
        f"- Completed matches loaded: {len(data)}",
        f"- Out-of-sample predictions: {len(match_results)}",
        "- Dixon-Coles refit every 20 evaluated matches or for a newly observed team",
        "- Totals: v1.3 OOS blend + tilt, then OOS Brier-optimal safety blend",
        "- Asian handicap: pure Dixon-Coles",
        "- 11 totals lines and 17 AH lines",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 | v1.4 safety | Naive | v1.4 vs v1.3 | v1.4 vs naive |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Totals multi-line Brier ↓ | {total_v13:.4f} | {total_model:.4f} | "
        f"{total_naive:.4f} | {100.0*(total_v13-total_model)/total_v13:+.2f}% | "
        f"{100.0*(total_naive-total_model)/total_naive:+.2f}% |",
        "",
        f"AH multi-line Brier: **{ah_model:.4f}** vs naive **{ah_naive:.4f}** "
        f"({100.0*(ah_naive-ah_model)/ah_naive:+.2f}%)",
        f"Totals lines beating naive: **{total_better}/{len(totals)}**",
        f"Totals lines improved vs v1.3: **{total_vs_v13}/{len(totals)}**",
        f"AH lines beating naive: **{ah_better}/{len(ah)}**",
        f"v1.3 exact-total NLL: **{match_results['v13_total_nll'].mean():.4f}**",
        f"v1.4 exact-total NLL: **{match_results['total_nll'].mean():.4f}**",
        f"v1.3 median total MAE: **{match_results['v13_point_mae'].mean():.4f}**",
        f"v1.4 median total MAE: **{match_results['point_mae'].mean():.4f}**",
        f"Goal-margin MAE: **{match_results['margin_mae'].mean():.4f}**",
        f"Mean DC blend weight: **{match_results['blend_weight'].mean():.3f}**",
        f"Mean totals tilt beta: **{match_results['beta'].mean():+.4f}**",
        f"Mean v1.4 safety alpha: **{match_results['safety_alpha'].mean():.3f}**",
        "",
        "## Totals by line",
        "",
        "| Line | v1.3 Brier | v1.4 Brier | Naive | v1.4 vs v1.3 | v1.4 vs naive | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in totals.sort_values("line").iterrows():
        lines.append(
            f"| {row['line']:+.2f} | {row['v13_brier']:.4f} | "
            f"{row['model_brier']:.4f} | {row['naive_brier']:.4f} | "
            f"{row['vs_v13_pct']:+.2f}% | {row['improvement_pct']:+.2f}% | "
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
