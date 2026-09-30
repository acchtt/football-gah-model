from __future__ import annotations

import argparse
import math
from collections import Counter
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
from gah.markets import price_total, settle_total
from gah.model import DixonColesModel
from gah.totals import (
    absolute_error_optimal_point,
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    pmf_probability,
    recency_weighted_home_away_rates,
    tilt_score_matrix_by_total,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TOTAL_LINES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
REGIME_HALF_LIVES: tuple[float | None, ...] = (90.0, 180.0, 365.0, None)


def _label(half_life: float | None) -> str:
    return "unweighted" if half_life is None else f"{int(half_life)}d"


def _nll(prob: float) -> float:
    return -math.log(max(float(prob), 1e-12))


def _select_regime(
    histories: dict[str, list[float]],
    min_history: int = 100,
    window: int = 380,
) -> str:
    if not histories or min(len(v) for v in histories.values()) < min_history:
        return "unweighted"

    best_label = "unweighted"
    best_nll = math.inf
    for label, probs in histories.items():
        start = max(0, len(probs) - window)
        arr = np.asarray(probs[start:], dtype=float)
        score = float(-np.log(np.clip(arr, 1e-12, 1.0)).mean())
        if score < best_nll:
            best_nll = score
            best_label = label
    return best_label


def _build_baselines(
    train: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for half_life in REGIME_HALF_LIVES:
        home_rate, away_rate = recency_weighted_home_away_rates(
            train,
            as_of=as_of,
            half_life_days=half_life,
            recent_matches=None,
        )
        out[_label(half_life)] = _independent_poisson_matrix(
            home_rate,
            away_rate,
            max_goals=10,
        )
    return out


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None

    regime_histories = {_label(h): [] for h in REGIME_HALF_LIVES}

    v13_hist_dc_prob: list[float] = []
    v13_hist_baseline_prob: list[float] = []
    v13_hist_base_pmfs: list[np.ndarray] = []
    v13_hist_actual_totals: list[int] = []

    v15_hist_dc_prob: list[float] = []
    v15_hist_baseline_prob: list[float] = []
    v15_hist_base_pmfs: list[np.ndarray] = []
    v15_hist_actual_totals: list[int] = []

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

        as_of = train["match_date"].max()
        baselines = _build_baselines(train, as_of)
        v13_baseline = baselines["unweighted"]

        selected_regime = _select_regime(regime_histories)
        v15_baseline = baselines[selected_regime]

        v13_weight = 1.0
        if len(v13_hist_dc_prob) >= 100:
            start = max(0, len(v13_hist_dc_prob) - 380)
            v13_weight = _fit_total_blend_weight(
                v13_hist_dc_prob[start:],
                v13_hist_baseline_prob[start:],
            )

        v15_weight = 1.0
        if len(v15_hist_dc_prob) >= 100:
            start = max(0, len(v15_hist_dc_prob) - 380)
            v15_weight = _fit_total_blend_weight(
                v15_hist_dc_prob[start:],
                v15_hist_baseline_prob[start:],
            )

        v13_base = v13_weight * raw_matrix + (1.0 - v13_weight) * v13_baseline
        v13_base = v13_base / v13_base.sum()
        v13_base_pmf = matrix_to_total_pmf(v13_base)

        v15_base = v15_weight * raw_matrix + (1.0 - v15_weight) * v15_baseline
        v15_base = v15_base / v15_base.sum()
        v15_base_pmf = matrix_to_total_pmf(v15_base)

        v13_beta = 0.0
        if len(v13_hist_base_pmfs) >= 100:
            start = max(0, len(v13_hist_base_pmfs) - 380)
            v13_beta = fit_total_tilt_beta(
                v13_hist_base_pmfs[start:],
                v13_hist_actual_totals[start:],
            )

        v15_beta = 0.0
        if len(v15_hist_base_pmfs) >= 100:
            start = max(0, len(v15_hist_base_pmfs) - 380)
            v15_beta = fit_total_tilt_beta(
                v15_hist_base_pmfs[start:],
                v15_hist_actual_totals[start:],
            )

        v13_matrix = tilt_score_matrix_by_total(v13_base, v13_beta)
        v15_matrix = tilt_score_matrix_by_total(v15_base, v15_beta)

        v13_pmf = matrix_to_total_pmf(v13_matrix)
        v15_pmf = matrix_to_total_pmf(v15_matrix)

        actual_total = int(target["home_goals"] + target["away_goals"])

        match_rows.append(
            {
                "season": target["season"],
                "selected_regime": selected_regime,
                "v13_weight": v13_weight,
                "v15_weight": v15_weight,
                "v13_beta": v13_beta,
                "v15_beta": v15_beta,
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
            naive_pricing = price_total(v13_baseline, line, "over")
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

        dc_actual_prob = _total_probability(raw_matrix, actual_total)
        for label, baseline in baselines.items():
            regime_histories[label].append(
                _total_probability(baseline, actual_total)
            )

        v13_hist_dc_prob.append(dc_actual_prob)
        v13_hist_baseline_prob.append(
            _total_probability(v13_baseline, actual_total)
        )
        v13_hist_base_pmfs.append(v13_base_pmf)
        v13_hist_actual_totals.append(actual_total)

        v15_hist_dc_prob.append(dc_actual_prob)
        v15_hist_baseline_prob.append(
            _total_probability(v15_baseline, actual_total)
        )
        v15_hist_base_pmfs.append(v15_base_pmf)
        v15_hist_actual_totals.append(actual_total)

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
    data: pd.DataFrame,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    summary = summarize_lines(lines)
    v13_brier = float(lines["v13_brier"].mean())
    v15_brier = float(lines["v15_brier"].mean())
    naive_brier = float(lines["naive_brier"].mean())
    regime_counts = Counter(matches["selected_regime"])

    report = [
        f"# GAH v1.5 Adaptive-Regime Totals — {competition}",
        "",
        "Candidate experiment only. Asian handicap is unchanged and excluded.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 | v1.5 candidate | Naive | v1.5 vs v1.3 |",
        "|---|---:|---:|---:|---:|",
        f"| Multi-line Brier ↓ | {v13_brier:.4f} | {v15_brier:.4f} | "
        f"{naive_brier:.4f} | {100.0*(v13_brier-v15_brier)/v13_brier:+.2f}% |",
        f"| Exact-total NLL ↓ | {matches['v13_nll'].mean():.4f} | "
        f"{matches['v15_nll'].mean():.4f} | — | "
        f"{100.0*(matches['v13_nll'].mean()-matches['v15_nll'].mean())/matches['v13_nll'].mean():+.2f}% |",
        f"| Median point MAE ↓ | {matches['v13_point_mae'].mean():.4f} | "
        f"{matches['v15_point_mae'].mean():.4f} | — | "
        f"{100.0*(matches['v13_point_mae'].mean()-matches['v15_point_mae'].mean())/matches['v13_point_mae'].mean():+.2f}% |",
        "",
        f"Lines improved vs v1.3: **{int((summary['v15_brier'] < summary['v13_brier']).sum())}/{len(summary)}**",
        f"Lines beating naive: **{int((summary['v15_brier'] < summary['naive_brier']).sum())}/{len(summary)}**",
        f"Mean v1.5 DC blend weight: **{matches['v15_weight'].mean():.3f}**",
        f"Mean v1.5 tilt beta: **{matches['v15_beta'].mean():+.4f}**",
        "",
        "## Selected scoring regimes",
        "",
    ]

    for label in ["90d", "180d", "365d", "unweighted"]:
        count = int(regime_counts.get(label, 0))
        share = count / len(matches) if len(matches) else 0.0
        report.append(f"- {label}: **{count}** predictions ({share:.1%})")

    report += [
        "",
        "## By line",
        "",
        "| Line | v1.3 Brier | v1.5 Brier | Naive | v1.5 vs v1.3 | v1.5 vs naive | ECE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['v13_brier']:.4f} | "
            f"{row['v15_brier']:.4f} | {row['naive_brier']:.4f} | "
            f"{row['v15_vs_v13_pct']:+.2f}% | "
            f"{row['v15_vs_naive_pct']:+.2f}% | "
            f"{row['ece']:.4f} | {row['bias']:+.4f} |"
        )

    report += [
        "",
        "## Season diagnostic",
        "",
        "| Season | N | v1.3 Brier | v1.5 Brier | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for season, group in lines.groupby("season", sort=True):
        a = float(group["v13_brier"].mean())
        b = float(group["v15_brier"].mean())
        report.append(
            f"| {season} | {int(len(group)/len(TOTAL_LINES))} | "
            f"{a:.4f} | {b:.4f} | {100.0*(a-b)/a:+.2f}% |"
        )

    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_openfootball_league_seasons(args.seasons, competition)
    lines, matches = run_benchmark(data)
    report = build_report(competition, data, lines, matches)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
