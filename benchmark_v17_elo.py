from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from gah.calibration import multiclass_brier
from gah.data import load_openfootball_league_seasons
from gah.elo import EloState, fit_elo_margin_scale, tilt_score_matrix_by_margin
from gah.markets import price_handicap, settle_handicap
from gah.model import DixonColesModel


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
AH_LINES = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
]


def expected_margin(matrix: np.ndarray) -> float:
    h, a = np.indices(matrix.shape)
    return float(np.sum(matrix * (h - a)))


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    elo = EloState()
    for _, row in data.iloc[:380].iterrows():
        elo.update(
            row["home_team"],
            row["away_team"],
            int(row["home_goals"]),
            int(row["away_goals"]),
        )

    model = None
    last_fit_i = None

    hist_matrices: list[np.ndarray] = []
    hist_gaps: list[float] = []
    hist_margins: list[int] = []
    alpha = 0.0
    last_alpha_fit_n = 0

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

        home_count = elo.match_count(target["home_team"])
        away_count = elo.match_count(target["away_team"])
        sparse = min(home_count, away_count) < 10
        elo_gap = elo.normalized_gap(target["home_team"], target["away_team"])

        if can_predict:
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

            if (
                len(hist_matrices) >= 200
                and (
                    last_alpha_fit_n == 0
                    or len(hist_matrices) - last_alpha_fit_n >= 20
                )
            ):
                start = max(0, len(hist_matrices) - 760)
                alpha = fit_elo_margin_scale(
                    hist_matrices[start:],
                    hist_gaps[start:],
                    hist_margins[start:],
                    bound=0.25,
                )
                last_alpha_fit_n = len(hist_matrices)

            pred = model.predict(target["home_team"], target["away_team"])
            base_matrix = pred["score_matrix"]
            candidate_matrix = tilt_score_matrix_by_margin(
                base_matrix,
                alpha * elo_gap,
            )

            hg = int(target["home_goals"])
            ag = int(target["away_goals"])
            actual_margin = hg - ag

            match_rows.append(
                {
                    "season": target["season"],
                    "sparse": sparse,
                    "home_elo_matches": home_count,
                    "away_elo_matches": away_count,
                    "elo_gap": elo_gap,
                    "alpha": alpha,
                    "base_margin_mae": abs(
                        expected_margin(base_matrix) - actual_margin
                    ),
                    "elo_margin_mae": abs(
                        expected_margin(candidate_matrix) - actual_margin
                    ),
                }
            )

            for line in AH_LINES:
                base_price = price_handicap(base_matrix, line, "home")
                elo_price = price_handicap(candidate_matrix, line, "home")
                actual = settle_handicap(actual_margin, line, "home")

                line_rows.append(
                    {
                        "season": target["season"],
                        "sparse": sparse,
                        "line": line,
                        "base_brier": multiclass_brier(base_price, actual),
                        "elo_brier": multiclass_brier(elo_price, actual),
                    }
                )

            hist_matrices.append(base_matrix)
            hist_gaps.append(elo_gap)
            hist_margins.append(actual_margin)

        elo.update(
            target["home_team"],
            target["away_team"],
            int(target["home_goals"]),
            int(target["away_goals"]),
        )

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def build_report(
    competition: str,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    rows = []
    for line, group in lines.groupby("line", sort=True):
        base = float(group["base_brier"].mean())
        elo = float(group["elo_brier"].mean())
        rows.append(
            {
                "line": float(line),
                "base": base,
                "elo": elo,
                "change": 100.0 * (base - elo) / base,
            }
        )
    summary = pd.DataFrame(rows)

    base_all = float(lines["base_brier"].mean())
    elo_all = float(lines["elo_brier"].mean())

    sparse_lines = lines[lines["sparse"]]
    sparse_matches = matches[matches["sparse"]]
    if len(sparse_lines):
        sparse_base = float(sparse_lines["base_brier"].mean())
        sparse_elo = float(sparse_lines["elo_brier"].mean())
        sparse_change = 100.0 * (sparse_base - sparse_elo) / sparse_base
    else:
        sparse_base = sparse_elo = sparse_change = float("nan")

    report = [
        f"# GAH v1.7 Elo Prior — {competition}",
        "",
        "Production Dixon-Coles remains the base AH distribution. Elo is a",
        "leakage-safe sequential team-strength prior that applies only a small",
        "goal-margin tilt learned from prior OOS matches.",
        "",
        "## Aggregate",
        "",
        "| Metric | v1.3 AH | v1.7 Elo | Change |",
        "|---|---:|---:|---:|",
        f"| Multi-line AH Brier ↓ | {base_all:.4f} | {elo_all:.4f} | "
        f"{100.0*(base_all-elo_all)/base_all:+.2f}% |",
        f"| Goal-margin MAE ↓ | {matches['base_margin_mae'].mean():.4f} | "
        f"{matches['elo_margin_mae'].mean():.4f} | "
        f"{100.0*(matches['base_margin_mae'].mean()-matches['elo_margin_mae'].mean())/matches['base_margin_mae'].mean():+.2f}% |",
        "",
        f"AH lines improved vs v1.3: **{int((summary['elo'] < summary['base']).sum())}/{len(summary)}**",
        f"Mean fitted Elo scale alpha: **{matches['alpha'].mean():+.4f}**",
        "",
        "## Sparse-team segment",
        "",
        "Sparse means at least one side had fewer than 10 prior top-flight Elo matches.",
        "",
        f"- Sparse predictions: **{len(sparse_matches)}/{len(matches)}**",
        f"- Sparse AH Brier: **{sparse_base:.4f} → {sparse_elo:.4f} ({sparse_change:+.2f}%)**",
        f"- Sparse margin MAE: **{sparse_matches['base_margin_mae'].mean():.4f} → "
        f"{sparse_matches['elo_margin_mae'].mean():.4f}**"
        if len(sparse_matches) else "- Sparse margin MAE: n/a",
        "",
        "## By line",
        "",
        "| AH line | v1.3 | v1.7 Elo | Change |",
        "|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values("line").iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['base']:.4f} | "
            f"{row['elo']:.4f} | {row['change']:+.2f}% |"
        )

    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_openfootball_league_seasons(args.seasons, competition)
    lines, matches = run_benchmark(data)
    print(build_report(competition, lines, matches))


if __name__ == "__main__":
    main()
