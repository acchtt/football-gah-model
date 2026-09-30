from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gah.calibration import multiclass_brier
from gah.lineups import (
    LineupContinuityState,
    fit_lineup_margin_scale,
    load_statsbomb_matches,
    load_statsbomb_starting_xi,
    tilt_score_matrix_by_margin,
)
from gah.markets import price_handicap, settle_handicap
from gah.model import DixonColesModel


AH_LINES = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
]


def _expected_margin(matrix: np.ndarray) -> float:
    h, a = np.indices(matrix.shape)
    return float(np.sum(matrix * (h - a)))


def run_benchmark(
    competition_id: int,
    season_id: int,
    min_train: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_statsbomb_matches(competition_id, season_id)
    state = LineupContinuityState(window=10, min_history=5)

    model = None
    last_fit_i = None

    hist_matrices: list[np.ndarray] = []
    hist_gaps: list[float] = []
    hist_margins: list[int] = []
    alpha = 0.0
    last_alpha_fit_n = 0

    line_rows: list[dict] = []
    match_rows: list[dict] = []
    diagnostics = {
        "matches": len(data),
        "lineup_fetch_errors": 0,
        "complete_lineup_files": 0,
        "post_warmup_complete": 0,
        "finite_coverage": 0,
    }
    first_errors: list[str] = []

    for i, target in data.iterrows():
        try:
            xis = load_statsbomb_starting_xi(int(target["match_id"]))
        except Exception as exc:
            diagnostics["lineup_fetch_errors"] += 1
            if len(first_errors) < 3:
                first_errors.append(
                    f'{int(target["match_id"])}: {type(exc).__name__}: {exc}'
                )
            xis = {}

        if len(xis) == 2:
            diagnostics["complete_lineup_files"] += 1
            if i >= min_train:
                diagnostics["post_warmup_complete"] += 1

        home_xi = xis.get(int(target["home_team_id"]))
        away_xi = xis.get(int(target["away_team_id"]))

        if home_xi is not None:
            home_coverage = state.coverage(int(target["home_team_id"]), home_xi)
        else:
            home_coverage = float("nan")
        if away_xi is not None:
            away_coverage = state.coverage(int(target["away_team_id"]), away_xi)
        else:
            away_coverage = float("nan")

        if i >= min_train and np.isfinite(home_coverage) and np.isfinite(away_coverage):
            diagnostics["finite_coverage"] += 1
            train = data.iloc[:i]

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
                len(hist_matrices) >= 50
                and (
                    last_alpha_fit_n == 0
                    or len(hist_matrices) - last_alpha_fit_n >= 20
                )
            ):
                start = max(0, len(hist_matrices) - 200)
                alpha = fit_lineup_margin_scale(
                    hist_matrices[start:],
                    hist_gaps[start:],
                    hist_margins[start:],
                    bound=0.35,
                )
                last_alpha_fit_n = len(hist_matrices)

            pred = model.predict(target["home_team"], target["away_team"])
            base_matrix = pred["score_matrix"]
            coverage_gap = float(home_coverage - away_coverage)
            lineup_matrix = tilt_score_matrix_by_margin(
                base_matrix,
                alpha * coverage_gap,
            )

            hg = int(target["home_goals"])
            ag = int(target["away_goals"])
            actual_margin = hg - ag
            disrupted = min(home_coverage, away_coverage) < 0.80

            match_rows.append(
                {
                    "match_id": int(target["match_id"]),
                    "match_date": target["match_date"],
                    "home_coverage": float(home_coverage),
                    "away_coverage": float(away_coverage),
                    "coverage_gap": coverage_gap,
                    "disrupted": disrupted,
                    "alpha": alpha,
                    "base_margin_mae": abs(
                        _expected_margin(base_matrix) - actual_margin
                    ),
                    "lineup_margin_mae": abs(
                        _expected_margin(lineup_matrix) - actual_margin
                    ),
                }
            )

            for line in AH_LINES:
                base_price = price_handicap(base_matrix, line, "home")
                lineup_price = price_handicap(lineup_matrix, line, "home")
                actual = settle_handicap(actual_margin, line, "home")
                line_rows.append(
                    {
                        "match_id": int(target["match_id"]),
                        "line": line,
                        "disrupted": disrupted,
                        "base_brier": multiclass_brier(base_price, actual),
                        "lineup_brier": multiclass_brier(lineup_price, actual),
                    }
                )

            hist_matrices.append(base_matrix)
            hist_gaps.append(coverage_gap)
            hist_margins.append(actual_margin)

        if home_xi is not None:
            state.update(int(target["home_team_id"]), home_xi)
        if away_xi is not None:
            state.update(int(target["away_team_id"]), away_xi)

    print("XI_DIAGNOSTICS", diagnostics)
    if first_errors:
        print("XI_FIRST_ERRORS")
        for error in first_errors:
            print(error)

    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def _pct_change(base: float, candidate: float) -> float:
    return 100.0 * (base - candidate) / base if base else 0.0


def build_report(
    label: str,
    lines: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    if lines.empty or matches.empty:
        return f"# GAH v1.8 Confirmed-XI Continuity — {label}\n\nNo evaluable rows."

    by_line = (
        lines.groupby("line", sort=True)[["base_brier", "lineup_brier"]]
        .mean()
        .reset_index()
    )
    base_all = float(lines["base_brier"].mean())
    lineup_all = float(lines["lineup_brier"].mean())
    base_mae = float(matches["base_margin_mae"].mean())
    lineup_mae = float(matches["lineup_margin_mae"].mean())

    disrupted_lines = lines[lines["disrupted"]]
    disrupted_matches = matches[matches["disrupted"]]

    report = [
        f"# GAH v1.8 Confirmed-XI Continuity — {label}",
        "",
        "Confirmed lineups are compared with each team's prior 10 starting XIs.",
        "Coverage measures how much of the recent core-starter mass is present.",
        "Only prior lineups/outcomes fit the adjustment; Dixon-Coles is unchanged.",
        "",
        "## Aggregate",
        "",
        "| Metric | Base DC | v1.8 XI | Change |",
        "|---|---:|---:|---:|",
        f"| Multi-line AH Brier ↓ | {base_all:.4f} | {lineup_all:.4f} | "
        f"{_pct_change(base_all, lineup_all):+.2f}% |",
        f"| Goal-margin MAE ↓ | {base_mae:.4f} | {lineup_mae:.4f} | "
        f"{_pct_change(base_mae, lineup_mae):+.2f}% |",
        "",
        f"AH lines improved: **{int((by_line['lineup_brier'] < by_line['base_brier']).sum())}/{len(by_line)}**",
        f"Evaluable confirmed-XI matches: **{len(matches)}**",
        f"Residual-active matches: **{int((matches['alpha'].abs() > 1e-9).sum())}/{len(matches)}**",
        f"Mean fitted alpha: **{matches['alpha'].mean():+.4f}**",
        f"Mean home XI coverage: **{matches['home_coverage'].mean():.3f}**",
        f"Mean away XI coverage: **{matches['away_coverage'].mean():.3f}**",
        "",
        "## Disrupted-lineup segment",
        "",
        "Disrupted means at least one team's confirmed XI has <80% recent-core coverage.",
        "",
    ]

    if len(disrupted_lines):
        dbase = float(disrupted_lines["base_brier"].mean())
        dxi = float(disrupted_lines["lineup_brier"].mean())
        report.extend(
            [
                f"- Matches: **{len(disrupted_matches)}/{len(matches)}**",
                f"- AH Brier: **{dbase:.4f} → {dxi:.4f} ({_pct_change(dbase, dxi):+.2f}%)**",
                f"- Margin MAE: **{disrupted_matches['base_margin_mae'].mean():.4f} → "
                f"{disrupted_matches['lineup_margin_mae'].mean():.4f}**",
            ]
        )
    else:
        report.append("- No disrupted-lineup rows.")

    report.extend(
        [
            "",
            "## By line",
            "",
            "| AH line | Base DC | v1.8 XI | Change |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_line.iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['base_brier']:.4f} | "
            f"{row['lineup_brier']:.4f} | "
            f"{_pct_change(row['base_brier'], row['lineup_brier']):+.2f}% |"
        )
    return "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition-id", type=int, required=True)
    ap.add_argument("--season-id", type=int, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    lines, matches = run_benchmark(args.competition_id, args.season_id)
    print(build_report(args.label, lines, matches))


if __name__ == "__main__":
    main()
