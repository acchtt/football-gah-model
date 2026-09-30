from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from gah.calibration import multiclass_brier
from gah.lineups import (
    LineupContinuityState,
    fit_lineup_uncertainty_scale,
    load_statsbomb_matches,
    load_statsbomb_starting_xis,
    margin_probability,
    reshape_score_matrix_by_margin_uncertainty,
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


def _nll(p: float) -> float:
    return -math.log(max(float(p), 1e-12))


def run_benchmark(
    competition_id: int,
    season_id: int,
    min_train: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_statsbomb_matches(competition_id, season_id)
    lineup_cache, lineup_fetch = load_statsbomb_starting_xis(
        data["match_id"].tolist(),
        max_workers=8,
    )
    state = LineupContinuityState(window=10, min_history=5)

    model = None
    last_fit_i = None

    hist_matrices: list[np.ndarray] = []
    hist_disruptions: list[float] = []
    hist_margins: list[int] = []
    gamma = 0.0
    last_gamma_fit_n = 0

    line_rows: list[dict] = []
    match_rows: list[dict] = []

    diagnostics = {
        "matches": len(data),
        "lineup_fetch_errors": int(lineup_fetch["fetch_errors"]),
        "complete_lineup_files": int(lineup_fetch["complete_lineup_files"]),
        "post_warmup_complete": 0,
        "finite_coverage": 0,
    }

    for i, target in data.iterrows():
        xis = lineup_cache.get(int(target["match_id"]), {})
        if len(xis) == 2 and i >= min_train:
            diagnostics["post_warmup_complete"] += 1

        home_xi = xis.get(int(target["home_team_id"]))
        away_xi = xis.get(int(target["away_team_id"]))

        home_coverage = (
            state.coverage(int(target["home_team_id"]), home_xi)
            if home_xi is not None else float("nan")
        )
        away_coverage = (
            state.coverage(int(target["away_team_id"]), away_xi)
            if away_xi is not None else float("nan")
        )

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
                    last_gamma_fit_n == 0
                    or len(hist_matrices) - last_gamma_fit_n >= 20
                )
            ):
                start = max(0, len(hist_matrices) - 200)
                gamma = fit_lineup_uncertainty_scale(
                    hist_matrices[start:],
                    hist_disruptions[start:],
                    hist_margins[start:],
                    bound=0.12,
                )
                last_gamma_fit_n = len(hist_matrices)

            pred = model.predict(target["home_team"], target["away_team"])
            base_matrix = pred["score_matrix"]

            disruption = float(
                1.0 - 0.5 * (float(home_coverage) + float(away_coverage))
            )
            disruption = float(np.clip(disruption, 0.0, 1.0))
            candidate_matrix = reshape_score_matrix_by_margin_uncertainty(
                base_matrix,
                disruption,
                gamma,
            )

            hg = int(target["home_goals"])
            ag = int(target["away_goals"])
            actual_margin = hg - ag
            disrupted = min(home_coverage, away_coverage) < 0.80

            match_rows.append(
                {
                    "match_id": int(target["match_id"]),
                    "disruption": disruption,
                    "disrupted": disrupted,
                    "gamma": gamma,
                    "base_margin_nll": _nll(
                        margin_probability(base_matrix, actual_margin)
                    ),
                    "uncertainty_margin_nll": _nll(
                        margin_probability(candidate_matrix, actual_margin)
                    ),
                    "base_margin_mean": _expected_margin(base_matrix),
                    "candidate_margin_mean": _expected_margin(candidate_matrix),
                }
            )

            for line in AH_LINES:
                base_price = price_handicap(base_matrix, line, "home")
                candidate_price = price_handicap(candidate_matrix, line, "home")
                actual = settle_handicap(actual_margin, line, "home")
                line_rows.append(
                    {
                        "match_id": int(target["match_id"]),
                        "line": line,
                        "disrupted": disrupted,
                        "base_brier": multiclass_brier(base_price, actual),
                        "candidate_brier": multiclass_brier(candidate_price, actual),
                    }
                )

            hist_matrices.append(base_matrix)
            hist_disruptions.append(disruption)
            hist_margins.append(actual_margin)

        if home_xi is not None:
            state.update(int(target["home_team_id"]), home_xi)
        if away_xi is not None:
            state.update(int(target["away_team_id"]), away_xi)

    print("XI_UNCERTAINTY_DIAGNOSTICS", diagnostics)
    return pd.DataFrame(line_rows), pd.DataFrame(match_rows)


def _pct(base: float, candidate: float) -> float:
    return 100.0 * (base - candidate) / base if base else 0.0


def build_report(label: str, lines: pd.DataFrame, matches: pd.DataFrame) -> str:
    if lines.empty or matches.empty:
        return f"# GAH v1.8 Lineup Uncertainty — {label}\n\nNo evaluable rows."

    by_line = (
        lines.groupby("line", sort=True)[["base_brier", "candidate_brier"]]
        .mean()
        .reset_index()
    )
    base_brier = float(lines["base_brier"].mean())
    cand_brier = float(lines["candidate_brier"].mean())
    base_nll = float(matches["base_margin_nll"].mean())
    cand_nll = float(matches["uncertainty_margin_nll"].mean())

    disrupted_lines = lines[lines["disrupted"]]
    disrupted_matches = matches[matches["disrupted"]]

    mean_drift = float(
        np.mean(
            np.abs(
                matches["candidate_margin_mean"].to_numpy()
                - matches["base_margin_mean"].to_numpy()
            )
        )
    )

    report = [
        f"# GAH v1.8 Lineup Uncertainty — {label}",
        "",
        "Confirmed-XI disruption changes only goal-margin dispersion.",
        "The transform is explicitly re-centred to preserve the Dixon-Coles expected margin.",
        "The dispersion scale is fitted only from prior OOS rows.",
        "",
        "## Aggregate",
        "",
        "| Metric | Base DC | XI uncertainty | Change |",
        "|---|---:|---:|---:|",
        f"| Multi-line AH Brier ↓ | {base_brier:.4f} | {cand_brier:.4f} | {_pct(base_brier, cand_brier):+.2f}% |",
        f"| Exact-margin NLL ↓ | {base_nll:.4f} | {cand_nll:.4f} | {_pct(base_nll, cand_nll):+.2f}% |",
        "",
        f"AH lines improved: **{int((by_line['candidate_brier'] < by_line['base_brier']).sum())}/{len(by_line)}**",
        f"Evaluable matches: **{len(matches)}**",
        f"Residual-active matches: **{int((matches['gamma'].abs() > 1e-9).sum())}/{len(matches)}**",
        f"Mean fitted gamma: **{matches['gamma'].mean():+.4f}**",
        f"Mean disruption score: **{matches['disruption'].mean():.3f}**",
        f"Mean absolute expected-margin drift: **{mean_drift:.6f}**",
        "",
        "## Disrupted-lineup segment",
        "",
    ]

    if len(disrupted_lines):
        dbase = float(disrupted_lines["base_brier"].mean())
        dcand = float(disrupted_lines["candidate_brier"].mean())
        dnll0 = float(disrupted_matches["base_margin_nll"].mean())
        dnll1 = float(disrupted_matches["uncertainty_margin_nll"].mean())
        report.extend(
            [
                f"- Matches: **{len(disrupted_matches)}/{len(matches)}**",
                f"- AH Brier: **{dbase:.4f} → {dcand:.4f} ({_pct(dbase, dcand):+.2f}%)**",
                f"- Margin NLL: **{dnll0:.4f} → {dnll1:.4f} ({_pct(dnll0, dnll1):+.2f}%)**",
            ]
        )
    else:
        report.append("- No disrupted-lineup rows.")

    report.extend(
        [
            "",
            "## By line",
            "",
            "| AH line | Base DC | XI uncertainty | Change |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_line.iterrows():
        report.append(
            f"| {row['line']:+.2f} | {row['base_brier']:.4f} | "
            f"{row['candidate_brier']:.4f} | "
            f"{_pct(row['base_brier'], row['candidate_brier']):+.2f}% |"
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
