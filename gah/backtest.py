from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .model import DixonColesModel
from .markets import price_total, price_handicap


def _independent_poisson_matrix(home_rate: float, away_rate: float, max_goals: int = 10) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    return matrix / matrix.sum()


def _matrix_score_nll(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    if home_goals >= matrix.shape[0] or away_goals >= matrix.shape[1]:
        return -math.log(1e-12)
    p = float(matrix[home_goals, away_goals])
    return -math.log(max(p, 1e-12))


def walk_forward_backtest(
    df: pd.DataFrame,
    half_life_days: float = 180.0,
    min_train_matches: int = 250,
    refit_every: int = 50,
    total_line: float = 2.5,
    handicap_line: float = -0.5,
) -> pd.DataFrame:
    """
    Expanding-window evaluation.

    For every prediction, training data contains only earlier matches.
    The model is refitted on a fixed cadence and also when a team has appeared
    since the previous fit but is not yet represented in model parameters.
    """
    data = df.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    rows = []
    model = None
    last_fit_i = None

    for i in range(min_train_matches, len(data)):
        target = data.iloc[i]
        train = data.iloc[:i]

        # A promoted/new team needs at least one historical match before GAH can
        # estimate an attack and defence parameter for it.
        known = set(train["home_team"]).union(train["away_team"])
        if target["home_team"] not in known or target["away_team"] not in known:
            continue

        needs_refit = (
            model is None
            or last_fit_i is None
            or (i - last_fit_i) >= refit_every
            or target["home_team"] not in model.team_to_idx_
            or target["away_team"] not in model.team_to_idx_
        )
        if needs_refit:
            model = DixonColesModel(half_life_days=half_life_days, max_goals=10)
            model.fit(train, as_of=train["match_date"].max())
            last_fit_i = i

        pred = model.predict(target["home_team"], target["away_team"])
        matrix = pred["score_matrix"]
        hg = int(target["home_goals"])
        ag = int(target["away_goals"])

        score_nll = _matrix_score_nll(matrix, hg, ag)

        over = price_total(matrix, total_line, "over")
        home_ah = price_handicap(matrix, handicap_line, "home")

        # Simple league-average comparator computed from training data only.
        naive_home_rate = float(train["home_goals"].mean())
        naive_away_rate = float(train["away_goals"].mean())
        naive_matrix = _independent_poisson_matrix(
            naive_home_rate,
            naive_away_rate,
            max_goals=10,
        )
        naive_score_nll = _matrix_score_nll(naive_matrix, hg, ag)
        naive_over = price_total(naive_matrix, total_line, "over")
        naive_home_ah = price_handicap(naive_matrix, handicap_line, "home")

        actual_total = hg + ag
        actual_margin = hg - ag
        over_label = 1.0 if actual_total > total_line else 0.0
        home_ah_label = 1.0 if (actual_margin + handicap_line) > 0 else 0.0

        row = {
            "match_date": target["match_date"],
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "home_goals": hg,
            "away_goals": ag,
            "xg_home": pred["xg_home"],
            "xg_away": pred["xg_away"],
            "xg_total": pred["xg_total"],
            "expected_margin": pred["expected_margin"],
            "actual_total": actual_total,
            "actual_margin": actual_margin,
            "total_abs_error": abs(pred["xg_total"] - actual_total),
            "margin_abs_error": abs(pred["expected_margin"] - actual_margin),
            "score_nll": score_nll,
            "naive_xg_home": naive_home_rate,
            "naive_xg_away": naive_away_rate,
            "naive_xg_total": naive_home_rate + naive_away_rate,
            "naive_expected_margin": naive_home_rate - naive_away_rate,
            "naive_total_abs_error": abs((naive_home_rate + naive_away_rate) - actual_total),
            "naive_margin_abs_error": abs((naive_home_rate - naive_away_rate) - actual_margin),
            "naive_score_nll": naive_score_nll,
            f"over_{total_line}_prob": over["full_win"],
            f"naive_over_{total_line}_prob": naive_over["full_win"],
            f"over_{total_line}_brier": (over["full_win"] - over_label) ** 2,
            f"naive_over_{total_line}_brier": (naive_over["full_win"] - over_label) ** 2,
            f"home_{handicap_line}_prob": home_ah["full_win"],
            f"naive_home_{handicap_line}_prob": naive_home_ah["full_win"],
            f"home_{handicap_line}_brier": (home_ah["full_win"] - home_ah_label) ** 2,
            f"naive_home_{handicap_line}_brier": (naive_home_ah["full_win"] - home_ah_label) ** 2,
            f"over_{total_line}_fair_odds": over["fair_decimal_odds"],
            f"home_{handicap_line}_fair_odds": home_ah["fair_decimal_odds"],
        }

        for optional in ("season", "competition"):
            if optional in target.index:
                row[optional] = target[optional]

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"n_predictions": 0}

    return {
        "n_predictions": int(len(results)),
        "mean_total_abs_error": float(results["total_abs_error"].mean()),
        "mean_naive_total_abs_error": float(results["naive_total_abs_error"].mean()),
        "mean_margin_abs_error": float(results["margin_abs_error"].mean()),
        "mean_naive_margin_abs_error": float(results["naive_margin_abs_error"].mean()),
        "mean_score_nll": float(results["score_nll"].mean()),
        "mean_naive_score_nll": float(results["naive_score_nll"].mean()),
        "mean_over_25_brier": float(results["over_2.5_brier"].mean()),
        "mean_naive_over_25_brier": float(results["naive_over_2.5_brier"].mean()),
        "mean_home_m05_brier": float(results["home_-0.5_brier"].mean()),
        "mean_naive_home_m05_brier": float(results["naive_home_-0.5_brier"].mean()),
    }
