from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .model import DixonColesModel
from .markets import price_total, price_handicap
from .totals import absolute_error_optimal_point, matrix_to_total_pmf


def _independent_poisson_matrix(home_rate: float, away_rate: float, max_goals: int = 10) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    return matrix / matrix.sum()


def _matrix_score_nll(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    if home_goals >= matrix.shape[0] or away_goals >= matrix.shape[1]:
        return -math.log(1e-12)
    p = float(matrix[home_goals, away_goals])
    return -math.log(max(p, 1e-12))


def _total_probability(matrix: np.ndarray, total_goals: int) -> float:
    rows, cols = matrix.shape
    p = 0.0
    for hg in range(rows):
        ag = total_goals - hg
        if 0 <= ag < cols:
            p += float(matrix[hg, ag])
    return p


def _fit_total_blend_weight(
    model_actual_probs: list[float],
    naive_actual_probs: list[float],
    grid_size: int = 101,
) -> float:
    """
    Choose the mixture weight that maximizes historical total-goal likelihood.

    Only historical out-of-sample predictions should be supplied here.
    weight=1.0 means pure GAH; weight=0.0 means pure league baseline.
    """
    if not model_actual_probs:
        return 1.0
    if len(model_actual_probs) != len(naive_actual_probs):
        raise ValueError("Probability histories must have equal length.")

    pm = np.asarray(model_actual_probs, dtype=float)
    pn = np.asarray(naive_actual_probs, dtype=float)
    weights = np.linspace(0.0, 1.0, grid_size)

    best_weight = 1.0
    best_nll = math.inf
    for w in weights:
        mixed = np.clip(w * pm + (1.0 - w) * pn, 1e-12, 1.0)
        nll = float(-np.log(mixed).mean())
        if nll < best_nll:
            best_nll = nll
            best_weight = float(w)

    return best_weight


def walk_forward_backtest(
    df: pd.DataFrame,
    half_life_days: float = 180.0,
    min_train_matches: int = 250,
    refit_every: int = 50,
    total_line: float = 2.5,
    handicap_line: float = -0.5,
    total_blend: bool = False,
    total_blend_min_history: int = 100,
    total_blend_window: int = 380,
) -> pd.DataFrame:
    """
    Expanding-window evaluation.

    For every prediction, training data contains only earlier matches.
    The model is refitted on a fixed cadence and also when a team has appeared
    since the previous fit but is not yet represented in model parameters.

    When total_blend=True, the totals engine uses an online mixture between:
      1) the team-specific Dixon-Coles score distribution, and
      2) the training-window league-average independent-Poisson distribution.

    The blend weight is fitted ONLY from earlier out-of-sample predictions.
    The Asian-handicap engine always remains pure Dixon-Coles.
    """
    data = df.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    rows = []
    model = None
    last_fit_i = None

    # Historical OOS likelihood contributions used by the totals calibrator.
    total_hist_model_prob: list[float] = []
    total_hist_naive_prob: list[float] = []

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

        # League-average comparator computed from training data only.
        naive_home_rate = float(train["home_goals"].mean())
        naive_away_rate = float(train["away_goals"].mean())
        naive_matrix = _independent_poisson_matrix(
            naive_home_rate,
            naive_away_rate,
            max_goals=10,
        )

        blend_weight = 1.0
        if total_blend and len(total_hist_model_prob) >= total_blend_min_history:
            start = max(0, len(total_hist_model_prob) - total_blend_window)
            blend_weight = _fit_total_blend_weight(
                total_hist_model_prob[start:],
                total_hist_naive_prob[start:],
            )

        total_matrix = (
            blend_weight * matrix + (1.0 - blend_weight) * naive_matrix
            if total_blend
            else matrix
        )
        total_matrix = total_matrix / total_matrix.sum()

        # Current outcome is used only for evaluation and for FUTURE calibration.
        hg = int(target["home_goals"])
        ag = int(target["away_goals"])
        actual_total = hg + ag
        actual_margin = hg - ag

        score_nll = _matrix_score_nll(matrix, hg, ag)
        naive_score_nll = _matrix_score_nll(naive_matrix, hg, ag)

        model_total_prob = _total_probability(matrix, actual_total)
        naive_total_prob = _total_probability(naive_matrix, actual_total)
        blended_total_prob = _total_probability(total_matrix, actual_total)

        model_total_nll = -math.log(max(model_total_prob, 1e-12))
        naive_total_nll = -math.log(max(naive_total_prob, 1e-12))
        blended_total_nll = -math.log(max(blended_total_prob, 1e-12))

        over = price_total(matrix, total_line, "over")
        blended_over = price_total(total_matrix, total_line, "over")
        home_ah = price_handicap(matrix, handicap_line, "home")

        naive_over = price_total(naive_matrix, total_line, "over")
        naive_home_ah = price_handicap(naive_matrix, handicap_line, "home")

        over_label = 1.0 if actual_total > total_line else 0.0
        home_ah_label = 1.0 if (actual_margin + handicap_line) > 0 else 0.0

        naive_xg_total = naive_home_rate + naive_away_rate
        blended_xg_total = (
            blend_weight * pred["xg_total"]
            + (1.0 - blend_weight) * naive_xg_total
        )
        blended_total_point = absolute_error_optimal_point(
            matrix_to_total_pmf(total_matrix)
        )

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
            "total_nll": model_total_nll,
            "naive_xg_home": naive_home_rate,
            "naive_xg_away": naive_away_rate,
            "naive_xg_total": naive_xg_total,
            "naive_expected_margin": naive_home_rate - naive_away_rate,
            "naive_total_abs_error": abs(naive_xg_total - actual_total),
            "naive_margin_abs_error": abs((naive_home_rate - naive_away_rate) - actual_margin),
            "naive_score_nll": naive_score_nll,
            "naive_total_nll": naive_total_nll,
            "total_blend_weight": blend_weight,
            "blended_xg_total": blended_xg_total,
            "blended_total_abs_error": abs(blended_xg_total - actual_total),
            "blended_total_point": blended_total_point,
            "blended_total_point_abs_error": abs(blended_total_point - actual_total),
            "blended_total_nll": blended_total_nll,
            f"over_{total_line}_prob": over["full_win"],
            f"naive_over_{total_line}_prob": naive_over["full_win"],
            f"blended_over_{total_line}_prob": blended_over["full_win"],
            f"over_{total_line}_brier": (over["full_win"] - over_label) ** 2,
            f"naive_over_{total_line}_brier": (naive_over["full_win"] - over_label) ** 2,
            f"blended_over_{total_line}_brier": (blended_over["full_win"] - over_label) ** 2,
            f"home_{handicap_line}_prob": home_ah["full_win"],
            f"naive_home_{handicap_line}_prob": naive_home_ah["full_win"],
            f"home_{handicap_line}_brier": (home_ah["full_win"] - home_ah_label) ** 2,
            f"naive_home_{handicap_line}_brier": (naive_home_ah["full_win"] - home_ah_label) ** 2,
            f"over_{total_line}_fair_odds": over["fair_decimal_odds"],
            f"blended_over_{total_line}_fair_odds": blended_over["fair_decimal_odds"],
            f"home_{handicap_line}_fair_odds": home_ah["fair_decimal_odds"],
        }

        for optional in ("season", "competition"):
            if optional in target.index:
                row[optional] = target[optional]

        rows.append(row)

        # Append current OOS evaluation only AFTER the prediction has been made.
        total_hist_model_prob.append(model_total_prob)
        total_hist_naive_prob.append(naive_total_prob)

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"n_predictions": 0}

    summary = {
        "n_predictions": int(len(results)),
        "mean_total_abs_error": float(results["total_abs_error"].mean()),
        "mean_naive_total_abs_error": float(results["naive_total_abs_error"].mean()),
        "mean_margin_abs_error": float(results["margin_abs_error"].mean()),
        "mean_naive_margin_abs_error": float(results["naive_margin_abs_error"].mean()),
        "mean_score_nll": float(results["score_nll"].mean()),
        "mean_naive_score_nll": float(results["naive_score_nll"].mean()),
        "mean_total_nll": float(results["total_nll"].mean()),
        "mean_naive_total_nll": float(results["naive_total_nll"].mean()),
        "mean_over_25_brier": float(results["over_2.5_brier"].mean()),
        "mean_naive_over_25_brier": float(results["naive_over_2.5_brier"].mean()),
        "mean_home_m05_brier": float(results["home_-0.5_brier"].mean()),
        "mean_naive_home_m05_brier": float(results["naive_home_-0.5_brier"].mean()),
    }

    if "blended_total_abs_error" in results:
        summary.update(
            {
                "mean_blended_total_abs_error": float(results["blended_total_abs_error"].mean()),
                "mean_blended_total_point_abs_error": float(results["blended_total_point_abs_error"].mean()),
                "mean_blended_total_nll": float(results["blended_total_nll"].mean()),
                "mean_blended_over_25_brier": float(results["blended_over_2.5_brier"].mean()),
                "mean_total_blend_weight": float(results["total_blend_weight"].mean()),
            }
        )

    return summary


def fit_next_total_blend_weight(
    df: pd.DataFrame,
    half_life_days: float = 180.0,
    min_train_matches: int = 380,
    refit_every: int = 20,
    min_history: int = 100,
    window: int = 380,
) -> float:
    """
    Fit the totals blend weight for the NEXT match using only completed history.

    Historical component likelihoods come from walk-forward predictions, so the
    calibration itself never evaluates a match with information from its future.
    """
    if len(df) <= min_train_matches:
        return 1.0

    results = walk_forward_backtest(
        df,
        half_life_days=half_life_days,
        min_train_matches=min_train_matches,
        refit_every=refit_every,
        total_blend=False,
    )
    if len(results) < min_history:
        return 1.0

    model_probs = np.exp(-results["total_nll"].to_numpy(dtype=float))
    naive_probs = np.exp(-results["naive_total_nll"].to_numpy(dtype=float))
    start = max(0, len(results) - window)
    return _fit_total_blend_weight(
        model_probs[start:].tolist(),
        naive_probs[start:].tolist(),
    )
