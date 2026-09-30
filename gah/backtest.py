from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .model import DixonColesModel
from .markets import price_total, price_handicap


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

    For each prediction, training data contains only earlier matches.
    The model is refitted every `refit_every` predictions for practicality.
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

        # Skip if target contains a team not yet observed.
        known = set(train["home_team"]).union(train["away_team"])
        if target["home_team"] not in known or target["away_team"] not in known:
            continue

        if model is None or last_fit_i is None or (i - last_fit_i) >= refit_every:
            model = DixonColesModel(half_life_days=half_life_days)
            model.fit(train, as_of=train["match_date"].max())
            last_fit_i = i

        try:
            pred = model.predict(target["home_team"], target["away_team"])
        except KeyError:
            continue

        matrix = pred["score_matrix"]
        hg = int(target["home_goals"])
        ag = int(target["away_goals"])

        p_score = float(matrix[min(hg, matrix.shape[0]-1), min(ag, matrix.shape[1]-1)])
        score_nll = -math.log(max(p_score, 1e-12))

        over = price_total(matrix, total_line, "over")
        home_ah = price_handicap(matrix, handicap_line, "home")

        rows.append({
            "match_date": target["match_date"],
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "home_goals": hg,
            "away_goals": ag,
            "xg_home": pred["xg_home"],
            "xg_away": pred["xg_away"],
            "xg_total": pred["xg_total"],
            "expected_margin": pred["expected_margin"],
            "actual_total": hg + ag,
            "actual_margin": hg - ag,
            "total_abs_error": abs(pred["xg_total"] - (hg + ag)),
            "margin_abs_error": abs(pred["expected_margin"] - (hg - ag)),
            "score_nll": score_nll,
            f"over_{total_line}_fair_odds": over["fair_decimal_odds"],
            f"home_{handicap_line}_fair_odds": home_ah["fair_decimal_odds"],
        })

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"n_predictions": 0}

    return {
        "n_predictions": int(len(results)),
        "mean_total_abs_error": float(results["total_abs_error"].mean()),
        "mean_margin_abs_error": float(results["margin_abs_error"].mean()),
        "mean_score_nll": float(results["score_nll"].mean()),
    }
