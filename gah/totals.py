from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def matrix_to_total_pmf(matrix: np.ndarray) -> np.ndarray:
    """Collapse a home-away score matrix into a total-goals PMF."""
    rows, cols = matrix.shape
    pmf = np.zeros(rows + cols - 1, dtype=float)
    for hg in range(rows):
        for ag in range(cols):
            pmf[hg + ag] += float(matrix[hg, ag])
    total = pmf.sum()
    if total <= 0:
        raise ValueError("Invalid score matrix.")
    return pmf / total


def poisson_total_pmf(mean: float, max_goals: int = 20) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    pmf = poisson.pmf(goals, max(float(mean), 1e-9))
    return pmf / pmf.sum()


def negative_binomial_total_pmf(
    mean: float,
    variance: float,
    max_goals: int = 20,
) -> np.ndarray:
    """
    Negative-binomial PMF parameterized by mean/variance.

    Falls back to Poisson when empirical variance does not exceed the mean.
    """
    mean = max(float(mean), 1e-9)
    variance = float(variance)
    if variance <= mean + 1e-9:
        return poisson_total_pmf(mean, max_goals=max_goals)

    # scipy nbinom: mean = n * (1-p)/p; var = n * (1-p)/p^2
    p = mean / variance
    n = mean * p / (1.0 - p)

    goals = np.arange(max_goals + 1)
    pmf = nbinom.pmf(goals, n, p)
    return pmf / pmf.sum()


def pmf_probability(pmf: np.ndarray, goals: int) -> float:
    if goals < 0 or goals >= len(pmf):
        return 0.0
    return float(pmf[goals])


def pmf_mean(pmf: np.ndarray) -> float:
    goals = np.arange(len(pmf), dtype=float)
    return float(np.dot(goals, pmf))


def absolute_error_optimal_point(pmf: np.ndarray) -> float:
    """
    Bayes-optimal point forecast under absolute-error loss.

    For a discrete count distribution this is a predictive median, not the mean.
    """
    cdf = np.cumsum(pmf)
    return float(np.searchsorted(cdf, 0.5, side="left"))


def over_probability(pmf: np.ndarray, line: float) -> float:
    """
    Full-win probability for non-quarter binary checks such as Over 2.5.
    Intended for calibration metrics, not Asian quarter-line settlement.
    """
    threshold = math.floor(line) + 1
    return float(pmf[threshold:].sum())


def mix_pmfs(first: np.ndarray, second: np.ndarray, weight_first: float) -> np.ndarray:
    n = max(len(first), len(second))
    a = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)
    a[:len(first)] = first
    b[:len(second)] = second

    w = float(np.clip(weight_first, 0.0, 1.0))
    out = w * a + (1.0 - w) * b
    return out / out.sum()


def recency_weighted_total_stats(
    train: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_days: float = 180.0,
    recent_matches: int | None = 760,
) -> tuple[float, float]:
    """
    Leakage-safe scoring-regime estimate from historical match totals.

    Optionally cap the history before exponential time weighting so ancient
    scoring environments have negligible influence even in long datasets.
    """
    hist = train.sort_values("match_date")
    if recent_matches is not None:
        hist = hist.tail(int(recent_matches))

    dates = pd.to_datetime(hist["match_date"], utc=True)
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    age_days = (as_of - dates).dt.total_seconds().to_numpy() / 86400.0
    age_days = np.maximum(age_days, 0.0)
    weights = np.exp(-np.log(2.0) * age_days / float(half_life_days))

    totals = (
        hist["home_goals"].to_numpy(dtype=float)
        + hist["away_goals"].to_numpy(dtype=float)
    )
    wsum = float(weights.sum())
    if wsum <= 0:
        raise ValueError("No positive regime weights.")

    mean = float(np.dot(weights, totals) / wsum)
    variance = float(np.dot(weights, np.square(totals - mean)) / wsum)
    return mean, variance
