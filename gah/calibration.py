from __future__ import annotations

import numpy as np

from .markets import SETTLEMENT_CATEGORIES


SETTLEMENT_VALUE = {
    "full_win": 1.0,
    "half_win": 0.75,
    "push": 0.50,
    "half_loss": 0.25,
    "full_loss": 0.0,
}


def settlement_probability_vector(pricing: dict) -> np.ndarray:
    p = np.array(
        [float(pricing.get(category, 0.0)) for category in SETTLEMENT_CATEGORIES],
        dtype=float,
    )
    total = float(p.sum())
    if total <= 0:
        raise ValueError("Settlement probabilities must contain positive mass.")
    return p / total


def multiclass_brier(pricing: dict, actual_category: str) -> float:
    """
    Proper five-outcome Brier score for Asian settlement.

    Outcomes:
      full win / half win / push / half loss / full loss.
    Lower is better.
    """
    if actual_category not in SETTLEMENT_CATEGORIES:
        raise ValueError(f"Unknown settlement category: {actual_category}")

    p = settlement_probability_vector(pricing)
    y = np.zeros(len(SETTLEMENT_CATEGORIES), dtype=float)
    y[SETTLEMENT_CATEGORIES.index(actual_category)] = 1.0
    return float(np.square(p - y).sum())


def expected_settlement_value(pricing: dict) -> float:
    """Expected normalized settlement result in [0, 1]."""
    return float(
        sum(
            float(pricing.get(category, 0.0)) * SETTLEMENT_VALUE[category]
            for category in SETTLEMENT_CATEGORIES
        )
    )


def observed_settlement_value(actual_category: str) -> float:
    if actual_category not in SETTLEMENT_VALUE:
        raise ValueError(f"Unknown settlement category: {actual_category}")
    return SETTLEMENT_VALUE[actual_category]


def expected_calibration_error(
    predicted: list[float] | np.ndarray,
    observed: list[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Equal-width calibration error for normalized settlement value.

    This is descriptive rather than a replacement for the proper multiclass
    Brier score.
    """
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    if len(p) != len(y):
        raise ValueError("predicted and observed must have equal length.")
    if len(p) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        error += (n / len(p)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def calibration_bias(
    predicted: list[float] | np.ndarray,
    observed: list[float] | np.ndarray,
) -> float:
    """Mean predicted normalized settlement minus mean observed settlement."""
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    if len(p) != len(y):
        raise ValueError("predicted and observed must have equal length.")
    if len(p) == 0:
        return 0.0
    return float(p.mean() - y.mean())


def use_model_by_paired_loss(
    model_losses: list[float] | np.ndarray,
    baseline_losses: list[float] | np.ndarray,
    min_history: int = 100,
    z_threshold: float = 1.0,
    window: int | None = 380,
) -> bool:
    """
    Conservative sequential reliability gate.

    Keep the model unless prior paired out-of-sample losses show it is worse
    than the baseline by more than z_threshold standard errors. Losses must be
    one aggregate value per historical prediction, not repeated line rows.
    """
    model = np.asarray(model_losses, dtype=float)
    baseline = np.asarray(baseline_losses, dtype=float)
    if model.shape != baseline.shape:
        raise ValueError("model_losses and baseline_losses must have equal shape.")
    if len(model) < int(min_history):
        return True

    if window is not None:
        start = max(0, len(model) - int(window))
        model = model[start:]
        baseline = baseline[start:]

    diff = model - baseline
    mean_diff = float(diff.mean())
    if len(diff) < 2:
        return mean_diff <= 0.0

    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    if se <= 1e-15:
        return mean_diff <= 0.0

    return not (mean_diff > float(z_threshold) * se)
