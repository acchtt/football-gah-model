from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def devig_two_way(decimal_a: float, decimal_b: float) -> tuple[float, float]:
    """Proportional de-vig for a two-way market."""
    a = float(decimal_a)
    b = float(decimal_b)
    if not (math.isfinite(a) and math.isfinite(b) and a > 1.0 and b > 1.0):
        raise ValueError("Two valid decimal odds greater than 1 are required.")

    qa = 1.0 / a
    qb = 1.0 / b
    z = qa + qb
    return qa / z, qb / z


def model_effective_win_probability(pricing: dict) -> float:
    """
    Convert exact Asian settlement probabilities to the same two-way exposure
    space priced by bookmakers.

    Push stake is excluded; half-win/half-loss count as half active stake.
    """
    win = float(pricing.get("full_win", 0.0)) + 0.5 * float(
        pricing.get("half_win", 0.0)
    )
    loss = float(pricing.get("full_loss", 0.0)) + 0.5 * float(
        pricing.get("half_loss", 0.0)
    )
    active = win + loss
    if active <= 0:
        raise ValueError("Pricing has no active win/loss mass.")
    return win / active


def realized_exposure(actual_category: str) -> tuple[float, float] | None:
    """
    Return (binary outcome, active stake weight) for Asian settlement.

    Full win/loss carries weight 1; half win/loss carries weight 0.5;
    push has no active stake and is omitted from binary market scoring.
    """
    mapping = {
        "full_win": (1.0, 1.0),
        "half_win": (1.0, 0.5),
        "push": None,
        "half_loss": (0.0, 0.5),
        "full_loss": (0.0, 1.0),
    }
    if actual_category not in mapping:
        raise ValueError(f"Unknown settlement category: {actual_category}")
    return mapping[actual_category]


def blend_probability(
    pure_probability: float,
    market_probability: float,
    market_weight: float,
) -> float:
    w = float(np.clip(market_weight, 0.0, 1.0))
    p = (1.0 - w) * float(pure_probability) + w * float(market_probability)
    return float(np.clip(p, 1e-9, 1.0 - 1e-9))


def weighted_binary_brier(
    probabilities: list[float] | np.ndarray,
    outcomes: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (len(p) == len(y) == len(w)):
        raise ValueError("probabilities, outcomes and weights must align.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("At least one positive exposure weight is required.")
    return float(np.sum(w * np.square(p - y)) / total)


def fit_market_blend_weight(
    pure_probabilities: list[float] | np.ndarray,
    market_probabilities: list[float] | np.ndarray,
    outcomes: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
) -> float:
    """Fit a single market blend weight on prior OOS rows only."""
    pure = np.asarray(pure_probabilities, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    exposure = np.asarray(weights, dtype=float)

    if not (len(pure) == len(market) == len(y) == len(exposure)):
        raise ValueError("Inputs must align.")
    if len(pure) < 100:
        raise ValueError("At least 100 prior OOS rows are required.")

    def objective(w: float) -> float:
        p = (1.0 - float(w)) * pure + float(w) * market
        return weighted_binary_brier(p, y, exposure)

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)
