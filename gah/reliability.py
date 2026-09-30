from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from .market_aware import realized_exposure


def _logit(p: float) -> float:
    q = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(q / (1.0 - q))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def reliability_features(
    *,
    market: str,
    line: float,
    pure_probability: float,
    market_probability: float,
) -> np.ndarray:
    """
    Predeclared opening-time features for v2.0 reliability.

    The market logit is handled as an offset, so these features only ask
    whether GAH supplies systematic residual information beyond the market.
    """
    edge = float(pure_probability) - float(market_probability)
    confidence = abs(float(pure_probability) - 0.5)
    is_ah = 1.0 if str(market).upper() == "AH" else 0.0
    ah_line_magnitude = abs(float(line)) if is_ah else 0.0
    return np.array(
        [1.0, edge, confidence, is_ah, ah_line_magnitude],
        dtype=float,
    )


class MarketResidualReliability:
    """
    Small regularized logistic residual model with market probability as offset.
    """

    def __init__(self, l2: float = 10.0):
        self.l2 = float(l2)
        self.coef_: np.ndarray | None = None

    def fit(self, rows) -> "MarketResidualReliability":
        X = []
        offsets = []
        y = []
        weights = []

        for row in rows:
            exposure = realized_exposure(str(row["actual_category"]))
            if exposure is None:
                continue
            outcome, weight = exposure
            X.append(
                reliability_features(
                    market=str(row["market"]),
                    line=float(row["line"]),
                    pure_probability=float(row["pure_probability"]),
                    market_probability=float(row["market_probability"]),
                )
            )
            offsets.append(_logit(float(row["market_probability"])))
            y.append(float(outcome))
            weights.append(float(weight))

        if len(X) < 500:
            raise ValueError("At least 500 prior active-settlement rows are required.")

        X = np.asarray(X, dtype=float)
        offsets = np.asarray(offsets, dtype=float)
        y = np.asarray(y, dtype=float)
        weights = np.asarray(weights, dtype=float)

        def objective(beta: np.ndarray) -> float:
            z = offsets + X @ beta
            p = np.clip(_sigmoid(z), 1e-9, 1.0 - 1e-9)
            loss = -np.sum(
                weights * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            ) / np.sum(weights)
            penalty = self.l2 * float(np.square(beta[1:]).sum()) / len(y)
            return float(loss + penalty)

        result = minimize(
            objective,
            x0=np.zeros(X.shape[1], dtype=float),
            method="L-BFGS-B",
        )
        if not result.success:
            raise RuntimeError(f"Reliability fit failed: {result.message}")
        self.coef_ = np.asarray(result.x, dtype=float)
        return self

    def predict_probability(
        self,
        *,
        market: str,
        line: float,
        pure_probability: float,
        market_probability: float,
    ) -> float:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        x = reliability_features(
            market=market,
            line=line,
            pure_probability=pure_probability,
            market_probability=market_probability,
        )
        z = _logit(float(market_probability)) + float(x @ self.coef_)
        return float(_sigmoid(z))
