from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


def _recent_mean(values: list[float], n: int) -> float:
    vals = [float(v) for v in values[-n:] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class TacticalTeamState:
    """Sequential Understat team context, updated only after each prediction."""

    max_history: int = 30
    npxg_for: dict[str, list[float]] = field(default_factory=dict)
    npxg_against: dict[str, list[float]] = field(default_factory=dict)
    ppda: dict[str, list[float]] = field(default_factory=dict)
    deep: dict[str, list[float]] = field(default_factory=dict)

    def _append(self, store: dict[str, list[float]], team: str, value: float) -> None:
        if not np.isfinite(value):
            return
        values = store.setdefault(team, [])
        values.append(float(value))
        if len(values) > self.max_history:
            del values[:-self.max_history]

    def update(
        self,
        home_team: str,
        away_team: str,
        home_np_xg: float,
        away_np_xg: float,
        home_ppda: float,
        away_ppda: float,
        home_deep: float,
        away_deep: float,
    ) -> None:
        self._append(self.npxg_for, home_team, home_np_xg)
        self._append(self.npxg_against, home_team, away_np_xg)
        self._append(self.ppda, home_team, home_ppda)
        self._append(self.deep, home_team, home_deep)

        self._append(self.npxg_for, away_team, away_np_xg)
        self._append(self.npxg_against, away_team, home_np_xg)
        self._append(self.ppda, away_team, away_ppda)
        self._append(self.deep, away_team, away_deep)

    def features(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Team-level tactical form. PPDA is negated so larger means more pressure.
        Both short (5) and medium (10) windows are included for shot quality.
        """
        return np.array(
            [
                _recent_mean(self.npxg_for.get(home_team, []), 5),
                _recent_mean(self.npxg_for.get(home_team, []), 10),
                _recent_mean(self.npxg_against.get(home_team, []), 5),
                _recent_mean(self.npxg_against.get(home_team, []), 10),
                _recent_mean(self.npxg_for.get(away_team, []), 5),
                _recent_mean(self.npxg_for.get(away_team, []), 10),
                _recent_mean(self.npxg_against.get(away_team, []), 5),
                _recent_mean(self.npxg_against.get(away_team, []), 10),
                -_recent_mean(self.ppda.get(home_team, []), 5),
                -_recent_mean(self.ppda.get(away_team, []), 5),
                _recent_mean(self.deep.get(home_team, []), 5),
                _recent_mean(self.deep.get(away_team, []), 5),
            ],
            dtype=float,
        )


class ConditionalTacticalTilt:
    """
    Ridge-regularised match-specific tilt on an existing total-goals PMF.

    beta_i = x_i @ w
    p_i'(k) proportional to p_i(k) * exp(beta_i * k)

    This preserves the production score-shape logic and only learns whether
    tactical/xG context implies a higher or lower total-goal intensity.
    """

    def __init__(self, l2: float = 2.0, beta_clip: float = 0.08) -> None:
        self.l2 = float(l2)
        self.beta_clip = float(beta_clip)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray,
        pmfs: list[np.ndarray],
        actual_totals: list[int] | np.ndarray,
    ) -> "ConditionalTacticalTilt":
        x = np.asarray(features, dtype=float)
        y = np.asarray(actual_totals, dtype=int)

        if x.ndim != 2:
            raise ValueError("features must be a 2D array.")
        if len(x) != len(pmfs) or len(x) != len(y):
            raise ValueError("features, pmfs, and actual_totals must align.")
        if len(x) < 50:
            raise ValueError("At least 50 historical OOS rows are required.")

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.scale_

        max_len = max(len(p) for p in pmfs)
        base = np.zeros((len(pmfs), max_len), dtype=float)
        for i, pmf in enumerate(pmfs):
            arr = np.asarray(pmf, dtype=float)
            base[i, : len(arr)] = arr

        goals = np.arange(max_len, dtype=float)
        observed_base = np.array(
            [
                base[i, yi] if 0 <= yi < max_len else 1e-12
                for i, yi in enumerate(y)
            ],
            dtype=float,
        )

        def objective(w: np.ndarray) -> float:
            beta = np.clip(xs @ w, -self.beta_clip, self.beta_clip)
            expo = np.exp(beta[:, None] * goals[None, :])
            norm = np.sum(base * expo, axis=1)
            p_obs = (
                np.clip(observed_base, 1e-12, 1.0)
                * np.exp(beta * y)
                / np.clip(norm, 1e-12, None)
            )
            nll = float(-np.log(np.clip(p_obs, 1e-12, 1.0)).mean())
            penalty = self.l2 * float(np.mean(np.square(w)))
            return nll + penalty

        result = minimize(
            objective,
            np.zeros(x.shape[1], dtype=float),
            method="L-BFGS-B",
            bounds=[(-0.25, 0.25)] * x.shape[1],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        self.coef_ = np.asarray(result.x, dtype=float)
        return self

    def predict_beta(self, features: np.ndarray) -> float:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Tactical tilt model is not fitted.")
        x = (np.asarray(features, dtype=float) - self.mean_) / self.scale_
        beta = float(x @ self.coef_)
        return float(np.clip(beta, -self.beta_clip, self.beta_clip))
