from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


FEATURE_NAMES = (
    "home_attack_resid_5",
    "home_attack_resid_10",
    "home_defence_resid_5",
    "home_defence_resid_10",
    "away_attack_resid_5",
    "away_attack_resid_10",
    "away_defence_resid_5",
    "away_defence_resid_10",
    "home_total_resid_5",
    "away_total_resid_5",
    "home_total_std_10",
    "away_total_std_10",
    "home_rest",
    "away_rest",
    "home_history",
    "away_history",
)


def _tail_mean(values: list[float], n: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-n:]))


def _tail_std(values: list[float], n: int) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values[-n:], ddof=0))


@dataclass
class TeamResidualState:
    """
    Sequential team state built only from already-evaluated OOS matches.

    Attack/defence residuals are relative to Dixon-Coles expected goals, making
    them opponent-adjusted rather than simple raw scoring form.
    """

    max_history: int = 30
    attack_resid: dict[str, list[float]] = field(default_factory=dict)
    defence_resid: dict[str, list[float]] = field(default_factory=dict)
    total_resid: dict[str, list[float]] = field(default_factory=dict)
    total_actual: dict[str, list[float]] = field(default_factory=dict)
    dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    def _values(self, store: dict[str, list[float]], team: str) -> list[float]:
        return store.get(team, [])

    def _rest_feature(self, team: str, match_date: pd.Timestamp) -> float:
        dates = self.dates.get(team, [])
        if not dates:
            return 0.0
        current = pd.Timestamp(match_date)
        last = pd.Timestamp(dates[-1])
        days = max((current - last).total_seconds() / 86400.0, 0.0)
        # 7 days is neutral; clip unusual gaps so offseason does not dominate.
        return float((min(days, 21.0) - 7.0) / 14.0)

    def _history_feature(self, team: str) -> float:
        n = len(self.total_actual.get(team, []))
        return float(min(n, 10) / 10.0)

    def features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
    ) -> np.ndarray:
        ha = self._values(self.attack_resid, home_team)
        hd = self._values(self.defence_resid, home_team)
        aa = self._values(self.attack_resid, away_team)
        ad = self._values(self.defence_resid, away_team)
        ht = self._values(self.total_resid, home_team)
        at = self._values(self.total_resid, away_team)
        hactual = self._values(self.total_actual, home_team)
        aactual = self._values(self.total_actual, away_team)

        return np.array(
            [
                _tail_mean(ha, 5),
                _tail_mean(ha, 10),
                _tail_mean(hd, 5),
                _tail_mean(hd, 10),
                _tail_mean(aa, 5),
                _tail_mean(aa, 10),
                _tail_mean(ad, 5),
                _tail_mean(ad, 10),
                _tail_mean(ht, 5),
                _tail_mean(at, 5),
                _tail_std(hactual, 10),
                _tail_std(aactual, 10),
                self._rest_feature(home_team, match_date),
                self._rest_feature(away_team, match_date),
                self._history_feature(home_team),
                self._history_feature(away_team),
            ],
            dtype=float,
        )

    def _append(self, store: dict, key: str, value) -> None:
        values = store.setdefault(key, [])
        values.append(value)
        if len(values) > self.max_history:
            del values[:-self.max_history]

    def update(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        home_goals: int,
        away_goals: int,
        dc_xg_home: float,
        dc_xg_away: float,
        base_total_mean: float,
    ) -> None:
        actual_total = float(home_goals + away_goals)
        total_resid = actual_total - float(base_total_mean)

        self._append(
            self.attack_resid,
            home_team,
            float(home_goals) - float(dc_xg_home),
        )
        self._append(
            self.defence_resid,
            home_team,
            float(away_goals) - float(dc_xg_away),
        )
        self._append(
            self.attack_resid,
            away_team,
            float(away_goals) - float(dc_xg_away),
        )
        self._append(
            self.defence_resid,
            away_team,
            float(home_goals) - float(dc_xg_home),
        )

        for team in (home_team, away_team):
            self._append(self.total_resid, team, total_resid)
            self._append(self.total_actual, team, actual_total)
            self._append(self.dates, team, pd.Timestamp(match_date))


class ConditionalTotalTiltModel:
    """
    Ridge-regularised conditional exponential tilt for a base total-goals PMF.

    For match i:
        p_i'(k) proportional to p_i(k) * exp(beta_i * k)
        beta_i = x_i @ w

    The model is fitted only to historical OOS base distributions.
    """

    def __init__(
        self,
        l2: float = 1.0,
        beta_clip: float = 0.15,
    ) -> None:
        self.l2 = float(l2)
        self.beta_clip = float(beta_clip)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.success_: bool | None = None

    def fit(
        self,
        x: np.ndarray,
        pmfs: list[np.ndarray],
        actual_totals: list[int] | np.ndarray,
    ) -> "ConditionalTotalTiltModel":
        x = np.asarray(x, dtype=float)
        y = np.asarray(actual_totals, dtype=int)

        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if len(x) != len(pmfs) or len(x) != len(y):
            raise ValueError("x, pmfs, and actual_totals must have equal length.")
        if len(x) < 20:
            raise ValueError("At least 20 OOS rows are required.")

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
        base_actual = np.array(
            [
                base[i, yi] if 0 <= yi < max_len else 1e-12
                for i, yi in enumerate(y)
            ],
            dtype=float,
        )

        def objective(w: np.ndarray) -> float:
            beta = np.clip(xs @ w, -self.beta_clip, self.beta_clip)
            expo = np.exp(beta[:, None] * goals[None, :])
            z = np.sum(base * expo, axis=1)
            p_actual = (
                np.clip(base_actual, 1e-12, 1.0)
                * np.exp(beta * y)
                / np.clip(z, 1e-12, None)
            )
            nll = float(-np.log(np.clip(p_actual, 1e-12, 1.0)).mean())
            penalty = self.l2 * float(np.mean(np.square(w)))
            return nll + penalty

        x0 = np.zeros(x.shape[1], dtype=float)
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=[(-0.5, 0.5)] * x.shape[1],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        self.coef_ = np.asarray(res.x, dtype=float)
        self.success_ = bool(res.success)
        return self

    def predict_beta(self, x: np.ndarray) -> float:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Residual model is not fitted.")

        row = np.asarray(x, dtype=float)
        xs = (row - self.mean_) / self.scale_
        beta = float(xs @ self.coef_)
        return float(np.clip(beta, -self.beta_clip, self.beta_clip))


def fit_residual_scale(
    raw_betas: list[float] | np.ndarray,
    base_pmfs: list[np.ndarray],
    actual_totals: list[int] | np.ndarray,
    grid_size: int = 21,
) -> float:
    """
    Leakage-safe scalar shrinkage for historical OOS residual betas.

    scale=0 disables the residual layer; scale=1 trusts the raw conditional
    model fully. The scale is selected by exact-total log likelihood.
    """
    betas = np.asarray(raw_betas, dtype=float)
    y = np.asarray(actual_totals, dtype=int)
    if len(betas) == 0:
        return 0.0
    if len(betas) != len(base_pmfs) or len(betas) != len(y):
        raise ValueError("raw_betas, base_pmfs, and actual_totals must align.")

    max_len = max(len(p) for p in base_pmfs)
    base = np.zeros((len(base_pmfs), max_len), dtype=float)
    for i, pmf in enumerate(base_pmfs):
        arr = np.asarray(pmf, dtype=float)
        base[i, : len(arr)] = arr

    goals = np.arange(max_len, dtype=float)
    base_actual = np.array(
        [
            base[i, yi] if 0 <= yi < max_len else 1e-12
            for i, yi in enumerate(y)
        ],
        dtype=float,
    )

    best_scale = 0.0
    best_nll = np.inf
    for scale in np.linspace(0.0, 1.0, int(grid_size)):
        beta = scale * betas
        z = np.sum(
            base * np.exp(beta[:, None] * goals[None, :]),
            axis=1,
        )
        p_actual = (
            np.clip(base_actual, 1e-12, 1.0)
            * np.exp(beta * y)
            / np.clip(z, 1e-12, None)
        )
        nll = float(-np.log(np.clip(p_actual, 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_nll = nll
            best_scale = float(scale)

    return best_scale
