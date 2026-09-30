from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


REQUIRED_COLUMNS = {
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
}


@dataclass
class FitSummary:
    success: bool
    message: str
    n_matches: int
    n_teams: int
    objective: float
    rho: float
    home_advantage: float
    intercept: float


class DixonColesModel:
    """
    Time-decayed Dixon-Coles score model.

    Expected goals:
        log(lambda_home) = intercept + home_advantage
                           + attack_home - defence_away
        log(lambda_away) = intercept
                           + attack_away - defence_home

    The attack parameters are centred after fitting for identifiability.
    """

    def __init__(
        self,
        half_life_days: float = 180.0,
        max_goals: int = 8,
        l2: float = 0.01,
    ) -> None:
        self.half_life_days = float(half_life_days)
        self.max_goals = int(max_goals)
        self.l2 = float(l2)

        self.teams_: list[str] = []
        self.team_to_idx_: Dict[str, int] = {}
        self.attack_: np.ndarray | None = None
        self.defence_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.home_advantage_: float | None = None
        self.rho_: float | None = None
        self.fit_date_: pd.Timestamp | None = None
        self.summary_: FitSummary | None = None

    @staticmethod
    def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy()
        out["match_date"] = pd.to_datetime(out["match_date"], utc=True)
        out["home_goals"] = pd.to_numeric(out["home_goals"], errors="raise").astype(int)
        out["away_goals"] = pd.to_numeric(out["away_goals"], errors="raise").astype(int)

        if (out[["home_goals", "away_goals"]] < 0).any().any():
            raise ValueError("Goals must be non-negative.")
        return out.sort_values("match_date").reset_index(drop=True)

    def _weights(self, dates: pd.Series, as_of: pd.Timestamp) -> np.ndarray:
        age_days = (as_of - dates).dt.total_seconds().to_numpy() / 86400.0
        age_days = np.maximum(age_days, 0.0)
        return np.exp(-np.log(2.0) * age_days / self.half_life_days)

    @staticmethod
    def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
        tau = np.ones_like(lam, dtype=float)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        tau[m00] = 1.0 - (lam[m00] * mu[m00] * rho)
        tau[m01] = 1.0 + (lam[m01] * rho)
        tau[m10] = 1.0 + (mu[m10] * rho)
        tau[m11] = 1.0 - rho
        return tau

    def fit(self, df: pd.DataFrame, as_of=None) -> FitSummary:
        df = self._validate_frame(df)
        if len(df) < 20:
            raise ValueError("At least 20 completed matches are required for a baseline fit.")

        if as_of is None:
            as_of = df["match_date"].max()
        as_of = pd.Timestamp(as_of)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")

        df = df[df["match_date"] <= as_of].copy()
        if len(df) < 20:
            raise ValueError("Too few matches remain before as_of.")

        teams = sorted(set(df["home_team"]).union(df["away_team"]))
        self.teams_ = teams
        self.team_to_idx_ = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = df["home_team"].map(self.team_to_idx_).to_numpy()
        ai = df["away_team"].map(self.team_to_idx_).to_numpy()
        hg = df["home_goals"].to_numpy()
        ag = df["away_goals"].to_numpy()
        w = self._weights(df["match_date"], as_of)

        # Params: attacks[n], defences[n], intercept, home_adv, rho
        x0 = np.zeros(2 * n + 3, dtype=float)
        x0[2 * n] = np.log(max((hg.sum() + ag.sum()) / (2.0 * len(df)), 0.2))
        x0[2 * n + 1] = 0.15
        x0[2 * n + 2] = -0.05

        bounds = [(-2.5, 2.5)] * (2 * n)
        bounds += [(-2.0, 1.5), (-1.0, 1.0), (-0.20, 0.20)]

        def objective(theta: np.ndarray) -> float:
            attack = theta[:n]
            defence = theta[n:2*n]
            intercept = theta[2*n]
            home_adv = theta[2*n + 1]
            rho = theta[2*n + 2]

            log_lam = intercept + home_adv + attack[hi] - defence[ai]
            log_mu = intercept + attack[ai] - defence[hi]
            lam = np.exp(np.clip(log_lam, -5, 4))
            mu = np.exp(np.clip(log_mu, -5, 4))

            tau = self._tau(hg, ag, lam, mu, rho)
            if np.any(tau <= 1e-12):
                return 1e12

            ll = (
                poisson.logpmf(hg, lam)
                + poisson.logpmf(ag, mu)
                + np.log(tau)
            )

            # Centre attacks softly and regularise both strength vectors.
            centre_penalty = 50.0 * (attack.mean() ** 2)
            ridge = self.l2 * (np.square(attack).sum() + np.square(defence).sum())
            return float(-(w * ll).sum() + centre_penalty + ridge)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-10},
        )

        theta = res.x
        attack = theta[:n].copy()
        defence = theta[n:2*n].copy()

        # Exact centring while preserving predictions by moving the mean into intercept.
        shift = float(attack.mean())
        attack -= shift
        intercept = float(theta[2*n] + shift)

        self.attack_ = attack
        self.defence_ = defence
        self.intercept_ = intercept
        self.home_advantage_ = float(theta[2*n + 1])
        self.rho_ = float(theta[2*n + 2])
        self.fit_date_ = as_of

        self.summary_ = FitSummary(
            success=bool(res.success),
            message=str(res.message),
            n_matches=len(df),
            n_teams=n,
            objective=float(res.fun),
            rho=self.rho_,
            home_advantage=self.home_advantage_,
            intercept=self.intercept_,
        )
        return self.summary_

    def _check_fitted(self) -> None:
        attrs = [self.attack_, self.defence_, self.intercept_, self.home_advantage_, self.rho_]
        if any(x is None for x in attrs):
            raise RuntimeError("Model is not fitted.")

    def expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        self._check_fitted()
        if home_team not in self.team_to_idx_:
            raise KeyError(f"Unknown home team: {home_team}")
        if away_team not in self.team_to_idx_:
            raise KeyError(f"Unknown away team: {away_team}")

        h = self.team_to_idx_[home_team]
        a = self.team_to_idx_[away_team]
        lam = np.exp(
            self.intercept_
            + self.home_advantage_
            + self.attack_[h]
            - self.defence_[a]
        )
        mu = np.exp(
            self.intercept_
            + self.attack_[a]
            - self.defence_[h]
        )
        return float(lam), float(mu)

    def score_matrix(self, home_team: str, away_team: str, max_goals: int | None = None) -> np.ndarray:
        self._check_fitted()
        max_goals = self.max_goals if max_goals is None else int(max_goals)
        lam, mu = self.expected_goals(home_team, away_team)

        goals = np.arange(max_goals + 1)
        hp = poisson.pmf(goals, lam)
        ap = poisson.pmf(goals, mu)
        matrix = np.outer(hp, ap)

        # Apply Dixon-Coles correction to low-score cells.
        rho = self.rho_
        matrix[0, 0] *= 1.0 - lam * mu * rho
        matrix[0, 1] *= 1.0 + lam * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho

        # Tail beyond max_goals is small; renormalise the represented mass.
        total = matrix.sum()
        if total <= 0:
            raise RuntimeError("Invalid score matrix.")
        return matrix / total

    def predict(self, home_team: str, away_team: str) -> dict:
        lam, mu = self.expected_goals(home_team, away_team)
        matrix = self.score_matrix(home_team, away_team)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "xg_home": lam,
            "xg_away": mu,
            "xg_total": lam + mu,
            "expected_margin": lam - mu,
            "score_matrix": matrix,
        }
