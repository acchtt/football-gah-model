from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class EloState:
    """Simple sequential football Elo state with no future information."""

    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    scale: float = 400.0
    ratings: dict[str, float] = field(default_factory=dict)
    matches: dict[str, int] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return float(self.ratings.get(team, self.base_rating))

    def match_count(self, team: str) -> int:
        return int(self.matches.get(team, 0))

    def expected_home_score(self, home_team: str, away_team: str) -> float:
        rh = self.rating(home_team) + self.home_advantage
        ra = self.rating(away_team)
        return float(1.0 / (1.0 + 10.0 ** ((ra - rh) / self.scale)))

    def normalized_gap(self, home_team: str, away_team: str) -> float:
        """
        Elo gap in 400-point units, including fixed home advantage.

        A value of +1 means the home side is about 400 Elo points stronger
        after home advantage; -1 means the reverse.
        """
        return float(
            (self.rating(home_team) + self.home_advantage - self.rating(away_team))
            / self.scale
        )

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        expected = self.expected_home_score(home_team, away_team)
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals < away_goals:
            actual = 0.0
        else:
            actual = 0.5

        change = self.k_factor * (actual - expected)
        self.ratings[home_team] = self.rating(home_team) + change
        self.ratings[away_team] = self.rating(away_team) - change
        self.matches[home_team] = self.match_count(home_team) + 1
        self.matches[away_team] = self.match_count(away_team) + 1


def tilt_score_matrix_by_margin(matrix: np.ndarray, beta: float) -> np.ndarray:
    """Exponentially tilt a score matrix by goal margin only."""
    base = np.asarray(matrix, dtype=float)
    h, a = np.indices(base.shape)
    margin = h - a
    tilted = base * np.exp(float(beta) * margin)
    total = float(tilted.sum())
    if total <= 0:
        raise ValueError("Tilted score matrix has no probability mass.")
    return tilted / total


def margin_probability(matrix: np.ndarray, actual_margin: int) -> float:
    base = np.asarray(matrix, dtype=float)
    h, a = np.indices(base.shape)
    return float(base[(h - a) == int(actual_margin)].sum())


def fit_elo_margin_scale(
    matrices: list[np.ndarray],
    elo_gaps: list[float] | np.ndarray,
    actual_margins: list[int] | np.ndarray,
    bound: float = 0.25,
) -> float:
    """
    Fit one global Elo-to-margin tilt scale on prior OOS predictions only.

    beta_i = alpha * elo_gap_i

    The objective is goal-margin NLL, not a particular betting line.
    """
    gaps = np.asarray(elo_gaps, dtype=float)
    margins = np.asarray(actual_margins, dtype=int)

    if len(matrices) != len(gaps) or len(gaps) != len(margins):
        raise ValueError("matrices, elo_gaps and actual_margins must align.")
    if len(gaps) < 50:
        raise ValueError("At least 50 OOS rows are required.")

    def objective(alpha: float) -> float:
        losses = []
        for matrix, gap, actual in zip(matrices, gaps, margins):
            tilted = tilt_score_matrix_by_margin(matrix, float(alpha) * float(gap))
            p = margin_probability(tilted, int(actual))
            losses.append(-np.log(max(p, 1e-12)))
        return float(np.mean(losses))

    result = minimize_scalar(
        objective,
        bounds=(-float(bound), float(bound)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)
