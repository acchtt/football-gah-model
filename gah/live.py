from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from scipy.stats import poisson

from .market_aware import model_effective_win_probability
from .markets import expected_value_at_odds, price_handicap


class LiveChangeType(str, Enum):
    NONE = "NONE"
    MARKET_ONLY = "MARKET_ONLY"
    MATCH_STATE = "MATCH_STATE"
    MATERIAL_EVENT = "MATERIAL_EVENT"


@dataclass(frozen=True)
class LiveAdjustment:
    home_rate_multiplier: float = 1.0
    away_rate_multiplier: float = 1.0
    material_events: tuple[str, ...] = ()
    reason: str | None = None

    def validate(self) -> None:
        for name, value in {
            "home_rate_multiplier": self.home_rate_multiplier,
            "away_rate_multiplier": self.away_rate_multiplier,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero.")

        adjusted = (
            abs(float(self.home_rate_multiplier) - 1.0) > 1e-12
            or abs(float(self.away_rate_multiplier) - 1.0) > 1e-12
            or bool(self.material_events)
        )
        if adjusted and not (self.reason and self.reason.strip()):
            raise ValueError(
                "Material live adjustments require a documented reason/source."
            )


@dataclass(frozen=True)
class LiveState:
    minute: float
    home_score: int
    away_score: int
    pre_match_home_xg: float
    pre_match_away_xg: float
    adjustment: LiveAdjustment = LiveAdjustment()

    def validate(self) -> None:
        if not 0.0 <= float(self.minute) <= 90.0:
            raise ValueError("minute must be between 0 and 90 for regulation time.")
        if int(self.home_score) != self.home_score or self.home_score < 0:
            raise ValueError("home_score must be a non-negative integer.")
        if int(self.away_score) != self.away_score or self.away_score < 0:
            raise ValueError("away_score must be a non-negative integer.")
        if not math.isfinite(float(self.pre_match_home_xg)) or self.pre_match_home_xg <= 0:
            raise ValueError("pre_match_home_xg must be finite and greater than zero.")
        if not math.isfinite(float(self.pre_match_away_xg)) or self.pre_match_away_xg <= 0:
            raise ValueError("pre_match_away_xg must be finite and greater than zero.")
        self.adjustment.validate()

    @property
    def remaining_fraction(self) -> float:
        self.validate()
        return max(0.0, (90.0 - float(self.minute)) / 90.0)

    @property
    def remaining_home_xg(self) -> float:
        return (
            float(self.pre_match_home_xg)
            * self.remaining_fraction
            * float(self.adjustment.home_rate_multiplier)
        )

    @property
    def remaining_away_xg(self) -> float:
        return (
            float(self.pre_match_away_xg)
            * self.remaining_fraction
            * float(self.adjustment.away_rate_multiplier)
        )


@dataclass(frozen=True)
class LiveAHQuote:
    side: str
    line: float
    effective_win_probability: float
    fair_decimal_odds: float
    expected_value: float | None
    pricing: dict


def conditional_final_score_matrix(
    state: LiveState,
    *,
    max_remaining_goals: int = 10,
) -> np.ndarray:
    state.validate()
    if max_remaining_goals < 1:
        raise ValueError("max_remaining_goals must be at least 1.")

    home_rem = np.arange(max_remaining_goals + 1)
    away_rem = np.arange(max_remaining_goals + 1)
    hp = poisson.pmf(home_rem, state.remaining_home_xg)
    ap = poisson.pmf(away_rem, state.remaining_away_xg)

    # Fold truncated tail mass into the final bucket rather than discarding it.
    hp[-1] += max(0.0, 1.0 - float(hp.sum()))
    ap[-1] += max(0.0, 1.0 - float(ap.sum()))

    matrix = np.zeros(
        (
            state.home_score + max_remaining_goals + 1,
            state.away_score + max_remaining_goals + 1,
        ),
        dtype=float,
    )
    matrix[
        state.home_score : state.home_score + max_remaining_goals + 1,
        state.away_score : state.away_score + max_remaining_goals + 1,
    ] = np.outer(hp, ap)

    total = float(matrix.sum())
    if total <= 0.0:
        raise RuntimeError("Conditional score matrix has no probability mass.")
    return matrix / total


def reprice_live_ah(
    state: LiveState,
    *,
    side: str,
    line: float,
    decimal_odds: float | None = None,
    max_remaining_goals: int = 10,
) -> LiveAHQuote:
    matrix = conditional_final_score_matrix(
        state,
        max_remaining_goals=max_remaining_goals,
    )
    pricing = price_handicap(matrix, float(line), side)
    probability = model_effective_win_probability(pricing)
    ev = (
        expected_value_at_odds(pricing, float(decimal_odds))
        if decimal_odds is not None
        else None
    )
    return LiveAHQuote(
        side=side,
        line=float(line),
        effective_win_probability=float(probability),
        fair_decimal_odds=float(pricing["fair_decimal_odds"]),
        expected_value=None if ev is None else float(ev),
        pricing=pricing,
    )


def live_ah_table(
    state: LiveState,
    *,
    side: str,
    lines: list[float] | tuple[float, ...],
) -> list[LiveAHQuote]:
    return [
        reprice_live_ah(state, side=side, line=float(line))
        for line in lines
    ]


def classify_live_change(
    previous: LiveState,
    current: LiveState,
    *,
    market_changed: bool = False,
) -> LiveChangeType:
    previous.validate()
    current.validate()

    if previous.adjustment != current.adjustment:
        return LiveChangeType.MATERIAL_EVENT

    if (
        float(previous.minute) != float(current.minute)
        or previous.home_score != current.home_score
        or previous.away_score != current.away_score
    ):
        return LiveChangeType.MATCH_STATE

    if market_changed:
        return LiveChangeType.MARKET_ONLY

    return LiveChangeType.NONE
