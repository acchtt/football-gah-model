from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EDGE_BANDS = (
    (0.00, 0.02, "<2pp"),
    (0.02, 0.04, "2-4pp"),
    (0.04, 0.06, "4-6pp"),
    (0.06, 0.10, "6-10pp"),
    (0.10, float("inf"), "10pp+"),
)


def probability_edge(
    pure_probability: float,
    market_probability: float,
) -> float:
    """Signed GAH Pure minus de-vigged market probability."""
    return float(pure_probability) - float(market_probability)


def edge_band(edge: float) -> str:
    """Fixed, predeclared absolute disagreement band."""
    value = abs(float(edge))
    for low, high, label in EDGE_BANDS:
        if low <= value < high:
            return label
    raise RuntimeError("Unreachable edge band.")


def realized_net_return(actual_category: str, decimal_odds: float) -> float:
    """Net return per 1 unit stake for exact Asian settlement."""
    win_profit = float(decimal_odds) - 1.0
    mapping = {
        "full_win": win_profit,
        "half_win": 0.5 * win_profit,
        "push": 0.0,
        "half_loss": -0.5,
        "full_loss": -1.0,
    }
    if actual_category not in mapping:
        raise ValueError(f"Unknown settlement category: {actual_category}")
    return float(mapping[actual_category])


@dataclass(frozen=True)
class SelectionCandidate:
    match_key: str
    market: str
    side: str
    line: float
    pure_probability: float
    market_probability: float
    decimal_odds: float
    model_ev: float

    @property
    def edge(self) -> float:
        return probability_edge(
            self.pure_probability,
            self.market_probability,
        )

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)

    @property
    def band(self) -> str:
        return edge_band(self.edge)


def rank_candidates(
    candidates: Iterable[SelectionCandidate],
) -> list[SelectionCandidate]:
    """
    Deterministic ranking for research boards.

    Primary: absolute model-vs-market disagreement.
    Secondary: GAH-implied EV at the available opening price.
    """
    return sorted(
        list(candidates),
        key=lambda c: (c.abs_edge, c.model_ev),
        reverse=True,
    )


def select_board(
    candidates: Iterable[SelectionCandidate],
    *,
    max_matches: int = 8,
    min_abs_edge: float = 0.0,
) -> list[SelectionCandidate]:
    """
    Build a capped board while allowing at most one candidate per match.

    Thresholds are explicit inputs rather than silently learned in this layer.
    """
    if max_matches < 1:
        raise ValueError("max_matches must be positive.")
    if min_abs_edge < 0:
        raise ValueError("min_abs_edge cannot be negative.")

    selected: list[SelectionCandidate] = []
    used_matches: set[str] = set()

    for candidate in rank_candidates(candidates):
        if candidate.abs_edge < float(min_abs_edge):
            continue
        if candidate.match_key in used_matches:
            continue
        selected.append(candidate)
        used_matches.add(candidate.match_key)
        if len(selected) >= int(max_matches):
            break

    return selected
