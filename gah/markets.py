from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


def _split_asian_line(line: float) -> list[float]:
    """
    Split quarter lines into two adjacent half/integer lines.
    Examples:
      -0.25 -> [0.0, -0.5]
      -0.75 -> [-0.5, -1.0]
      +0.25 -> [0.0, +0.5]
      +0.75 -> [+0.5, +1.0]
    """
    q = round(line * 4)
    if q % 2 == 0:
        return [q / 4.0]

    lo = math.floor(line * 2) / 2.0
    hi = math.ceil(line * 2) / 2.0
    return [lo, hi]


def _leg_result(value: float, eps: float = 1e-9) -> str:
    if value > eps:
        return "win"
    if value < -eps:
        return "loss"
    return "push"


def _settlement_category(base_margin: float, line: float) -> str:
    legs = _split_asian_line(line)
    results = [_leg_result(base_margin + leg) for leg in legs]

    if len(results) == 1:
        return {"win": "full_win", "push": "push", "loss": "full_loss"}[results[0]]

    pair = tuple(sorted(results))
    mapping = {
        ("win", "win"): "full_win",
        ("push", "win"): "half_win",
        ("loss", "win"): "push",       # Theoretically not reached with adjacent Asian legs.
        ("push", "push"): "push",
        ("loss", "push"): "half_loss",
        ("loss", "loss"): "full_loss",
    }
    return mapping[pair]


def _price_from_categories(categories: dict[str, float]) -> dict:
    p_fw = categories.get("full_win", 0.0)
    p_hw = categories.get("half_win", 0.0)
    p_p = categories.get("push", 0.0)
    p_hl = categories.get("half_loss", 0.0)
    p_fl = categories.get("full_loss", 0.0)

    effective_win = p_fw + 0.5 * p_hw
    effective_loss = p_fl + 0.5 * p_hl

    fair_odds = math.inf if effective_win <= 0 else 1.0 + effective_loss / effective_win

    return {
        **{k: float(categories.get(k, 0.0)) for k in
           ["full_win", "half_win", "push", "half_loss", "full_loss"]},
        "effective_win": float(effective_win),
        "effective_loss": float(effective_loss),
        "fair_decimal_odds": float(fair_odds),
    }


def expected_value_at_odds(pricing: dict, decimal_odds: float) -> float:
    """
    Net expected return per 1 unit staked, correctly accounting for half wins/losses.
    """
    win_profit = decimal_odds - 1.0
    return (
        pricing["full_win"] * win_profit
        + pricing["half_win"] * (0.5 * win_profit)
        - pricing["half_loss"] * 0.5
        - pricing["full_loss"]
    )


def price_handicap(matrix: np.ndarray, line: float, side: str = "home") -> dict:
    side = side.lower()
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")

    cats = defaultdict(float)
    rows, cols = matrix.shape
    for hg in range(rows):
        for ag in range(cols):
            p = float(matrix[hg, ag])
            margin = (hg - ag) if side == "home" else (ag - hg)
            cats[_settlement_category(margin, float(line))] += p

    out = _price_from_categories(cats)
    out.update({"market": "asian_handicap", "side": side, "line": float(line)})
    return out


def price_total(matrix: np.ndarray, line: float, side: str = "over") -> dict:
    side = side.lower()
    if side not in {"over", "under"}:
        raise ValueError("side must be 'over' or 'under'")

    cats = defaultdict(float)
    rows, cols = matrix.shape
    for hg in range(rows):
        for ag in range(cols):
            p = float(matrix[hg, ag])
            total = hg + ag
            margin = (total - line) if side == "over" else (line - total)
            cats[_settlement_category(margin, 0.0)] += p

    # Quarter totals must split the market line, not the already-computed margin.
    legs = _split_asian_line(float(line))
    if len(legs) == 2:
        cats = defaultdict(float)
        for hg in range(rows):
            for ag in range(cols):
                p = float(matrix[hg, ag])
                total = hg + ag
                results = []
                for leg in legs:
                    margin = (total - leg) if side == "over" else (leg - total)
                    results.append(_leg_result(margin))
                pair = tuple(sorted(results))
                mapping = {
                    ("win", "win"): "full_win",
                    ("push", "win"): "half_win",
                    ("loss", "win"): "push",
                    ("push", "push"): "push",
                    ("loss", "push"): "half_loss",
                    ("loss", "loss"): "full_loss",
                }
                cats[mapping[pair]] += p

    out = _price_from_categories(cats)
    out.update({"market": "total", "side": side, "line": float(line)})
    return out


def standard_market_table(matrix: np.ndarray) -> list[dict]:
    rows = []
    for line in [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
        rows.append(price_total(matrix, line, "over"))
        rows.append(price_total(matrix, line, "under"))

    for line in [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
                 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        rows.append(price_handicap(matrix, line, "home"))
        rows.append(price_handicap(matrix, -line, "away"))
    return rows
