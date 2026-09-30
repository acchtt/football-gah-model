import numpy as np

from gah.selection import (
    SelectionCandidate,
    edge_band,
    probability_edge,
    realized_net_return,
    select_board,
)


def _candidate(match_key: str, edge: float, ev: float = 0.0):
    market = 0.50
    pure = market + edge
    return SelectionCandidate(
        match_key=match_key,
        market="AH",
        side="home",
        line=0.0,
        pure_probability=pure,
        market_probability=market,
        decimal_odds=1.95,
        model_ev=ev,
    )


def test_edge_and_fixed_bands():
    assert np.isclose(probability_edge(0.57, 0.51), 0.06)
    assert edge_band(0.019) == "<2pp"
    assert edge_band(-0.03) == "2-4pp"
    assert edge_band(0.055) == "4-6pp"
    assert edge_band(-0.08) == "6-10pp"
    assert edge_band(0.12) == "10pp+"


def test_realized_asian_returns():
    assert np.isclose(realized_net_return("full_win", 1.90), 0.90)
    assert np.isclose(realized_net_return("half_win", 1.90), 0.45)
    assert np.isclose(realized_net_return("push", 1.90), 0.0)
    assert np.isclose(realized_net_return("half_loss", 1.90), -0.5)
    assert np.isclose(realized_net_return("full_loss", 1.90), -1.0)


def test_board_caps_and_deduplicates_matches():
    candidates = [
        _candidate("a", 0.08, 0.05),
        _candidate("a", 0.07, 0.10),
        _candidate("b", -0.06, 0.03),
        _candidate("c", 0.03, 0.02),
    ]
    board = select_board(candidates, max_matches=2, min_abs_edge=0.04)
    assert [x.match_key for x in board] == ["a", "b"]
