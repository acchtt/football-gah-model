import numpy as np

from predict import build_market_table


def test_totals_use_blended_matrix_and_ah_uses_raw_matrix():
    raw = np.zeros((3, 3))
    raw[1, 0] = 1.0

    totals = np.zeros((3, 3))
    totals[0, 0] = 1.0

    markets = build_market_table(raw, totals)

    under_15 = markets[
        (markets["market"] == "total")
        & (markets["side"] == "under")
        & (markets["line"] == 1.5)
    ].iloc[0]
    assert under_15["full_win"] == 1.0

    home_m05 = markets[
        (markets["market"] == "asian_handicap")
        & (markets["side"] == "home")
        & (markets["line"] == -0.5)
    ].iloc[0]
    assert home_m05["full_win"] == 1.0
