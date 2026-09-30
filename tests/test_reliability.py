import numpy as np

from gah.reliability import MarketResidualReliability, reliability_features


def test_reliability_features_are_opening_time_only_shape():
    x = reliability_features(
        market="AH",
        line=-0.75,
        pure_probability=0.61,
        market_probability=0.54,
    )
    assert x.shape == (5,)
    assert np.isclose(x[1], 0.07)
    assert np.isclose(x[2], 0.11)
    assert x[3] == 1.0
    assert np.isclose(x[4], 0.75)


def test_residual_model_learns_positive_edge_signal():
    rows = []
    for i in range(800):
        high = i % 2 == 0
        rows.append(
            {
                "market": "AH",
                "line": -0.5,
                "pure_probability": 0.65 if high else 0.45,
                "market_probability": 0.50,
                "actual_category": "full_win" if high else "full_loss",
            }
        )

    model = MarketResidualReliability(l2=10.0).fit(rows)
    high_p = model.predict_probability(
        market="AH",
        line=-0.5,
        pure_probability=0.65,
        market_probability=0.50,
    )
    low_p = model.predict_probability(
        market="AH",
        line=-0.5,
        pure_probability=0.45,
        market_probability=0.50,
    )
    assert high_p > low_p


def test_fit_minimum_counts_active_settlements_not_pushes():
    active = []
    for i in range(500):
        active.append(
            {
                "market": "AH",
                "line": -0.5,
                "pure_probability": 0.56 if i % 2 == 0 else 0.44,
                "market_probability": 0.50,
                "actual_category": "full_win" if i % 2 == 0 else "full_loss",
            }
        )
    pushes = [
        {
            "market": "AH",
            "line": 0.0,
            "pure_probability": 0.50,
            "market_probability": 0.50,
            "actual_category": "push",
        }
        for _ in range(100)
    ]

    # 600 total rows, but exactly 500 active settlements: valid.
    MarketResidualReliability(l2=10.0).fit(active + pushes)

    # 599 total rows, but only 499 active settlements: invalid.
    try:
        MarketResidualReliability(l2=10.0).fit(active[:-1] + pushes)
    except ValueError as exc:
        assert "500 prior active-settlement rows" in str(exc)
    else:
        raise AssertionError("Expected active-settlement minimum to be enforced")
