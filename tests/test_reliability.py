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
