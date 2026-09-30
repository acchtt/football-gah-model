import numpy as np

from gah.market_aware import (
    blend_probability,
    devig_two_way,
    fit_market_blend_weight,
    model_effective_win_probability,
    realized_exposure,
    weighted_binary_brier,
)


def test_devig_two_way_normalizes():
    a, b = devig_two_way(1.91, 1.91)
    assert np.isclose(a + b, 1.0)
    assert np.isclose(a, 0.5)


def test_model_effective_win_probability_handles_push_mass():
    pricing = {
        "full_win": 0.40,
        "half_win": 0.10,
        "push": 0.10,
        "half_loss": 0.10,
        "full_loss": 0.30,
    }
    p = model_effective_win_probability(pricing)
    assert np.isclose(p, 0.45 / 0.80)


def test_realized_exposure_weights_half_results():
    assert realized_exposure("full_win") == (1.0, 1.0)
    assert realized_exposure("half_win") == (1.0, 0.5)
    assert realized_exposure("push") is None
    assert realized_exposure("half_loss") == (0.0, 0.5)
    assert realized_exposure("full_loss") == (0.0, 1.0)


def test_market_blend_fit_prefers_better_market_signal():
    pure = np.full(200, 0.5)
    market = np.array([0.8] * 100 + [0.2] * 100)
    outcomes = np.array([1.0] * 100 + [0.0] * 100)
    weights = np.ones(200)

    fitted = fit_market_blend_weight(pure, market, outcomes, weights)
    assert fitted > 0.9
    assert blend_probability(0.5, 0.8, fitted) > 0.75
    assert weighted_binary_brier(market, outcomes, weights) < weighted_binary_brier(
        pure, outcomes, weights
    )
