import numpy as np

from gah.calibration import (
    calibration_bias,
    expected_calibration_error,
    expected_settlement_value,
    fit_brier_blend_weight,
    multiclass_brier,
    observed_settlement_value,
    use_model_by_paired_loss,
)
from gah.markets import (
    price_handicap,
    price_total,
    settle_handicap,
    settle_total,
)


def one_score(hg: int, ag: int, max_goals: int = 5) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    matrix[hg, ag] = 1.0
    return matrix


def test_realized_quarter_total_settlements():
    assert settle_total(2, 2.25, "over") == "half_loss"
    assert settle_total(3, 2.75, "over") == "half_win"
    assert settle_total(2, 2.25, "under") == "half_win"


def test_realized_quarter_handicap_settlements():
    assert settle_handicap(0, -0.25, "home") == "half_loss"
    assert settle_handicap(0, 0.25, "home") == "half_win"
    assert settle_handicap(1, -1.0, "home") == "push"


def test_perfect_settlement_distribution_has_zero_brier():
    pricing = price_total(one_score(1, 1), 2.25, "over")
    actual = settle_total(2, 2.25, "over")
    assert multiclass_brier(pricing, actual) == 0.0


def test_expected_settlement_value_matches_certainty():
    pricing = price_handicap(one_score(2, 1), -0.75, "home")
    assert expected_settlement_value(pricing) == 0.75
    assert observed_settlement_value("half_win") == 0.75


def test_calibration_metrics_are_zero_when_perfect():
    predicted = [0.1, 0.3, 0.6, 0.9]
    observed = [0.1, 0.3, 0.6, 0.9]
    assert expected_calibration_error(predicted, observed, n_bins=5) < 1e-12
    assert abs(calibration_bias(predicted, observed)) < 1e-12


def test_reliability_gate_keeps_model_before_minimum_history():
    assert use_model_by_paired_loss(
        [0.6] * 20,
        [0.5] * 20,
        min_history=100,
    )


def test_reliability_gate_falls_back_when_model_is_reliably_worse():
    model = np.linspace(0.54, 0.56, 200)
    baseline = np.linspace(0.50, 0.52, 200)
    assert not use_model_by_paired_loss(
        model,
        baseline,
        min_history=100,
        z_threshold=1.0,
    )


def test_reliability_gate_keeps_model_when_it_is_better():
    model = np.linspace(0.48, 0.50, 200)
    baseline = np.linspace(0.52, 0.54, 200)
    assert use_model_by_paired_loss(
        model,
        baseline,
        min_history=100,
        z_threshold=1.0,
    )


def test_brier_blend_prefers_perfect_model():
    model = np.array([[1.0, 0.0], [0.0, 1.0]])
    baseline = np.array([[0.5, 0.5], [0.5, 0.5]])
    observed = model.copy()
    assert fit_brier_blend_weight(model, baseline, observed) > 0.99


def test_brier_blend_prefers_perfect_baseline():
    model = np.array([[0.5, 0.5], [0.5, 0.5]])
    baseline = np.array([[1.0, 0.0], [0.0, 1.0]])
    observed = baseline.copy()
    assert fit_brier_blend_weight(model, baseline, observed) < 0.01
