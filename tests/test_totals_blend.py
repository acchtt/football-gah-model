import numpy as np

from gah.backtest import _fit_total_blend_weight, _total_probability


def test_total_probability_sums_matching_diagonal():
    matrix = np.zeros((4, 4))
    matrix[0, 2] = 0.2
    matrix[1, 1] = 0.3
    matrix[2, 0] = 0.4
    matrix[3, 0] = 0.1

    assert abs(_total_probability(matrix, 2) - 0.9) < 1e-12


def test_blend_weight_prefers_better_model_history():
    model_probs = [0.50, 0.45, 0.40, 0.55]
    naive_probs = [0.10, 0.12, 0.08, 0.09]

    w = _fit_total_blend_weight(model_probs, naive_probs)
    assert w > 0.9


def test_blend_weight_prefers_naive_when_it_is_better():
    model_probs = [0.08, 0.10, 0.09, 0.12]
    naive_probs = [0.45, 0.50, 0.40, 0.55]

    w = _fit_total_blend_weight(model_probs, naive_probs)
    assert w < 0.1
