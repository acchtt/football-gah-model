import numpy as np

from gah.lineups import (
    LineupContinuityState,
    fit_lineup_margin_scale,
    margin_probability,
    reshape_score_matrix_by_margin_uncertainty,
    fit_lineup_uncertainty_scale,
    tilt_score_matrix_by_margin,
)


def _xi(start: int) -> tuple[int, ...]:
    return tuple(range(start, start + 11))


def test_lineup_coverage_rewards_recent_core():
    state = LineupContinuityState(window=5, min_history=3)
    regular = _xi(1)
    state.update(10, regular)
    state.update(10, regular)
    state.update(10, regular)

    assert np.isclose(state.coverage(10, regular), 1.0)

    rotated = tuple(list(range(1, 7)) + list(range(100, 105)))
    assert state.coverage(10, rotated) < 0.7


def test_lineup_coverage_is_nan_before_min_history():
    state = LineupContinuityState(window=5, min_history=3)
    state.update(10, _xi(1))
    assert np.isnan(state.coverage(10, _xi(1)))


def test_margin_tilt_moves_home_probability():
    matrix = np.ones((5, 5), dtype=float)
    matrix /= matrix.sum()

    tilted = tilt_score_matrix_by_margin(matrix, 0.2)
    assert np.isclose(tilted.sum(), 1.0)
    assert margin_probability(tilted, 2) > margin_probability(matrix, 2)


def test_fit_lineup_margin_scale_learns_positive_direction():
    matrix = np.ones((5, 5), dtype=float)
    matrix /= matrix.sum()

    matrices = []
    gaps = []
    margins = []
    for _ in range(40):
        matrices.append(matrix.copy())
        gaps.append(0.4)
        margins.append(2)
    for _ in range(40):
        matrices.append(matrix.copy())
        gaps.append(-0.4)
        margins.append(-2)

    alpha = fit_lineup_margin_scale(matrices, gaps, margins)
    assert alpha > 0


def test_uncertainty_transform_preserves_expected_margin():
    matrix = np.arange(1, 26, dtype=float).reshape(5, 5)
    matrix /= matrix.sum()
    h, a = np.indices(matrix.shape)
    margin = h - a
    before = float(np.sum(matrix * margin))

    shaped = reshape_score_matrix_by_margin_uncertainty(
        matrix,
        disruption=0.5,
        gamma=0.08,
    )
    after = float(np.sum(shaped * margin))

    assert np.isclose(shaped.sum(), 1.0)
    assert abs(after - before) < 1e-4


def test_uncertainty_scale_learns_wider_direction():
    matrix = np.zeros((5, 5), dtype=float)
    matrix[2, 2] = 0.5
    matrix[4, 0] = 0.25
    matrix[0, 4] = 0.25

    matrices = [matrix.copy() for _ in range(80)]
    disruptions = [0.8 for _ in range(80)]
    margins = [4 if i % 2 == 0 else -4 for i in range(80)]

    gamma = fit_lineup_uncertainty_scale(
        matrices,
        disruptions,
        margins,
    )
    assert gamma > 0
