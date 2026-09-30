import numpy as np

from gah.elo import (
    EloState,
    fit_elo_margin_scale,
    margin_probability,
    tilt_score_matrix_by_margin,
)


def test_elo_updates_without_future_information():
    elo = EloState()
    assert elo.rating("A") == 1500.0
    assert elo.match_count("A") == 0

    before = elo.normalized_gap("A", "B")
    elo.update("A", "B", 2, 0)

    assert elo.rating("A") > 1500.0
    assert elo.rating("B") < 1500.0
    assert elo.match_count("A") == 1
    assert elo.normalized_gap("A", "B") > before


def test_margin_tilt_moves_probability_toward_home():
    matrix = np.ones((4, 4), dtype=float)
    matrix /= matrix.sum()

    base_home_win = sum(
        matrix[h, a]
        for h in range(4)
        for a in range(4)
        if h > a
    )
    tilted = tilt_score_matrix_by_margin(matrix, 0.2)
    tilted_home_win = sum(
        tilted[h, a]
        for h in range(4)
        for a in range(4)
        if h > a
    )

    assert np.isclose(tilted.sum(), 1.0)
    assert tilted_home_win > base_home_win


def test_fit_elo_margin_scale_learns_positive_direction():
    matrix = np.ones((5, 5), dtype=float)
    matrix /= matrix.sum()

    matrices = []
    gaps = []
    margins = []
    for _ in range(40):
        matrices.append(matrix.copy())
        gaps.append(1.0)
        margins.append(2)
    for _ in range(40):
        matrices.append(matrix.copy())
        gaps.append(-1.0)
        margins.append(-2)

    alpha = fit_elo_margin_scale(matrices, gaps, margins)
    assert alpha > 0

    p_base = margin_probability(matrix, 2)
    p_tilt = margin_probability(
        tilt_score_matrix_by_margin(matrix, alpha),
        2,
    )
    assert p_tilt > p_base
