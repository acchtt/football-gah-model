import numpy as np
import pandas as pd

from gah.totals import (
    matrix_to_total_pmf,
    negative_binomial_total_pmf,
    poisson_total_pmf,
    recency_weighted_total_stats,
)


def test_matrix_to_total_pmf():
    matrix = np.zeros((3, 3))
    matrix[0, 0] = 0.1
    matrix[1, 0] = 0.2
    matrix[0, 1] = 0.3
    matrix[1, 1] = 0.4

    pmf = matrix_to_total_pmf(matrix)
    assert abs(pmf[0] - 0.1) < 1e-12
    assert abs(pmf[1] - 0.5) < 1e-12
    assert abs(pmf[2] - 0.4) < 1e-12


def test_negative_binomial_falls_back_when_not_overdispersed():
    p1 = negative_binomial_total_pmf(2.8, 2.7)
    p2 = poisson_total_pmf(2.8)
    assert np.allclose(p1, p2)


def test_recency_weighting_favors_recent_scoring_regime():
    df = pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-12-30", "2025-12-31"],
                utc=True,
            ),
            "home_goals": [0, 0, 3, 4],
            "away_goals": [0, 1, 2, 2],
        }
    )
    mean, variance = recency_weighted_total_stats(
        df,
        as_of=pd.Timestamp("2026-01-01", tz="UTC"),
        half_life_days=30,
        recent_matches=None,
    )
    assert mean > 4.0
    assert variance >= 0.0
