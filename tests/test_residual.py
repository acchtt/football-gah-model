import numpy as np
import pandas as pd
from scipy.stats import poisson

from gah.residual import ConditionalTotalTiltModel, TeamResidualState


def test_team_state_is_leakage_safe_and_updates_after_match():
    state = TeamResidualState()
    date = pd.Timestamp("2026-01-10", tz="UTC")

    before = state.features("Home", "Away", date)
    assert np.allclose(before, 0.0)

    state.update(
        "Home",
        "Away",
        date,
        home_goals=3,
        away_goals=1,
        dc_xg_home=1.5,
        dc_xg_away=1.0,
        base_total_mean=2.6,
    )

    after = state.features(
        "Home",
        "Away",
        pd.Timestamp("2026-01-17", tz="UTC"),
    )
    assert after[0] > 0.0  # home attack exceeded DC expectation
    assert after[4] == 0.0  # away attack matched DC expectation
    assert after[8] > 0.0  # match total exceeded base total mean
    assert after[14] > 0.0
    assert after[15] > 0.0


def test_conditional_tilt_learns_positive_feature_for_high_totals():
    goals = np.arange(8)
    base = poisson.pmf(goals, 2.5)
    base = base / base.sum()

    x = []
    pmfs = []
    actual = []

    for _ in range(30):
        x.append([-1.0])
        pmfs.append(base.copy())
        actual.append(1)
    for _ in range(30):
        x.append([1.0])
        pmfs.append(base.copy())
        actual.append(4)

    model = ConditionalTotalTiltModel(l2=0.05, beta_clip=0.2)
    model.fit(np.asarray(x), pmfs, actual)

    assert model.predict_beta(np.array([1.0])) > 0.0
    assert model.predict_beta(np.array([-1.0])) < 0.0
