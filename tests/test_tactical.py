import numpy as np
from scipy.stats import poisson

from gah.tactical import ConditionalTacticalTilt, TacticalTeamState


def test_tactical_state_uses_only_prior_updates():
    state = TacticalTeamState()
    before = state.features("A", "B")
    assert np.allclose(before, 0.0)

    state.update(
        "A", "B",
        home_np_xg=2.0,
        away_np_xg=0.5,
        home_ppda=7.0,
        away_ppda=14.0,
        home_deep=9.0,
        away_deep=3.0,
    )
    after = state.features("A", "B")
    assert after[0] > after[4]
    assert after[8] > after[9]
    assert after[10] > after[11]


def test_tactical_tilt_learns_directional_signal():
    goals = np.arange(8)
    base = poisson.pmf(goals, 2.5)
    base = base / base.sum()

    x = []
    pmfs = []
    actual = []
    for _ in range(40):
        x.append([-1.0, 0.0])
        pmfs.append(base.copy())
        actual.append(1)
    for _ in range(40):
        x.append([1.0, 0.0])
        pmfs.append(base.copy())
        actual.append(4)

    model = ConditionalTacticalTilt(l2=0.1, beta_clip=0.1)
    model.fit(np.asarray(x), pmfs, actual)

    assert model.predict_beta(np.array([1.0, 0.0])) > 0
    assert model.predict_beta(np.array([-1.0, 0.0])) < 0
