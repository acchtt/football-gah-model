import pandas as pd

from gah.xg import (
    XGTeamState,
    _normalise_understat_schedule,
    season_to_understat,
)


def test_season_to_understat():
    assert season_to_understat("2021-22") == "2021/2022"
    assert season_to_understat("2025-26") == "2025/2026"


def test_normalise_understat_schedule():
    idx = pd.MultiIndex.from_tuples(
        [("ENG-Premier League", "2526", "game-1")],
        names=["league", "season", "game"],
    )
    raw = pd.DataFrame(
        {
            "date": ["2025-08-15 20:00:00"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [2],
            "away_goals": [1],
            "home_xg": [1.8],
            "away_xg": [0.9],
            "is_result": [True],
        },
        index=idx,
    )

    df = _normalise_understat_schedule(raw, "EPL")
    assert len(df) == 1
    assert df.iloc[0]["competition"] == "EPL"
    assert df.iloc[0]["home_xg"] == 1.8
    assert str(df.iloc[0]["match_date"].tz) == "UTC"


def test_xg_team_state_uses_team_and_opponent_information():
    state = XGTeamState()

    for _ in range(8):
        state.update("StrongHome", "WeakAway", 2.2, 0.6)
        state.update("OtherHome", "TargetAway", 2.0, 0.7)
        state.update("TargetHome", "OtherAway", 1.0, 1.8)
        state.update("OtherHome2", "StrongAway", 0.7, 2.0)

    strong_home, _ = state.expected_xg("StrongHome", "TargetAway")
    weak_home, _ = state.expected_xg("TargetHome", "WeakAway")
    assert strong_home > weak_home

    _, strong_away = state.expected_xg("TargetHome", "StrongAway")
    _, weak_away = state.expected_xg("StrongHome", "WeakAway")
    assert strong_away > weak_away
