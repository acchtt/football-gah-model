import pandas as pd

from gah.xg import (
    NPXGTeamState,
    XGTeamState,
    _normalise_understat_team_stats,
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


def test_normalise_understat_team_stats():
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
            "home_np_xg": [1.1],
            "away_np_xg": [0.9],
            "home_ppda": [8.5],
            "away_ppda": [12.0],
            "home_deep_completions": [8],
            "away_deep_completions": [4],
        },
        index=idx,
    )
    df = _normalise_understat_team_stats(raw, "EPL")
    assert len(df) == 1
    assert df.iloc[0]["home_np_xg"] == 1.1
    assert df.iloc[0]["home_deep_completions"] == 8


def test_npxg_state_keeps_penalty_component_as_league_prior():
    state = NPXGTeamState()
    for _ in range(10):
        state.update("A", "B", 1.76, 0.90, 1.00, 0.90)
        state.update("C", "D", 1.00, 1.76, 1.00, 1.00)

    home, away = state.expected_xg("A", "B")
    assert home > 1.0
    assert away > 0.8
