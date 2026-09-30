from gah.data import (
    openfootball_league_url,
    parse_openfootball_json,
)


def test_parse_openfootball_completed_matches_only():
    payload = {
        "matches": [
            {
                "date": "2025-08-01",
                "team1": "A",
                "team2": "B",
                "score": {"ft": [2, 1]},
            },
            {
                "date": "2025-08-02",
                "team1": "C",
                "team2": "D",
                "score": {},
            },
        ]
    }

    df = parse_openfootball_json(payload, season="2025-26")
    assert len(df) == 1
    assert df.iloc[0]["home_team"] == "A"
    assert df.iloc[0]["home_goals"] == 2
    assert df.iloc[0]["away_goals"] == 1
    assert df.iloc[0]["season"] == "2025-26"


def test_parse_openfootball_direct_score_list():
    payload = {
        "matches": [
            {
                "date": "2024-08-10",
                "team1": "A",
                "team2": "B",
                "score": [3, 2],
            }
        ]
    }

    df = parse_openfootball_json(payload, season="2024-25")
    assert len(df) == 1
    assert df.iloc[0]["home_goals"] == 3
    assert df.iloc[0]["away_goals"] == 2


def test_major_league_url_mapping():
    assert openfootball_league_url("2025-26", "EPL").endswith("/2025-26/en.1.json")
    assert openfootball_league_url("2025-26", "BUNDESLIGA").endswith("/2025-26/de.1.json")
    assert openfootball_league_url("2025-26", "LA_LIGA").endswith("/2025-26/es.1.json")
    assert openfootball_league_url("2025-26", "SERIE_A").endswith("/2025-26/it.1.json")
    assert openfootball_league_url("2025-26", "LIGUE_1").endswith("/2025-26/fr.1.json")
