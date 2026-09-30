from gah.data import parse_openfootball_json


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
