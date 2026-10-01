from __future__ import annotations

import json

from gah.data import load_openfootball_league_seasons
from gah.market_aware import model_effective_win_probability
from gah.markets import price_handicap
from gah.model import DixonColesModel


SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

FIXTURES = [
    {
        "competition": "BUNDESLIGA",
        "match_id": "bundesliga-20261009-dortmund-bremen",
        "home": "Borussia Dortmund",
        "away": "SV Werder Bremen",
        "line": -1.5,
        "selected_odds": 2.00,
        "opposing_odds": 1.92,
        "market_source": "OddStorm indexed 2026-10-01",
        "kickoff_utc": "2026-10-09T18:30:00+00:00",
    },
    {
        "competition": "LIGUE_1",
        "match_id": "ligue1-20261009-lens-lyon",
        "home": "Racing Club de Lens",
        "away": "Olympique Lyonnais",
        "line": -0.25,
        "selected_odds": 1.83,
        "opposing_odds": 1.91,
        "market_source": "Stake indexed 2026-10-01",
        "kickoff_utc": "2026-10-09T18:45:00+00:00",
    },
    {
        "competition": "LA_LIGA",
        "match_id": "laliga-20261009-malaga-espanyol",
        "home": "Málaga CF",
        "away": "RCD Espanyol de Barcelona",
        "line": 0.0,
        "selected_odds": 1.97,
        "opposing_odds": 1.96,
        "market_source": "OddStorm indexed 2026-10-01",
        "kickoff_utc": "2026-10-09T19:00:00+00:00",
    },
    {
        "competition": "SERIE_A",
        "match_id": "seriea-20261011-como-roma",
        "home": "Como 1907",
        "away": "AS Roma",
        "line": 0.0,
        "selected_odds": 1.95,
        "opposing_odds": 1.95,
        "market_source": "OddStorm indexed 2026-10-01",
        "kickoff_utc": "2026-10-11T10:30:00+00:00",
    },
]


def main() -> None:
    output = {"fixtures": []}

    for competition in ("BUNDESLIGA", "LIGUE_1", "LA_LIGA", "SERIE_A"):
        data = load_openfootball_league_seasons(SEASONS, competition)
        model = DixonColesModel(half_life_days=180.0, max_goals=10)
        summary = model.fit(data, as_of=data["match_date"].max())

        for row in [x for x in FIXTURES if x["competition"] == competition]:
            pred = model.predict(row["home"], row["away"])
            pricing = price_handicap(
                pred["score_matrix"],
                float(row["line"]),
                "home",
            )
            output["fixtures"].append(
                {
                    **row,
                    "side": "home",
                    "fit_date": model.fit_date_.isoformat(),
                    "completed_matches": int(summary.n_matches),
                    "teams": int(summary.n_teams),
                    "xg_home": pred["xg_home"],
                    "xg_away": pred["xg_away"],
                    "expected_margin": pred["expected_margin"],
                    "gah_probability": model_effective_win_probability(pricing),
                    "gah_fair_odds": pricing["fair_decimal_odds"],
                    "full_win": pricing["full_win"],
                    "half_win": pricing["half_win"],
                    "push": pricing["push"],
                    "half_loss": pricing["half_loss"],
                    "full_loss": pricing["full_loss"],
                }
            )

    print("GAH_FOUR_LEAGUE_FORWARD_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
