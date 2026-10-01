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
        "match_id": "epl-20261010-arsenal-leeds",
        "home": "Arsenal FC",
        "away": "Leeds United FC",
        "line": -1.25,
        "selected_odds": 1.85,
        "opposing_odds": 1.89,
        "market_source": "Stake indexed 2026-10-01",
    },
    {
        "match_id": "epl-20261010-aston-villa-brentford",
        "home": "Aston Villa FC",
        "away": "Brentford FC",
        "line": 0.0,
        "selected_odds": 1.92,
        "opposing_odds": 1.92,
        "market_source": "Bet365 via SportScore indexed 2026-10-01",
    },
    {
        "match_id": "epl-20261010-chelsea-bournemouth",
        "home": "Chelsea FC",
        "away": "AFC Bournemouth",
        "line": -0.75,
        "selected_odds": 1.81,
        "opposing_odds": 1.93,
        "market_source": "Stake indexed 2026-10-01",
    },
    {
        "match_id": "epl-20261010-ipswich-fulham",
        "home": "Ipswich Town FC",
        "away": "Fulham FC",
        "line": 0.0,
        "selected_odds": 1.97,
        "opposing_odds": 1.87,
        "market_source": "Bet365 via SportScore indexed 2026-10-01",
    },
    {
        "match_id": "epl-20261010-sunderland-brighton",
        "home": "Sunderland AFC",
        "away": "Brighton & Hove Albion FC",
        "line": 0.0,
        "selected_odds": 2.02,
        "opposing_odds": 1.82,
        "market_source": "AiScore opening feed indexed 2026-10-01",
    },
    {
        "match_id": "epl-20261010-man-utd-tottenham",
        "home": "Manchester United FC",
        "away": "Tottenham Hotspur FC",
        "line": -0.75,
        "selected_odds": 1.84,
        "opposing_odds": 1.90,
        "market_source": "Stake indexed 2026-10-01",
    },
]


def main() -> None:
    data = load_openfootball_league_seasons(SEASONS, "EPL")
    model = DixonColesModel(half_life_days=180.0, max_goals=10)
    summary = model.fit(data, as_of=data["match_date"].max())

    output = {
        "competition": "EPL",
        "seasons": SEASONS,
        "fit_date": model.fit_date_.isoformat(),
        "completed_matches": int(summary.n_matches),
        "teams": int(summary.n_teams),
        "fixtures": [],
    }

    for row in FIXTURES:
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

    print("GAH_FIRST_FORWARD_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
