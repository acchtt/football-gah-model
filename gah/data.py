from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd


OPENFOOTBALL_RAW_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/master"
)

LEAGUE_FILES = {
    "EPL": "en.1.json",
    "BUNDESLIGA": "de.1.json",
    "LA_LIGA": "es.1.json",
    "SERIE_A": "it.1.json",
    "LIGUE_1": "fr.1.json",
}


def openfootball_league_url(season: str, competition: str) -> str:
    key = competition.upper()
    if key not in LEAGUE_FILES:
        raise ValueError(
            f"Unsupported competition {competition!r}. "
            f"Supported: {sorted(LEAGUE_FILES)}"
        )
    return f"{OPENFOOTBALL_RAW_BASE}/{season}/{LEAGUE_FILES[key]}"


def openfootball_epl_url(season: str) -> str:
    return openfootball_league_url(season, "EPL")


def _read_json_source(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        req = Request(
            source,
            headers={"User-Agent": "football-gah-model/1.0"},
        )
        with urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    return json.loads(Path(source).read_text(encoding="utf-8"))


def parse_openfootball_json(
    payload: dict,
    season: str,
    competition: str = "EPL",
) -> pd.DataFrame:
    rows: list[dict] = []

    for match in payload.get("matches", []):
        score = match.get("score")
        if isinstance(score, list):
            ft = score
        elif isinstance(score, dict):
            ft = score.get("ft")
        else:
            ft = None

        if not isinstance(ft, list) or len(ft) != 2:
            continue
        if ft[0] is None or ft[1] is None:
            continue

        date = match.get("date")
        if not date:
            continue

        rows.append(
            {
                "match_date": pd.to_datetime(date, utc=True),
                "competition": competition,
                "season": season,
                "home_team": match["team1"],
                "away_team": match["team2"],
                "home_goals": int(ft[0]),
                "away_goals": int(ft[1]),
            }
        )

    if not rows:
        raise ValueError(f"No completed matches found for {competition} {season}")

    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)


def load_openfootball_league_season(
    season: str,
    competition: str,
) -> pd.DataFrame:
    key = competition.upper()
    payload = _read_json_source(openfootball_league_url(season, key))
    return parse_openfootball_json(
        payload,
        season=season,
        competition=key,
    )


def load_openfootball_league_seasons(
    seasons: Iterable[str],
    competition: str,
) -> pd.DataFrame:
    frames = [
        load_openfootball_league_season(season, competition)
        for season in seasons
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("match_date")
        .reset_index(drop=True)
    )


def load_openfootball_epl_season(season: str) -> pd.DataFrame:
    return load_openfootball_league_season(season, "EPL")


def load_openfootball_epl_seasons(seasons: Iterable[str]) -> pd.DataFrame:
    return load_openfootball_league_seasons(seasons, "EPL")
