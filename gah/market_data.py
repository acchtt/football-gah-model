from __future__ import annotations

from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd


FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

LEAGUE_CODES = {
    "EPL": "E0",
    "BUNDESLIGA": "D1",
    "LA_LIGA": "SP1",
    "SERIE_A": "I1",
    "LIGUE_1": "F1",
}


def season_code(season: str) -> str:
    """Convert 2025-26 -> 2526."""
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


def football_data_url(season: str, competition: str) -> str:
    key = competition.upper()
    if key not in LEAGUE_CODES:
        raise ValueError(f"Unsupported competition: {competition}")
    return f"{FOOTBALL_DATA_BASE}/{season_code(season)}/{LEAGUE_CODES[key]}.csv"


def _read_csv(url: str) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "football-gah-model/1.9"})
    with urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(StringIO(raw))


def load_football_data_season(
    season: str,
    competition: str,
) -> pd.DataFrame:
    df = _read_csv(football_data_url(season, competition))
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing Football-Data columns: {sorted(missing)}")

    out = df.copy()
    out["match_date"] = pd.to_datetime(
        out["Date"],
        dayfirst=True,
        errors="coerce",
        utc=True,
    )
    out["season"] = season
    out["competition"] = competition.upper()
    out["home_team"] = out["HomeTeam"].astype(str)
    out["away_team"] = out["AwayTeam"].astype(str)
    out["home_goals"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["FTAG"], errors="coerce")

    numeric_columns = [
        "AHh", "AvgAHH", "AvgAHA",
        "AHCh", "AvgCAHH", "AvgCAHA",
        "Avg>2.5", "Avg<2.5",
        "AvgC>2.5", "AvgC<2.5",
    ]
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(
        subset=["match_date", "home_goals", "away_goals"]
    ).copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    return out.sort_values("match_date").reset_index(drop=True)


def load_football_data_seasons(
    seasons: list[str],
    competition: str,
) -> pd.DataFrame:
    frames = [
        load_football_data_season(season, competition)
        for season in seasons
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("match_date")
        .reset_index(drop=True)
    )
