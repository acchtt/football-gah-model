from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


UNDERSTAT_LEAGUES = {
    "EPL": "ENG-Premier League",
    "BUNDESLIGA": "GER-Bundesliga",
    "LA_LIGA": "ESP-La Liga",
    "SERIE_A": "ITA-Serie A",
    "LIGUE_1": "FRA-Ligue 1",
}


def season_to_understat(season: str) -> str:
    """Convert 2021-22 to soccerdata/Understat's 2021/2022 form."""
    start_text, end_text = season.split("-", 1)
    start = int(start_text)
    if len(end_text) == 2:
        end = (start // 100) * 100 + int(end_text)
        if end <= start:
            end += 100
    else:
        end = int(end_text)
    return f"{start}/{end}"


def _normalise_understat_schedule(
    schedule: pd.DataFrame,
    competition: str,
) -> pd.DataFrame:
    """
    Normalise soccerdata Understat.read_schedule() output to the GAH schema.

    Kept separate from the network loader so schema handling is unit-testable.
    """
    df = schedule.reset_index().copy()

    rename = {
        "date": "match_date",
        "home_xg": "home_xg",
        "away_xg": "away_xg",
    }
    df = df.rename(columns=rename)

    required = [
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Understat schedule missing columns: {missing}")

    if "is_result" in df.columns:
        df = df[df["is_result"].fillna(False).astype(bool)]

    df = df.dropna(subset=required).copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["home_xg"] = df["home_xg"].astype(float)
    df["away_xg"] = df["away_xg"].astype(float)
    df["competition"] = competition.upper()

    if "season" not in df.columns:
        df["season"] = "unknown"
    else:
        df["season"] = df["season"].astype(str)

    return (
        df[
            [
                "match_date",
                "competition",
                "season",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "home_xg",
                "away_xg",
            ]
        ]
        .sort_values("match_date")
        .reset_index(drop=True)
    )


def load_understat_xg_seasons(
    seasons: list[str] | tuple[str, ...],
    competition: str,
) -> pd.DataFrame:
    """
    Download completed match-level goals + xG from Understat via soccerdata.

    soccerdata is an optional dependency installed with requirements-xg.txt.
    """
    try:
        import soccerdata as sd
    except ImportError as exc:  # pragma: no cover - exercised in xG CI
        raise RuntimeError(
            "soccerdata is required for Understat xG. "
            "Install requirements-xg.txt."
        ) from exc

    key = competition.upper()
    if key not in UNDERSTAT_LEAGUES:
        raise ValueError(
            f"Unsupported competition {competition!r}. "
            f"Supported: {sorted(UNDERSTAT_LEAGUES)}"
        )

    scraper = sd.Understat(
        leagues=[UNDERSTAT_LEAGUES[key]],
        seasons=[season_to_understat(season) for season in seasons],
        no_cache=True,
        no_store=True,
    )
    schedule = scraper.read_schedule(include_matches_without_data=False)
    return _normalise_understat_schedule(schedule, key)


def _shrunk_mean(
    values: list[float],
    prior: float,
    prior_matches: float = 5.0,
    window: int = 10,
) -> float:
    recent = values[-window:]
    if not recent:
        return float(prior)
    n = float(len(recent))
    return float(
        (sum(recent) + prior_matches * float(prior))
        / (n + prior_matches)
    )


@dataclass
class XGTeamState:
    """
    Sequential xG state for next-match intensities.

    Only already-completed matches are stored. Venue-specific xG-for/xG-against
    histories are shrunk toward the historical league home/away xG environment.
    """

    max_history: int = 30
    home_for: dict[str, list[float]] = field(default_factory=dict)
    home_against: dict[str, list[float]] = field(default_factory=dict)
    away_for: dict[str, list[float]] = field(default_factory=dict)
    away_against: dict[str, list[float]] = field(default_factory=dict)
    league_home_xg: list[float] = field(default_factory=list)
    league_away_xg: list[float] = field(default_factory=list)

    def _append(self, store: dict[str, list[float]], key: str, value: float) -> None:
        values = store.setdefault(key, [])
        values.append(float(value))
        if len(values) > self.max_history:
            del values[:-self.max_history]

    def update(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
    ) -> None:
        self._append(self.home_for, home_team, home_xg)
        self._append(self.home_against, home_team, away_xg)
        self._append(self.away_for, away_team, away_xg)
        self._append(self.away_against, away_team, home_xg)

        self.league_home_xg.append(float(home_xg))
        self.league_away_xg.append(float(away_xg))
        if len(self.league_home_xg) > 760:
            del self.league_home_xg[:-760]
            del self.league_away_xg[:-760]

    def league_rates(self) -> tuple[float, float]:
        if not self.league_home_xg:
            return 1.45, 1.20
        return (
            float(np.mean(self.league_home_xg)),
            float(np.mean(self.league_away_xg)),
        )

    def expected_xg(
        self,
        home_team: str,
        away_team: str,
        window: int = 10,
        prior_matches: float = 5.0,
    ) -> tuple[float, float]:
        league_home, league_away = self.league_rates()

        home_attack = _shrunk_mean(
            self.home_for.get(home_team, []),
            league_home,
            prior_matches=prior_matches,
            window=window,
        )
        away_defence = _shrunk_mean(
            self.away_against.get(away_team, []),
            league_home,
            prior_matches=prior_matches,
            window=window,
        )
        away_attack = _shrunk_mean(
            self.away_for.get(away_team, []),
            league_away,
            prior_matches=prior_matches,
            window=window,
        )
        home_defence = _shrunk_mean(
            self.home_against.get(home_team, []),
            league_away,
            prior_matches=prior_matches,
            window=window,
        )

        home_lambda = max(0.05, 0.5 * (home_attack + away_defence))
        away_lambda = max(0.05, 0.5 * (away_attack + home_defence))
        return float(home_lambda), float(away_lambda)



def _normalise_understat_team_stats(
    stats: pd.DataFrame,
    competition: str,
) -> pd.DataFrame:
    """Normalise Understat.read_team_match_stats() output."""
    df = stats.reset_index().copy()
    df = df.rename(columns={"date": "match_date"})

    required = [
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_np_xg",
        "away_np_xg",
        "home_ppda",
        "away_ppda",
        "home_deep_completions",
        "away_deep_completions",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Understat team stats missing columns: {missing}")

    df = df.dropna(
        subset=[
            "match_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "home_np_xg",
            "away_np_xg",
        ]
    ).copy()

    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    for column in ("home_goals", "away_goals"):
        df[column] = df[column].astype(int)
    for column in (
        "home_xg",
        "away_xg",
        "home_np_xg",
        "away_np_xg",
        "home_ppda",
        "away_ppda",
        "home_deep_completions",
        "away_deep_completions",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["competition"] = competition.upper()
    if "season" not in df.columns:
        df["season"] = "unknown"
    else:
        df["season"] = df["season"].astype(str)

    columns = [
        "match_date",
        "competition",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_np_xg",
        "away_np_xg",
        "home_ppda",
        "away_ppda",
        "home_deep_completions",
        "away_deep_completions",
    ]
    return df[columns].sort_values("match_date").reset_index(drop=True)


def load_understat_team_stats_seasons(
    seasons: list[str] | tuple[str, ...],
    competition: str,
) -> pd.DataFrame:
    """Download completed Understat team-match xG/context features."""
    try:
        import soccerdata as sd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soccerdata is required for Understat features. "
            "Install requirements-xg.txt."
        ) from exc

    key = competition.upper()
    if key not in UNDERSTAT_LEAGUES:
        raise ValueError(
            f"Unsupported competition {competition!r}. "
            f"Supported: {sorted(UNDERSTAT_LEAGUES)}"
        )

    scraper = sd.Understat(
        leagues=[UNDERSTAT_LEAGUES[key]],
        seasons=[season_to_understat(season) for season in seasons],
        no_cache=True,
        no_store=True,
    )
    stats = scraper.read_team_match_stats(force_cache=False)
    return _normalise_understat_team_stats(stats, key)


@dataclass
class NPXGTeamState:
    """
    Venue-specific non-penalty xG state with a historical penalty-xG allowance.

    This separates repeatable shot quality from the relatively sparse penalty
    component instead of treating penalty xG as team attacking form.
    """

    max_history: int = 30
    home_for: dict[str, list[float]] = field(default_factory=dict)
    home_against: dict[str, list[float]] = field(default_factory=dict)
    away_for: dict[str, list[float]] = field(default_factory=dict)
    away_against: dict[str, list[float]] = field(default_factory=dict)
    league_home_npxg: list[float] = field(default_factory=list)
    league_away_npxg: list[float] = field(default_factory=list)
    league_home_penalty_xg: list[float] = field(default_factory=list)
    league_away_penalty_xg: list[float] = field(default_factory=list)

    def _append(self, store: dict[str, list[float]], key: str, value: float) -> None:
        values = store.setdefault(key, [])
        values.append(float(value))
        if len(values) > self.max_history:
            del values[:-self.max_history]

    def update(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        home_np_xg: float,
        away_np_xg: float,
    ) -> None:
        self._append(self.home_for, home_team, home_np_xg)
        self._append(self.home_against, home_team, away_np_xg)
        self._append(self.away_for, away_team, away_np_xg)
        self._append(self.away_against, away_team, home_np_xg)

        self.league_home_npxg.append(float(home_np_xg))
        self.league_away_npxg.append(float(away_np_xg))
        self.league_home_penalty_xg.append(
            max(float(home_xg) - float(home_np_xg), 0.0)
        )
        self.league_away_penalty_xg.append(
            max(float(away_xg) - float(away_np_xg), 0.0)
        )

        for values in (
            self.league_home_npxg,
            self.league_away_npxg,
            self.league_home_penalty_xg,
            self.league_away_penalty_xg,
        ):
            if len(values) > 760:
                del values[:-760]

    def league_rates(self) -> tuple[float, float, float, float]:
        if not self.league_home_npxg:
            return 1.30, 1.05, 0.15, 0.15
        return (
            float(np.mean(self.league_home_npxg)),
            float(np.mean(self.league_away_npxg)),
            float(np.mean(self.league_home_penalty_xg)),
            float(np.mean(self.league_away_penalty_xg)),
        )

    def expected_xg(
        self,
        home_team: str,
        away_team: str,
        window: int = 10,
        prior_matches: float = 5.0,
    ) -> tuple[float, float]:
        league_home, league_away, pen_home, pen_away = self.league_rates()

        home_attack = _shrunk_mean(
            self.home_for.get(home_team, []),
            league_home,
            prior_matches,
            window,
        )
        away_defence = _shrunk_mean(
            self.away_against.get(away_team, []),
            league_home,
            prior_matches,
            window,
        )
        away_attack = _shrunk_mean(
            self.away_for.get(away_team, []),
            league_away,
            prior_matches,
            window,
        )
        home_defence = _shrunk_mean(
            self.home_against.get(home_team, []),
            league_away,
            prior_matches,
            window,
        )

        home_lambda = 0.5 * (home_attack + away_defence) + pen_home
        away_lambda = 0.5 * (away_attack + home_defence) + pen_away
        return max(float(home_lambda), 0.05), max(float(away_lambda), 0.05)
