from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


STATSBOMB_RAW_BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"


def _read_json(url: str):
    req = Request(url, headers={"User-Agent": "football-gah-model/1.8"})
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def load_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    payload = _read_json(
        f"{STATSBOMB_RAW_BASE}/matches/{int(competition_id)}/{int(season_id)}.json"
    )
    rows = []
    for match in payload:
        if match.get("home_score") is None or match.get("away_score") is None:
            continue
        rows.append(
            {
                "match_id": int(match["match_id"]),
                "match_date": pd.to_datetime(match["match_date"], utc=True),
                "season": match["season"]["season_name"],
                "home_team": match["home_team"]["home_team_name"],
                "away_team": match["away_team"]["away_team_name"],
                "home_team_id": int(match["home_team"]["home_team_id"]),
                "away_team_id": int(match["away_team"]["away_team_id"]),
                "home_goals": int(match["home_score"]),
                "away_goals": int(match["away_score"]),
            }
        )
    if not rows:
        raise ValueError("No completed StatsBomb matches found.")
    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)


def load_statsbomb_starting_xi(match_id: int) -> dict[int, tuple[int, ...]]:
    payload = _read_json(f"{STATSBOMB_RAW_BASE}/lineups/{int(match_id)}.json")
    out: dict[int, tuple[int, ...]] = {}
    for team in payload:
        starters = []
        for player in team.get("lineup", []):
            positions = player.get("positions") or []
            if any(p.get("start_reason") == "Starting XI" for p in positions):
                starters.append(int(player["player_id"]))
        if len(starters) == 11:
            out[int(team["team_id"])] = tuple(sorted(starters))
    return out


def load_statsbomb_starting_xis(
    match_ids,
    max_workers: int = 8,
) -> tuple[dict[int, dict[int, tuple[int, ...]]], dict]:
    """Fetch many StatsBomb lineup files concurrently for benchmark use."""
    ids = [int(x) for x in match_ids]
    results: dict[int, dict[int, tuple[int, ...]]] = {}
    errors: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        futures = {
            pool.submit(load_statsbomb_starting_xi, match_id): match_id
            for match_id in ids
        }
        for future in as_completed(futures):
            match_id = futures[future]
            try:
                results[match_id] = future.result()
            except Exception as exc:
                results[match_id] = {}
                if len(errors) < 10:
                    errors[match_id] = f"{type(exc).__name__}: {exc}"

    diagnostics = {
        "requested": len(ids),
        "fetch_errors": sum(1 for match_id in ids if not results.get(match_id) and match_id in errors),
        "complete_lineup_files": sum(
            1 for match_id in ids if len(results.get(match_id, {})) == 2
        ),
        "first_errors": errors,
    }
    return results, diagnostics


class LineupContinuityState:
    """
    Leakage-safe rolling starting-XI state.

    Coverage is the fraction of the team's recent 'core starter mass' that is
    represented by the confirmed current XI. It uses only earlier lineups.
    """

    def __init__(self, window: int = 10, min_history: int = 5) -> None:
        self.window = int(window)
        self.min_history = int(min_history)
        self._history: dict[int, deque[tuple[int, ...]]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def match_count(self, team_id: int) -> int:
        return len(self._history[int(team_id)])

    def update(self, team_id: int, starters: tuple[int, ...] | list[int]) -> None:
        unique = tuple(sorted(set(int(x) for x in starters)))
        if len(unique) != 11:
            raise ValueError("A confirmed starting XI must contain exactly 11 players.")
        self._history[int(team_id)].append(unique)

    def coverage(
        self,
        team_id: int,
        starters: tuple[int, ...] | list[int],
    ) -> float:
        history = self._history[int(team_id)]
        if len(history) < self.min_history:
            return float("nan")

        counts: dict[int, int] = defaultdict(int)
        for xi in history:
            for player_id in xi:
                counts[player_id] += 1

        denom = float(len(history))
        start_rates = {pid: count / denom for pid, count in counts.items()}
        core_mass = sum(sorted(start_rates.values(), reverse=True)[:11])
        if core_mass <= 0:
            return float("nan")

        current_mass = sum(start_rates.get(int(pid), 0.0) for pid in starters)
        return float(np.clip(current_mass / core_mass, 0.0, 1.0))


def tilt_score_matrix_by_margin(matrix: np.ndarray, beta: float) -> np.ndarray:
    base = np.asarray(matrix, dtype=float)
    h, a = np.indices(base.shape)
    tilted = base * np.exp(float(beta) * (h - a))
    total = float(tilted.sum())
    if total <= 0:
        raise ValueError("Tilted score matrix has no probability mass.")
    return tilted / total


def margin_probability(matrix: np.ndarray, actual_margin: int) -> float:
    base = np.asarray(matrix, dtype=float)
    h, a = np.indices(base.shape)
    return float(base[(h - a) == int(actual_margin)].sum())


def fit_lineup_margin_scale(
    matrices: list[np.ndarray],
    coverage_gaps: list[float] | np.ndarray,
    actual_margins: list[int] | np.ndarray,
    bound: float = 0.35,
) -> float:
    """
    Fit beta_i = alpha * (home_coverage - away_coverage) using prior OOS rows.
    """
    gaps = np.asarray(coverage_gaps, dtype=float)
    margins = np.asarray(actual_margins, dtype=int)
    if len(matrices) != len(gaps) or len(gaps) != len(margins):
        raise ValueError("Inputs must align.")
    if len(gaps) < 50:
        raise ValueError("At least 50 OOS rows are required.")

    def objective(alpha: float) -> float:
        losses = []
        for matrix, gap, actual in zip(matrices, gaps, margins):
            tilted = tilt_score_matrix_by_margin(matrix, float(alpha) * float(gap))
            p = margin_probability(tilted, int(actual))
            losses.append(-np.log(max(p, 1e-12)))
        return float(np.mean(losses))

    result = minimize_scalar(
        objective,
        bounds=(-float(bound), float(bound)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)


def reshape_score_matrix_by_margin_uncertainty(
    matrix: np.ndarray,
    disruption: float,
    gamma: float,
) -> np.ndarray:
    """
    Change margin dispersion while preserving the base expected margin.

    Positive gamma widens the goal-margin distribution as disruption rises;
    negative gamma narrows it. A compensating linear margin tilt is solved so
    the transformed matrix keeps the original expected goal margin.
    """
    base = np.asarray(matrix, dtype=float)
    h, a = np.indices(base.shape)
    margin = (h - a).astype(float)
    base_mean = float(np.sum(base * margin))
    centred_sq = np.square(margin - base_mean)
    strength = float(gamma) * float(disruption)

    def transformed(lam: float) -> np.ndarray:
        weights = np.exp(
            np.clip(strength * centred_sq + float(lam) * margin, -20.0, 20.0)
        )
        out = base * weights
        return out / out.sum()

    def mean_error(lam: float) -> float:
        out = transformed(lam)
        return abs(float(np.sum(out * margin)) - base_mean)

    solved = minimize_scalar(
        mean_error,
        bounds=(-1.5, 1.5),
        method="bounded",
        options={"xatol": 1e-7},
    )
    return transformed(float(solved.x))


def fit_lineup_uncertainty_scale(
    matrices: list[np.ndarray],
    disruptions: list[float] | np.ndarray,
    actual_margins: list[int] | np.ndarray,
    bound: float = 0.12,
) -> float:
    """
    Fit a single OOS disruption-to-margin-dispersion scale.

    This is intentionally non-directional: confirmed XI information can only
    alter uncertainty, not which side is expected to be stronger.
    """
    d = np.asarray(disruptions, dtype=float)
    margins = np.asarray(actual_margins, dtype=int)
    if len(matrices) != len(d) or len(d) != len(margins):
        raise ValueError("Inputs must align.")
    if len(d) < 50:
        raise ValueError("At least 50 OOS rows are required.")

    def objective(gamma: float) -> float:
        losses = []
        for matrix, disruption, actual in zip(matrices, d, margins):
            shaped = reshape_score_matrix_by_margin_uncertainty(
                matrix,
                float(disruption),
                float(gamma),
            )
            p = margin_probability(shaped, int(actual))
            losses.append(-np.log(max(p, 1e-12)))
        return float(np.mean(losses))

    result = minimize_scalar(
        objective,
        bounds=(-float(bound), float(bound)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)
