from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from benchmark_v20_selection import (
    DEFAULT_SEASONS,
    HOLDOUT_SEASONS,
    build_candidates,
)
from gah.market_data import load_football_data_seasons
from gah.selection import SelectionCandidate, select_board


COMPETITIONS = ["EPL", "BUNDESLIGA", "LA_LIGA", "SERIE_A", "LIGUE_1"]
THRESHOLDS = [0.00, 0.02, 0.04, 0.06, 0.10]


def _load_competition_candidates(competition: str) -> pd.DataFrame:
    data = load_football_data_seasons(DEFAULT_SEASONS, competition)
    rows = build_candidates(data)
    rows["competition"] = competition
    rows["global_match_key"] = (
        competition + "|" + rows["match_key"].astype(str)
    )
    return rows


def load_all_candidates() -> pd.DataFrame:
    # Each competition is independent. Parallelizing here preserves the
    # experiment exactly while avoiding a five-league serial bottleneck.
    with ThreadPoolExecutor(max_workers=2) as pool:
        frames = list(pool.map(_load_competition_candidates, COMPETITIONS))
    return pd.concat(frames, ignore_index=True)


def select_global_daily_board(
    rows: pd.DataFrame,
    threshold: float,
    max_matches: int = 8,
) -> pd.DataFrame:
    selected_indices: list[int] = []

    for _, day in rows.groupby(rows["match_date"].dt.date, sort=True):
        candidates = [
            SelectionCandidate(
                match_key=str(r["global_match_key"]),
                market=str(r["market"]),
                side=str(r["side"]),
                line=float(r["line"]),
                pure_probability=float(r["pure_probability"]),
                market_probability=float(r["market_probability"]),
                decimal_odds=float(r["decimal_odds"]),
                model_ev=float(r["model_ev"]),
            )
            for _, r in day.iterrows()
        ]

        picked = select_board(
            candidates,
            max_matches=max_matches,
            min_abs_edge=threshold,
        )
        keys = {
            (c.match_key, c.market, c.side, round(c.line, 4))
            for c in picked
        }

        for idx, r in day.iterrows():
            key = (
                str(r["global_match_key"]),
                str(r["market"]),
                str(r["side"]),
                round(float(r["line"]), 4),
            )
            if key in keys:
                selected_indices.append(idx)

    return rows.loc[selected_indices].copy()


def summarize(rows: pd.DataFrame) -> dict:
    clv = rows["clv_probability"].dropna()
    return {
        "bets": len(rows),
        "roi": float(rows["net_return"].mean()),
        "mean_edge": float(rows["abs_edge"].mean()),
        "mean_ev": float(rows["model_ev"].mean()),
        "mean_clv": float(clv.mean()) if len(clv) else float("nan"),
        "positive_clv_rate": (
            float((clv > 0).mean()) if len(clv) else float("nan")
        ),
    }


def main() -> None:
    rows = load_all_candidates()

    print("# GAH v2.0 Global Cross-League Board")
    print()
    print(
        "At most 8 total matches per date across EPL, Bundesliga, La Liga, "
        "Serie A and Ligue 1. One market per match."
    )
    print(
        "2021/22–2023/24 = development; 2024/25–2025/26 = untouched holdout."
    )
    print()
    print(
        "| Phase | Min edge | Bets | ROI | Mean edge | Mean model EV | "
        "Mean probability CLV | Positive CLV rate |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for phase in ["development", "holdout"]:
        phase_rows = rows[rows["phase"] == phase]
        for threshold in THRESHOLDS:
            board = select_global_daily_board(
                phase_rows,
                threshold,
                max_matches=8,
            )
            if board.empty:
                continue
            s = summarize(board)
            print(
                f"| {phase} | {threshold:.2f} | {s['bets']} | "
                f"{s['roi']:+.3f} | {s['mean_edge']:.3f} | "
                f"{s['mean_ev']:+.3f} | {s['mean_clv']:+.4f} | "
                f"{s['positive_clv_rate']:.3f} |"
            )

    print()
    print("## Holdout composition at 10pp+")
    print()
    holdout = rows[rows["phase"] == "holdout"]
    board = select_global_daily_board(holdout, 0.10, max_matches=8)
    if board.empty:
        print("No selections.")
        return

    print("| Competition | Market | N | ROI | Mean probability CLV |")
    print("|---|---|---:|---:|---:|")
    for (competition, market), group in board.groupby(
        ["competition", "market"],
        sort=True,
    ):
        s = summarize(group)
        print(
            f"| {competition} | {market} | {s['bets']} | "
            f"{s['roi']:+.3f} | {s['mean_clv']:+.4f} |"
        )


if __name__ == "__main__":
    main()
