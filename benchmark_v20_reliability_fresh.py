from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from benchmark_v20_global_board import (
    add_rolling_reliability,
    select_reliability_board,
)
from benchmark_v20_selection import build_candidates
from gah.market_data import load_football_data_seasons


COMPETITIONS = ["EPL", "BUNDESLIGA", "LA_LIGA", "SERIE_A", "LIGUE_1"]
FRESH_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]


def _load_one(competition: str) -> pd.DataFrame:
    data = load_football_data_seasons(FRESH_SEASONS, competition)
    rows = build_candidates(data)
    rows["competition"] = competition
    rows["global_match_key"] = competition + "|" + rows["match_key"].astype(str)
    return rows


def load_fresh_candidates() -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=2) as pool:
        frames = list(pool.map(_load_one, COMPETITIONS))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["match_date", "global_match_key", "market"])
        .reset_index(drop=True)
    )


def summarize(rows: pd.DataFrame) -> dict:
    clv = rows["clv_probability"].dropna()
    return {
        "bets": len(rows),
        "roi": float(rows["net_return"].mean()) if len(rows) else float("nan"),
        "mean_rel_edge": (
            float(rows["reliability_edge"].mean()) if len(rows) else float("nan")
        ),
        "mean_pred_ev": (
            float(rows["reliability_ev"].mean()) if len(rows) else float("nan")
        ),
        "mean_clv": float(clv.mean()) if len(clv) else float("nan"),
        "clv_n": len(clv),
        "positive_clv_rate": (
            float((clv > 0).mean()) if len(clv) else float("nan")
        ),
    }


def print_group(label: str, rows: pd.DataFrame) -> None:
    s = summarize(rows)
    print(
        f"| {label} | {s['bets']} | {s['roi']:+.3f} | "
        f"{s['mean_rel_edge']:+.4f} | {s['mean_pred_ev']:+.3f} | "
        f"{s['mean_clv']:+.4f} | {s['clv_n']} | "
        f"{s['positive_clv_rate']:.3f} |"
    )


def main() -> None:
    rows = load_fresh_candidates()
    scored = add_rolling_reliability(rows)
    board = select_reliability_board(scored, max_matches=8)

    print("# GAH v2.0 Frozen Reliability — Fresh Validation")
    print()
    print(
        "Frozen stage-3 specification applied without modification to "
        "Football-Data 2016/17-2020/21."
    )
    print()
    print(
        "| Slice | Bets | ROI | Mean residual edge | Mean predicted EV | "
        "Mean probability CLV | CLV N | Positive CLV rate |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    print_group("ALL", board)

    for season in FRESH_SEASONS:
        group = board[board["season"] == season]
        if len(group):
            print_group(season, group)

    print()
    print("## Competition / market composition")
    print()
    print(
        "| Competition | Market | Bets | ROI | Mean probability CLV | "
        "CLV N | Positive CLV rate |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for (competition, market), group in board.groupby(
        ["competition", "market"],
        sort=True,
    ):
        s = summarize(group)
        print(
            f"| {competition} | {market} | {s['bets']} | "
            f"{s['roi']:+.3f} | {s['mean_clv']:+.4f} | "
            f"{s['clv_n']} | {s['positive_clv_rate']:.3f} |"
        )

    print()
    print("## Selected rows")
    print()
    cols = [
        "match_date",
        "competition",
        "season",
        "market",
        "side",
        "line",
        "decimal_odds",
        "market_probability",
        "pure_probability",
        "reliability_probability",
        "reliability_edge",
        "reliability_ev",
        "net_return",
        "clv_probability",
    ]
    if board.empty:
        print("No selections.")
    else:
        print(board[cols].to_string(index=False))


if __name__ == "__main__":
    main()
