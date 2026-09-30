from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from benchmark_v20_selection import (
    DEFAULT_SEASONS,
    HOLDOUT_SEASONS,
    build_candidates,
)
from gah.market_data import load_football_data_seasons
from gah.reliability import MarketResidualReliability
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



def add_rolling_reliability(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Score each date using a reliability model fitted only on earlier dates.
    """
    rows = rows.sort_values(["match_date", "global_match_key", "market"]).copy()
    rows["reliability_probability"] = float("nan")
    rows["reliability_edge"] = float("nan")
    rows["reliability_ev"] = float("nan")

    history: list[dict] = []
    model = None
    last_fit_active_n = 0

    def active_count(items: list[dict]) -> int:
        return sum(
            1
            for item in items
            if str(item["actual_category"]) != "push"
        )

    for _, day in rows.groupby(rows["match_date"].dt.date, sort=True):
        active_n = active_count(history)
        if active_n >= 500 and (
            model is None or active_n - last_fit_active_n >= 250
        ):
            model = MarketResidualReliability(l2=10.0).fit(history)
            last_fit_active_n = active_n

        if model is not None:
            for idx, r in day.iterrows():
                p = model.predict_probability(
                    market=str(r["market"]),
                    line=float(r["line"]),
                    pure_probability=float(r["pure_probability"]),
                    market_probability=float(r["market_probability"]),
                )
                rows.at[idx, "reliability_probability"] = p
                rows.at[idx, "reliability_edge"] = (
                    p - float(r["market_probability"])
                )
                rows.at[idx, "reliability_ev"] = (
                    p * float(r["decimal_odds"]) - 1.0
                )

        history.extend(day.to_dict("records"))

    return rows


def select_reliability_board(
    rows: pd.DataFrame,
    max_matches: int = 8,
) -> pd.DataFrame:
    selected_indices: list[int] = []

    scored = rows[
        rows["reliability_ev"].notna()
        & (rows["reliability_ev"] > 0.0)
        & (rows["reliability_edge"] > 0.0)
    ]

    for _, day in scored.groupby(scored["match_date"].dt.date, sort=True):
        day = day.sort_values(
            ["reliability_ev", "reliability_edge"],
            ascending=False,
        )
        used_matches: set[str] = set()
        for idx, r in day.iterrows():
            key = str(r["global_match_key"])
            if key in used_matches:
                continue
            selected_indices.append(idx)
            used_matches.add(key)
            if len(used_matches) >= max_matches:
                break

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
    print("## Rolling OOS reliability board")
    print()
    reliable = add_rolling_reliability(rows)
    print(
        "Market probability is the baseline. The residual model is refit only "
        "on earlier dates using edge, pure confidence, market type and AH line magnitude."
    )
    print("Only positive predicted-EV candidates may enter; max 8 matches per date.")
    print()
    print("| Phase | Bets | ROI | Mean reliability edge | Mean predicted EV | Mean probability CLV | Positive CLV rate |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for phase in ["development", "holdout"]:
        board = select_reliability_board(
            reliable[reliable["phase"] == phase],
            max_matches=8,
        )
        if board.empty:
            continue
        s = summarize(board)
        print(
            f"| {phase} | {s['bets']} | {s['roi']:+.3f} | "
            f"{board['reliability_edge'].mean():+.4f} | "
            f"{board['reliability_ev'].mean():+.3f} | "
            f"{s['mean_clv']:+.4f} | {s['positive_clv_rate']:.3f} |"
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
