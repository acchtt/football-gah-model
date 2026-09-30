from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gah.backtest import (
    _fit_total_blend_weight,
    _independent_poisson_matrix,
    _total_probability,
)
from gah.market_aware import devig_two_way, model_effective_win_probability
from gah.market_data import load_football_data_seasons
from gah.markets import (
    expected_value_at_odds,
    price_handicap,
    price_total,
    settle_handicap,
    settle_total,
)
from gah.model import DixonColesModel
from gah.selection import (
    SelectionCandidate,
    edge_band,
    realized_net_return,
    select_board,
)
from gah.totals import (
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    tilt_score_matrix_by_total,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
HOLDOUT_SEASONS = {"2024-25", "2025-26"}
THRESHOLDS = [0.00, 0.02, 0.04, 0.06, 0.10]


def _valid_odds(*values) -> bool:
    try:
        nums = [float(x) for x in values]
    except (TypeError, ValueError):
        return False
    return all(np.isfinite(x) and x > 1.0 for x in nums)


def _valid_line(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _candidate_record(
    *,
    target,
    candidate: SelectionCandidate,
    actual_category: str,
    close_selected_probability: float | None,
) -> dict:
    return {
        "match_key": candidate.match_key,
        "match_date": target["match_date"],
        "season": target["season"],
        "phase": "holdout" if target["season"] in HOLDOUT_SEASONS else "development",
        "market": candidate.market,
        "side": candidate.side,
        "line": candidate.line,
        "pure_probability": candidate.pure_probability,
        "market_probability": candidate.market_probability,
        "edge": candidate.edge,
        "abs_edge": candidate.abs_edge,
        "band": edge_band(candidate.edge),
        "decimal_odds": candidate.decimal_odds,
        "model_ev": candidate.model_ev,
        "actual_category": actual_category,
        "net_return": realized_net_return(actual_category, candidate.decimal_odds),
        "close_selected_probability": close_selected_probability,
        "clv_probability": (
            None
            if close_selected_probability is None
            else float(close_selected_probability - candidate.market_probability)
        ),
    }


def build_candidates(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    hist_model_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []
    records: list[dict] = []

    for i in range(380, len(data)):
        target = data.iloc[i]
        train = data.iloc[:i]

        known = set(train["home_team"]).union(train["away_team"])
        if target["home_team"] not in known or target["away_team"] not in known:
            continue

        needs_refit = (
            model is None
            or last_fit_i is None
            or (i - last_fit_i) >= 20
            or target["home_team"] not in model.team_to_idx_
            or target["away_team"] not in model.team_to_idx_
        )
        if needs_refit:
            model = DixonColesModel(half_life_days=180.0, max_goals=10)
            model.fit(train, as_of=train["match_date"].max())
            last_fit_i = i

        pred = model.predict(target["home_team"], target["away_team"])
        raw_matrix = pred["score_matrix"]

        naive_matrix = _independent_poisson_matrix(
            float(train["home_goals"].mean()),
            float(train["away_goals"].mean()),
            max_goals=10,
        )

        dc_weight = 1.0
        if len(hist_model_prob) >= 100:
            start = max(0, len(hist_model_prob) - 380)
            dc_weight = _fit_total_blend_weight(
                hist_model_prob[start:],
                hist_naive_prob[start:],
            )

        base_totals = dc_weight * raw_matrix + (1.0 - dc_weight) * naive_matrix
        base_totals /= base_totals.sum()
        base_pmf = matrix_to_total_pmf(base_totals)

        beta = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            beta = fit_total_tilt_beta(
                hist_base_pmfs[start:],
                hist_actual_totals[start:],
            )
        totals_matrix = tilt_score_matrix_by_total(base_totals, beta)

        hg = int(target["home_goals"])
        ag = int(target["away_goals"])
        margin = hg - ag
        total = hg + ag
        match_key = (
            f"{target['match_date'].date()}|"
            f"{target['home_team']}|{target['away_team']}"
        )

        if (
            "AHh" in target.index
            and "AvgAHH" in target.index
            and "AvgAHA" in target.index
            and _valid_line(target["AHh"])
            and _valid_odds(target["AvgAHH"], target["AvgAHA"])
        ):
            home_line = float(target["AHh"])
            market_home, market_away = devig_two_way(
                float(target["AvgAHH"]),
                float(target["AvgAHA"]),
            )
            home_pricing = price_handicap(raw_matrix, home_line, "home")
            pure_home = model_effective_win_probability(home_pricing)

            if pure_home >= market_home:
                side = "home"
                line = home_line
                odds = float(target["AvgAHH"])
                pricing = home_pricing
                pure_selected = pure_home
                market_selected = market_home
                actual = settle_handicap(margin, line, "home")
                selected_home = True
            else:
                side = "away"
                line = -home_line
                odds = float(target["AvgAHA"])
                pricing = price_handicap(raw_matrix, line, "away")
                pure_selected = model_effective_win_probability(pricing)
                market_selected = market_away
                actual = settle_handicap(margin, line, "away")
                selected_home = False

            close_selected = None
            if (
                "AHCh" in target.index
                and "AvgCAHH" in target.index
                and "AvgCAHA" in target.index
                and _valid_line(target["AHCh"])
                and abs(float(target["AHCh"]) - home_line) < 1e-9
                and _valid_odds(target["AvgCAHH"], target["AvgCAHA"])
            ):
                close_home, close_away = devig_two_way(
                    float(target["AvgCAHH"]),
                    float(target["AvgCAHA"]),
                )
                close_selected = close_home if selected_home else close_away

            candidate = SelectionCandidate(
                match_key=match_key,
                market="AH",
                side=side,
                line=line,
                pure_probability=pure_selected,
                market_probability=market_selected,
                decimal_odds=odds,
                model_ev=expected_value_at_odds(pricing, odds),
            )
            records.append(
                _candidate_record(
                    target=target,
                    candidate=candidate,
                    actual_category=actual,
                    close_selected_probability=close_selected,
                )
            )

        if (
            "Avg>2.5" in target.index
            and "Avg<2.5" in target.index
            and _valid_odds(target["Avg>2.5"], target["Avg<2.5"])
        ):
            market_over, market_under = devig_two_way(
                float(target["Avg>2.5"]),
                float(target["Avg<2.5"]),
            )
            over_pricing = price_total(totals_matrix, 2.5, "over")
            pure_over = model_effective_win_probability(over_pricing)

            if pure_over >= market_over:
                side = "over"
                odds = float(target["Avg>2.5"])
                pricing = over_pricing
                pure_selected = pure_over
                market_selected = market_over
                actual = settle_total(total, 2.5, "over")
                selected_over = True
            else:
                side = "under"
                odds = float(target["Avg<2.5"])
                pricing = price_total(totals_matrix, 2.5, "under")
                pure_selected = model_effective_win_probability(pricing)
                market_selected = market_under
                actual = settle_total(total, 2.5, "under")
                selected_over = False

            close_selected = None
            if (
                "AvgC>2.5" in target.index
                and "AvgC<2.5" in target.index
                and _valid_odds(target["AvgC>2.5"], target["AvgC<2.5"])
            ):
                close_over, close_under = devig_two_way(
                    float(target["AvgC>2.5"]),
                    float(target["AvgC<2.5"]),
                )
                close_selected = close_over if selected_over else close_under

            candidate = SelectionCandidate(
                match_key=match_key,
                market="TOTAL25",
                side=side,
                line=2.5,
                pure_probability=pure_selected,
                market_probability=market_selected,
                decimal_odds=odds,
                model_ev=expected_value_at_odds(pricing, odds),
            )
            records.append(
                _candidate_record(
                    target=target,
                    candidate=candidate,
                    actual_category=actual,
                    close_selected_probability=close_selected,
                )
            )

        hist_model_prob.append(_total_probability(raw_matrix, total))
        hist_naive_prob.append(_total_probability(naive_matrix, total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(total)

    return pd.DataFrame(records)


def _summarize(group: pd.DataFrame) -> dict:
    clv = group["clv_probability"].dropna()
    return {
        "n": len(group),
        "roi": float(group["net_return"].mean()) if len(group) else float("nan"),
        "mean_edge": float(group["abs_edge"].mean()) if len(group) else float("nan"),
        "mean_model_ev": float(group["model_ev"].mean()) if len(group) else float("nan"),
        "mean_clv": float(clv.mean()) if len(clv) else float("nan"),
        "clv_n": len(clv),
    }


def _daily_board_rows(
    rows: pd.DataFrame,
    threshold: float,
    max_matches: int = 8,
) -> pd.DataFrame:
    selected_indices: list[int] = []
    for _, day in rows.groupby(rows["match_date"].dt.date, sort=True):
        candidates = [
            SelectionCandidate(
                match_key=str(r["match_key"]),
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
            (
                c.match_key,
                c.market,
                c.side,
                round(c.line, 4),
            )
            for c in picked
        }
        for idx, r in day.iterrows():
            key = (
                str(r["match_key"]),
                str(r["market"]),
                str(r["side"]),
                round(float(r["line"]), 4),
            )
            if key in keys:
                selected_indices.append(idx)
    return rows.loc[selected_indices].copy()


def build_report(competition: str, rows: pd.DataFrame) -> str:
    lines = [
        f"# GAH v2.0 Selection Screen — {competition}",
        "",
        "Selection direction is always the GAH side of disagreement versus the",
        "de-vigged opening market. Disagreement thresholds are fixed in advance.",
        "2024/25 and 2025/26 are reported as a separate holdout.",
        "",
        "## Fixed disagreement bands",
        "",
        "| Phase | Market | Band | N | ROI | Mean edge | Mean model EV | Mean probability CLV |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    band_order = ["<2pp", "2-4pp", "4-6pp", "6-10pp", "10pp+"]
    for phase in ["development", "holdout"]:
        for market in ["AH", "TOTAL25"]:
            subset = rows[(rows["phase"] == phase) & (rows["market"] == market)]
            for band in band_order:
                group = subset[subset["band"] == band]
                if group.empty:
                    continue
                s = _summarize(group)
                lines.append(
                    f"| {phase} | {market} | {band} | {s['n']} | "
                    f"{s['roi']:+.3f} | {s['mean_edge']:.3f} | "
                    f"{s['mean_model_ev']:+.3f} | {s['mean_clv']:+.4f} |"
                )

    lines.extend(
        [
            "",
            "## Capped daily boards",
            "",
            "At most 8 matches per day, one market per match, ranked by absolute",
            "GAH-vs-market disagreement then model EV.",
            "",
            "| Phase | Min edge | Bets | ROI | Mean edge | Mean model EV | Mean probability CLV |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for phase in ["development", "holdout"]:
        phase_rows = rows[rows["phase"] == phase]
        for threshold in THRESHOLDS:
            board = _daily_board_rows(phase_rows, threshold, max_matches=8)
            if board.empty:
                continue
            s = _summarize(board)
            lines.append(
                f"| {phase} | {threshold:.2f} | {s['n']} | {s['roi']:+.3f} | "
                f"{s['mean_edge']:.3f} | {s['mean_model_ev']:+.3f} | "
                f"{s['mean_clv']:+.4f} |"
            )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_football_data_seasons(args.seasons, competition)
    rows = build_candidates(data)
    print(build_report(competition, rows))


if __name__ == "__main__":
    main()
