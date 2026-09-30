from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gah.backtest import (
    _fit_total_blend_weight,
    _independent_poisson_matrix,
    _total_probability,
)
from gah.market_aware import (
    blend_probability,
    devig_two_way,
    fit_market_blend_weight,
    model_effective_win_probability,
    realized_exposure,
    weighted_binary_brier,
)
from gah.market_data import load_football_data_seasons
from gah.markets import (
    price_handicap,
    price_total,
    settle_handicap,
    settle_total,
)
from gah.model import DixonColesModel
from gah.totals import (
    fit_total_tilt_beta,
    matrix_to_total_pmf,
    tilt_score_matrix_by_total,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def _valid_odds(*values) -> bool:
    try:
        nums = [float(x) for x in values]
    except (TypeError, ValueError):
        return False
    return all(np.isfinite(x) and x > 1.0 for x in nums)


def _valid_line(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x))


def _score(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return weighted_binary_brier(
        [r[key] for r in rows],
        [r["outcome"] for r in rows],
        [r["exposure"] for r in rows],
    )


def run_benchmark(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None

    hist_model_prob: list[float] = []
    hist_naive_prob: list[float] = []
    hist_base_pmfs: list[np.ndarray] = []
    hist_actual_totals: list[int] = []

    ah_hist: list[dict] = []
    total_hist: list[dict] = []

    rows: list[dict] = []
    weight_rows: list[dict] = []

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

        naive_home_rate = float(train["home_goals"].mean())
        naive_away_rate = float(train["away_goals"].mean())
        naive_matrix = _independent_poisson_matrix(
            naive_home_rate,
            naive_away_rate,
            max_goals=10,
        )

        dc_weight = 1.0
        if len(hist_model_prob) >= 100:
            start = max(0, len(hist_model_prob) - 380)
            dc_weight = _fit_total_blend_weight(
                hist_model_prob[start:],
                hist_naive_prob[start:],
            )

        base_totals_matrix = (
            dc_weight * raw_matrix
            + (1.0 - dc_weight) * naive_matrix
        )
        base_totals_matrix /= base_totals_matrix.sum()
        base_pmf = matrix_to_total_pmf(base_totals_matrix)

        beta = 0.0
        if len(hist_base_pmfs) >= 100:
            start = max(0, len(hist_base_pmfs) - 380)
            beta = fit_total_tilt_beta(
                hist_base_pmfs[start:],
                hist_actual_totals[start:],
            )
        totals_matrix = tilt_score_matrix_by_total(base_totals_matrix, beta)

        hg = int(target["home_goals"])
        ag = int(target["away_goals"])
        actual_margin = hg - ag
        actual_total = hg + ag

        ah_market_weight = 0.0
        if len(ah_hist) >= 100:
            window = ah_hist[-380:]
            ah_market_weight = fit_market_blend_weight(
                [r["pure_probability"] for r in window],
                [r["market_probability"] for r in window],
                [r["outcome"] for r in window],
                [r["exposure"] for r in window],
            )

        total_market_weight = 0.0
        if len(total_hist) >= 100:
            window = total_hist[-380:]
            total_market_weight = fit_market_blend_weight(
                [r["pure_probability"] for r in window],
                [r["market_probability"] for r in window],
                [r["outcome"] for r in window],
                [r["exposure"] for r in window],
            )

        if (
            "AHh" in target.index
            and "AvgAHH" in target.index
            and "AvgAHA" in target.index
            and _valid_line(target["AHh"])
            and _valid_odds(target["AvgAHH"], target["AvgAHA"])
        ):
            line = float(target["AHh"])
            market_p, _ = devig_two_way(
                float(target["AvgAHH"]),
                float(target["AvgAHA"]),
            )
            pure_p = model_effective_win_probability(
                price_handicap(raw_matrix, line, "home")
            )
            actual = settle_handicap(actual_margin, line, "home")
            exposure = realized_exposure(actual)
            if exposure is not None:
                outcome, stake = exposure
                row = {
                    "season": target["season"],
                    "market": "AH_OPEN",
                    "pure_probability": pure_p,
                    "market_probability": market_p,
                    "blend_probability": blend_probability(
                        pure_p,
                        market_p,
                        ah_market_weight,
                    ),
                    "outcome": outcome,
                    "exposure": stake,
                    "blend_weight": ah_market_weight,
                }
                rows.append(row)
                ah_hist.append(row)

        if (
            "Avg>2.5" in target.index
            and "Avg<2.5" in target.index
            and _valid_odds(target["Avg>2.5"], target["Avg<2.5"])
        ):
            market_p, _ = devig_two_way(
                float(target["Avg>2.5"]),
                float(target["Avg<2.5"]),
            )
            pure_p = model_effective_win_probability(
                price_total(totals_matrix, 2.5, "over")
            )
            actual = settle_total(actual_total, 2.5, "over")
            outcome, stake = realized_exposure(actual)
            row = {
                "season": target["season"],
                "market": "TOTAL25_OPEN",
                "pure_probability": pure_p,
                "market_probability": market_p,
                "blend_probability": blend_probability(
                    pure_p,
                    market_p,
                    total_market_weight,
                ),
                "outcome": outcome,
                "exposure": stake,
                "blend_weight": total_market_weight,
            }
            rows.append(row)
            total_hist.append(row)

        if (
            "AHCh" in target.index
            and "AvgCAHH" in target.index
            and "AvgCAHA" in target.index
            and _valid_line(target["AHCh"])
            and _valid_odds(target["AvgCAHH"], target["AvgCAHA"])
        ):
            line = float(target["AHCh"])
            close_p, _ = devig_two_way(
                float(target["AvgCAHH"]),
                float(target["AvgCAHA"]),
            )
            pure_p = model_effective_win_probability(
                price_handicap(raw_matrix, line, "home")
            )
            actual = settle_handicap(actual_margin, line, "home")
            exposure = realized_exposure(actual)
            if exposure is not None:
                outcome, stake = exposure
                rows.append(
                    {
                        "season": target["season"],
                        "market": "AH_CLOSE_REFERENCE",
                        "pure_probability": pure_p,
                        "market_probability": close_p,
                        "blend_probability": close_p,
                        "outcome": outcome,
                        "exposure": stake,
                        "blend_weight": 1.0,
                    }
                )

        if (
            "AvgC>2.5" in target.index
            and "AvgC<2.5" in target.index
            and _valid_odds(target["AvgC>2.5"], target["AvgC<2.5"])
        ):
            close_p, _ = devig_two_way(
                float(target["AvgC>2.5"]),
                float(target["AvgC<2.5"]),
            )
            pure_p = model_effective_win_probability(
                price_total(totals_matrix, 2.5, "over")
            )
            actual = settle_total(actual_total, 2.5, "over")
            outcome, stake = realized_exposure(actual)
            rows.append(
                {
                    "season": target["season"],
                    "market": "TOTAL25_CLOSE_REFERENCE",
                    "pure_probability": pure_p,
                    "market_probability": close_p,
                    "blend_probability": close_p,
                    "outcome": outcome,
                    "exposure": stake,
                    "blend_weight": 1.0,
                }
            )

        weight_rows.append(
            {
                "season": target["season"],
                "ah_market_weight": ah_market_weight,
                "total_market_weight": total_market_weight,
            }
        )

        hist_model_prob.append(_total_probability(raw_matrix, actual_total))
        hist_naive_prob.append(_total_probability(naive_matrix, actual_total))
        hist_base_pmfs.append(base_pmf)
        hist_actual_totals.append(actual_total)

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def build_report(
    competition: str,
    rows: pd.DataFrame,
    weights: pd.DataFrame,
) -> str:
    lines = [
        f"# GAH v1.9 Market-Aware — {competition}",
        "",
        "Opening market averages are the only bookmaker input to the actionable",
        "blend. Closing market averages are reported as reference only.",
        "",
        "## Actionable opening markets",
        "",
        "| Market | N | GAH Pure Brier | Market Brier | Market-Aware Brier | Mean market weight |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for market in ["AH_OPEN", "TOTAL25_OPEN"]:
        group = rows[rows["market"] == market]
        if group.empty:
            continue
        pure = weighted_binary_brier(
            group["pure_probability"],
            group["outcome"],
            group["exposure"],
        )
        market_b = weighted_binary_brier(
            group["market_probability"],
            group["outcome"],
            group["exposure"],
        )
        blend = weighted_binary_brier(
            group["blend_probability"],
            group["outcome"],
            group["exposure"],
        )
        lines.append(
            f"| {market} | {len(group)} | {pure:.4f} | {market_b:.4f} | "
            f"{blend:.4f} | {group['blend_weight'].mean():.3f} |"
        )

    lines.extend(
        [
            "",
            "## Closing market reference",
            "",
            "| Market | N | GAH Pure Brier | Closing market Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for market in ["AH_CLOSE_REFERENCE", "TOTAL25_CLOSE_REFERENCE"]:
        group = rows[rows["market"] == market]
        if group.empty:
            continue
        pure = weighted_binary_brier(
            group["pure_probability"],
            group["outcome"],
            group["exposure"],
        )
        market_b = weighted_binary_brier(
            group["market_probability"],
            group["outcome"],
            group["exposure"],
        )
        lines.append(
            f"| {market} | {len(group)} | {pure:.4f} | {market_b:.4f} |"
        )

    if not weights.empty:
        lines.extend(
            [
                "",
                f"Mean OOS-fitted AH opening-market weight: **{weights['ah_market_weight'].mean():.3f}**",
                f"Mean OOS-fitted O2.5 opening-market weight: **{weights['total_market_weight'].mean():.3f}**",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    competition = args.competition.upper()
    data = load_football_data_seasons(args.seasons, competition)
    rows, weights = run_benchmark(data)
    print(build_report(competition, rows, weights))


if __name__ == "__main__":
    main()
