from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gah.backtest import _fit_total_blend_weight
from gah.data import load_openfootball_epl_seasons
from gah.markets import price_handicap
from gah.model import DixonColesModel
from gah.totals import (
    absolute_error_optimal_point,
    matrix_to_total_pmf,
    mix_pmfs,
    negative_binomial_total_pmf,
    over_probability,
    pmf_mean,
    pmf_probability,
    poisson_total_pmf,
    recency_weighted_total_stats,
)


DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def _nll(p: float) -> float:
    return -math.log(max(float(p), 1e-12))


def _fit_weight(hist_a: list[float], hist_b: list[float], min_hist=100, window=380) -> float:
    if len(hist_a) < min_hist:
        return 1.0
    start = max(0, len(hist_a) - window)
    return _fit_total_blend_weight(hist_a[start:], hist_b[start:])


def run_experiment(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["match_date"] = pd.to_datetime(data["match_date"], utc=True)
    data = data.sort_values("match_date").reset_index(drop=True)

    model = None
    last_fit_i = None
    rows = []

    # v1.1 history: DC vs expanding league Poisson.
    h_dc_v11: list[float] = []
    h_expanding: list[float] = []

    # v1.2 regime-Poisson history: DC vs recency-weighted Poisson.
    h_dc_regime_pois: list[float] = []
    h_regime_pois: list[float] = []

    # v1.2 regime-NB history: DC vs recency-weighted negative binomial.
    h_dc_regime_nb: list[float] = []
    h_regime_nb: list[float] = []

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
        dc_matrix = pred["score_matrix"]
        dc_total = matrix_to_total_pmf(dc_matrix)

        # v1.1 baseline: expanding league scoring mean.
        expanding_mean = float(
            (train["home_goals"] + train["away_goals"]).mean()
        )
        expanding_pmf = poisson_total_pmf(expanding_mean)

        # v1.2 scoring regime: exponentially weighted recent scoring environment.
        regime_mean, regime_var = recency_weighted_total_stats(
            train,
            as_of=train["match_date"].max(),
            half_life_days=180.0,
            recent_matches=760,
        )
        regime_poisson = poisson_total_pmf(regime_mean)
        regime_nb = negative_binomial_total_pmf(regime_mean, regime_var)

        w_v11 = _fit_weight(h_dc_v11, h_expanding)
        w_regime_pois = _fit_weight(h_dc_regime_pois, h_regime_pois)
        w_regime_nb = _fit_weight(h_dc_regime_nb, h_regime_nb)

        pmf_v11 = mix_pmfs(dc_total, expanding_pmf, w_v11)
        pmf_regime_pois = mix_pmfs(dc_total, regime_poisson, w_regime_pois)
        pmf_regime_nb = mix_pmfs(dc_total, regime_nb, w_regime_nb)

        actual_total = int(target["home_goals"] + target["away_goals"])
        actual_margin = int(target["home_goals"] - target["away_goals"])
        over_label = 1.0 if actual_total >= 3 else 0.0
        home_m05_label = 1.0 if actual_margin > 0 else 0.0

        dc_prob = pmf_probability(dc_total, actual_total)
        expanding_prob = pmf_probability(expanding_pmf, actual_total)
        regime_pois_prob = pmf_probability(regime_poisson, actual_total)
        regime_nb_prob = pmf_probability(regime_nb, actual_total)

        ah = price_handicap(dc_matrix, -0.5, "home")

        rows.append(
            {
                "season": target["season"],
                "match_date": target["match_date"],
                "actual_total": actual_total,
                "v11_mean": pmf_mean(pmf_v11),
                "v11_median": absolute_error_optimal_point(pmf_v11),
                "regime_pois_mean": pmf_mean(pmf_regime_pois),
                "regime_nb_mean": pmf_mean(pmf_regime_nb),
                "v11_mae": abs(pmf_mean(pmf_v11) - actual_total),
                "v11_median_mae": abs(absolute_error_optimal_point(pmf_v11) - actual_total),
                "regime_pois_mae": abs(pmf_mean(pmf_regime_pois) - actual_total),
                "regime_nb_mae": abs(pmf_mean(pmf_regime_nb) - actual_total),
                "v11_nll": _nll(pmf_probability(pmf_v11, actual_total)),
                "regime_pois_nll": _nll(pmf_probability(pmf_regime_pois, actual_total)),
                "regime_nb_nll": _nll(pmf_probability(pmf_regime_nb, actual_total)),
                "v11_o25_brier": (over_probability(pmf_v11, 2.5) - over_label) ** 2,
                "regime_pois_o25_brier": (over_probability(pmf_regime_pois, 2.5) - over_label) ** 2,
                "regime_nb_o25_brier": (over_probability(pmf_regime_nb, 2.5) - over_label) ** 2,
                "v11_weight": w_v11,
                "regime_pois_weight": w_regime_pois,
                "regime_nb_weight": w_regime_nb,
                "regime_mean": regime_mean,
                "regime_variance": regime_var,
                "home_m05_brier": (ah["full_win"] - home_m05_label) ** 2,
                "margin_abs_error": abs(pred["expected_margin"] - actual_margin),
            }
        )

        # Update OOS histories only after the prediction was evaluated.
        h_dc_v11.append(dc_prob)
        h_expanding.append(expanding_prob)

        h_dc_regime_pois.append(dc_prob)
        h_regime_pois.append(regime_pois_prob)

        h_dc_regime_nb.append(dc_prob)
        h_regime_nb.append(regime_nb_prob)

    return pd.DataFrame(rows)


def _avg(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean())


def improvement(new: float, old: float) -> float:
    return 100.0 * (old - new) / old


def build_report(results: pd.DataFrame) -> str:
    v11_mae = _avg(results, "v11_mae")
    rp_mae = _avg(results, "regime_pois_mae")
    rn_mae = _avg(results, "regime_nb_mae")

    v11_median_mae = _avg(results, "v11_median_mae")

    v11_nll = _avg(results, "v11_nll")
    rp_nll = _avg(results, "regime_pois_nll")
    rn_nll = _avg(results, "regime_nb_nll")

    v11_brier = _avg(results, "v11_o25_brier")
    rp_brier = _avg(results, "regime_pois_o25_brier")
    rn_brier = _avg(results, "regime_nb_o25_brier")

    lines = [
        "# EPL GAH v1.2 Regime + Overdispersion Experiment",
        "",
        "AH remains frozen. This experiment changes only the totals baseline inside",
        "the leakage-safe online blend.",
        "",
        "## Variants",
        "",
        "- **v1.1**: DC + expanding league-average Poisson",
        "- **Regime-Poisson**: DC + 180-day exponentially weighted scoring-regime Poisson",
        "- **Regime-NB**: DC + same scoring-regime mean with empirical negative-binomial overdispersion",
        "",
        "## Overall",
        "",
        "| Metric | v1.1 | Regime-Poisson | Regime-NB | RP vs v1.1 | NB vs v1.1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Total MAE ↓ | {v11_mae:.4f} | {rp_mae:.4f} | {rn_mae:.4f} | {improvement(rp_mae, v11_mae):+.2f}% | {improvement(rn_mae, v11_mae):+.2f}% |",
        f"| Total NLL ↓ | {v11_nll:.4f} | {rp_nll:.4f} | {rn_nll:.4f} | {improvement(rp_nll, v11_nll):+.2f}% | {improvement(rn_nll, v11_nll):+.2f}% |",
        f"| O2.5 Brier ↓ | {v11_brier:.4f} | {rp_brier:.4f} | {rn_brier:.4f} | {improvement(rp_brier, v11_brier):+.2f}% | {improvement(rn_brier, v11_brier):+.2f}% |",
        "",
        "## Point-forecast correction",
        "",
        f"- v1.1 predictive mean MAE: **{v11_mae:.4f}**",
        f"- v1.1 predictive median MAE: **{v11_median_mae:.4f}**",
        f"- Median vs mean improvement: **{improvement(v11_median_mae, v11_mae):+.2f}%**",
        "",
        f"Mean blend weight on DC — v1.1: **{_avg(results, 'v11_weight'):.3f}**",
        f"Mean blend weight on DC — Regime-Poisson: **{_avg(results, 'regime_pois_weight'):.3f}**",
        f"Mean blend weight on DC — Regime-NB: **{_avg(results, 'regime_nb_weight'):.3f}**",
        f"Mean regime total: **{_avg(results, 'regime_mean'):.3f}**",
        f"Mean regime variance: **{_avg(results, 'regime_variance'):.3f}**",
        "",
        "## AH guardrail",
        "",
        f"- Goal-margin MAE: **{_avg(results, 'margin_abs_error'):.4f}**",
        f"- Home -0.5 Brier: **{_avg(results, 'home_m05_brier'):.4f}**",
        "",
        "## By season",
        "",
        "| Season | v1.1 mean MAE | v1.1 median MAE | RP MAE | NB MAE | v1.1 O2.5 | RP O2.5 | NB O2.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for season, g in results.groupby("season", sort=True):
        lines.append(
            f"| {season} | "
            f"{_avg(g, 'v11_mae'):.4f} | "
            f"{_avg(g, 'v11_median_mae'):.4f} | "
            f"{_avg(g, 'regime_pois_mae'):.4f} | "
            f"{_avg(g, 'regime_nb_mae'):.4f} | "
            f"{_avg(g, 'v11_o25_brier'):.4f} | "
            f"{_avg(g, 'regime_pois_o25_brier'):.4f} | "
            f"{_avg(g, 'regime_nb_o25_brier'):.4f} |"
        )

    lines += [
        "",
        "## Promotion rule",
        "",
        "Promote only a variant that improves total MAE and does not materially worsen",
        "total NLL or O2.5 Brier relative to v1.1. AH must remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--output", default="benchmarks/epl_v12_regime_overdispersion.md")
    args = ap.parse_args()

    data = load_openfootball_epl_seasons(args.seasons)
    results = run_experiment(data)

    report = build_report(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
