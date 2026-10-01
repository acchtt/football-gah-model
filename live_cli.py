from __future__ import annotations

import argparse
import json

from gah.live import LiveAdjustment, LiveState, live_ah_table, reprice_live_ah


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.3 regulation-time live AH repricing"
    )
    ap.add_argument("--minute", type=float, required=True)
    ap.add_argument("--home-score", type=int, required=True)
    ap.add_argument("--away-score", type=int, required=True)
    ap.add_argument("--pre-home-xg", type=float, required=True)
    ap.add_argument("--pre-away-xg", type=float, required=True)
    ap.add_argument("--side", choices=["home", "away"], required=True)
    ap.add_argument("--line", type=float, required=True)
    ap.add_argument("--odds", type=float)
    ap.add_argument("--table-line", type=float, action="append", default=[])
    ap.add_argument("--home-rate-multiplier", type=float, default=1.0)
    ap.add_argument("--away-rate-multiplier", type=float, default=1.0)
    ap.add_argument("--event", action="append", default=[])
    ap.add_argument("--adjustment-reason")
    args = ap.parse_args()

    adjustment = LiveAdjustment(
        home_rate_multiplier=args.home_rate_multiplier,
        away_rate_multiplier=args.away_rate_multiplier,
        material_events=tuple(args.event),
        reason=args.adjustment_reason,
    )
    state = LiveState(
        minute=args.minute,
        home_score=args.home_score,
        away_score=args.away_score,
        pre_match_home_xg=args.pre_home_xg,
        pre_match_away_xg=args.pre_away_xg,
        adjustment=adjustment,
    )

    quote = reprice_live_ah(
        state,
        side=args.side,
        line=args.line,
        decimal_odds=args.odds,
    )

    payload = {
        "state": {
            "minute": state.minute,
            "score": [state.home_score, state.away_score],
            "remaining_home_xg": state.remaining_home_xg,
            "remaining_away_xg": state.remaining_away_xg,
            "material_events": list(state.adjustment.material_events),
            "adjustment_reason": state.adjustment.reason,
        },
        "quote": {
            "side": quote.side,
            "line": quote.line,
            "effective_win_probability": quote.effective_win_probability,
            "fair_decimal_odds": quote.fair_decimal_odds,
            "expected_value_at_current_odds": quote.expected_value,
            "settlement_probabilities": {
                key: quote.pricing[key]
                for key in (
                    "full_win",
                    "half_win",
                    "push",
                    "half_loss",
                    "full_loss",
                )
            },
        },
    }

    if args.table_line:
        payload["line_table"] = [
            {
                "side": row.side,
                "line": row.line,
                "effective_win_probability": row.effective_win_probability,
                "fair_decimal_odds": row.fair_decimal_odds,
            }
            for row in live_ah_table(
                state,
                side=args.side,
                lines=tuple(args.table_line),
            )
        ]

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
