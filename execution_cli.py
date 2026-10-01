from __future__ import annotations

import argparse
import json

from gah.execution import DecayStep, ExecutionPlan, evaluate_execution


def parse_decay(value: str) -> DecayStep:
    try:
        minutes, line, odds = value.split(",", 2)
        return DecayStep(
            activates_at_minutes_to_kickoff=int(minutes),
            minimum_line=float(line),
            minimum_odds=float(odds),
        )
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Decay must be MINUTES,LINE,ODDS (example: 60,-0.75,1.85)"
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a frozen GAH v2.2 AH execution plan"
    )
    ap.add_argument("--side", choices=["home", "away"], required=True)
    ap.add_argument("--protected-line", type=float, required=True)
    ap.add_argument("--protected-min-odds", type=float, required=True)
    ap.add_argument("--expiry-minutes", type=int, default=0)
    ap.add_argument(
        "--decay",
        type=parse_decay,
        action="append",
        default=[],
        help="Repeatable MINUTES,LINE,ODDS decay step.",
    )
    ap.add_argument("--current-line", type=float, required=True)
    ap.add_argument("--current-odds", type=float, required=True)
    ap.add_argument("--minutes-to-kickoff", type=int, required=True)
    args = ap.parse_args()

    plan = ExecutionPlan(
        side=args.side,
        protected_line=args.protected_line,
        protected_minimum_odds=args.protected_min_odds,
        expiry_minutes_to_kickoff=args.expiry_minutes,
        decay_steps=tuple(args.decay),
    )
    decision = evaluate_execution(
        plan=plan,
        current_line=args.current_line,
        current_odds=args.current_odds,
        minutes_to_kickoff=args.minutes_to_kickoff,
    )
    print(
        json.dumps(
            {
                "state": decision.state.value,
                "active_minimum_line": decision.active_minimum_line,
                "active_minimum_odds": decision.active_minimum_odds,
                "minutes_to_kickoff": decision.minutes_to_kickoff,
                "line_ok": decision.line_ok,
                "odds_ok": decision.odds_ok,
                "reason": decision.reason,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
