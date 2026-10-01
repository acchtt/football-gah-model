from __future__ import annotations

import argparse
import hashlib
import json

from gah.market_aware import devig_two_way
from gah.prospective import (
    ProspectiveAHOutcome,
    airtable_outcome_fields,
    airtable_snapshot_fields,
    settlement_net_return,
    snapshot_from_two_way_market,
)


def make_snapshot_id(
    fixture_key: str,
    captured_at: str,
    side: str,
    line: float,
) -> str:
    raw = f"{fixture_key}|{captured_at}|{side}|{float(line):+.2f}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"v21-{digest}"


def snapshot_command(args: argparse.Namespace) -> dict:
    snapshot = snapshot_from_two_way_market(
        fixture_key=args.match_id,
        snapshot_time_utc=args.captured_at,
        competition=args.competition,
        home_team=args.home,
        away_team=args.away,
        side=args.side,
        line=args.line,
        selected_odds=args.odds,
        opposing_odds=args.opposing_odds,
        gah_probability=args.gah_probability,
    )
    snapshot_id = args.snapshot_id or make_snapshot_id(
        args.match_id,
        args.captured_at,
        args.side,
        args.line,
    )
    return airtable_snapshot_fields(
        snapshot,
        snapshot_id=snapshot_id,
        kickoff=args.kickoff,
        source=args.source,
    )


def outcome_command(args: argparse.Namespace) -> dict:
    closing_probability = None
    if args.closing_odds is not None and args.closing_opposing_odds is not None:
        closing_probability, _ = devig_two_way(
            args.closing_odds,
            args.closing_opposing_odds,
        )
    outcome = ProspectiveAHOutcome(
        fixture_key=args.match_id,
        captured_side=args.side,
        captured_line=args.captured_line,
        closing_line=args.closing_line,
        captured_market_probability=args.captured_market_probability,
        closing_market_probability=closing_probability,
        settlement_category=args.settlement,
        realized_net_return=settlement_net_return(
            args.settlement,
            args.captured_odds,
        ),
    )
    return airtable_outcome_fields(
        outcome,
        closing_at=args.closing_at,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="GAH v2.1 prospective AH snapshot/outcome formatter"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot")
    snap.add_argument("--snapshot-id")
    snap.add_argument("--match-id", required=True)
    snap.add_argument("--kickoff", required=True)
    snap.add_argument("--captured-at", required=True)
    snap.add_argument("--competition", required=True)
    snap.add_argument("--home", required=True)
    snap.add_argument("--away", required=True)
    snap.add_argument("--side", choices=["home", "away"], required=True)
    snap.add_argument("--line", type=float, required=True)
    snap.add_argument("--odds", type=float, required=True)
    snap.add_argument("--opposing-odds", type=float, required=True)
    snap.add_argument("--gah-probability", type=float, required=True)
    snap.add_argument("--source", required=True)
    snap.set_defaults(handler=snapshot_command)

    out = sub.add_parser("outcome")
    out.add_argument("--match-id", required=True)
    out.add_argument("--side", choices=["home", "away"], required=True)
    out.add_argument("--captured-line", type=float, required=True)
    out.add_argument("--captured-odds", type=float, required=True)
    out.add_argument("--captured-market-probability", type=float, required=True)
    out.add_argument("--closing-line", type=float)
    out.add_argument("--closing-odds", type=float)
    out.add_argument("--closing-opposing-odds", type=float)
    out.add_argument("--closing-at")
    out.add_argument(
        "--settlement",
        choices=["full_win", "half_win", "push", "half_loss", "full_loss"],
        required=True,
    )
    out.set_defaults(handler=outcome_command)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
