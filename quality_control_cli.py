from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from gah.board_capacity import BoardCandidate
from gah.lifecycle import DecisionLedger, LifecycleEvent, LifecycleState
from gah.quality_control import operational_health


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_ledgers(path: str | None) -> list[DecisionLedger]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("ledgers", payload) if isinstance(payload, dict) else payload
    result = []
    for row in rows:
        events = tuple(
            LifecycleEvent(
                state=LifecycleState(event["state"]),
                at=dt(event["at"]),
                reason=str(event["reason"]),
                source=str(event["source"]),
            )
            for event in row.get("events", [])
        )
        result.append(DecisionLedger(str(row["match_id"]), events))
    return result


def parse_board(path: str | None) -> list[BoardCandidate]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = (
        payload.get("selected", payload)
        if isinstance(payload, dict)
        else payload
    )
    return [
        BoardCandidate(
            match_id=str(row["match_id"]),
            kickoff=dt(row["kickoff"]),
            priority=int(row["priority"]),
            monitor_minutes_before=int(row.get("monitor_minutes_before", 60)),
            monitor_minutes_after=int(row.get("monitor_minutes_after", 120)),
        )
        for row in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.10 operational quality-control report"
    )
    ap.add_argument("prospective_csv")
    ap.add_argument("--ledgers-json")
    ap.add_argument("--selected-board-json")
    ap.add_argument("--as-of")
    ap.add_argument("--stale-grace-hours", type=float, default=12.0)
    ap.add_argument("--max-matches", type=int, default=8)
    ap.add_argument("--max-concurrent", type=int, default=3)
    args = ap.parse_args()

    frame = pd.read_csv(args.prospective_csv)
    report = operational_health(
        frame,
        ledgers=parse_ledgers(args.ledgers_json),
        selected_board_candidates=parse_board(args.selected_board_json),
        max_matches=args.max_matches,
        max_concurrent=args.max_concurrent,
        as_of=None if args.as_of is None else dt(args.as_of),
        stale_grace_hours=args.stale_grace_hours,
    )
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
