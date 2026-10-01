from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from gah.lifecycle import DecisionLedger, LifecycleEvent, LifecycleState


def parse_event(row: dict) -> LifecycleEvent:
    at = datetime.fromisoformat(row["at"])
    return LifecycleEvent(
        state=LifecycleState(row["state"]),
        at=at,
        reason=str(row["reason"]),
        source=str(row["source"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a GAH v2.6 decision ledger")
    ap.add_argument("json_file", help="JSON object with match_id and events")
    args = ap.parse_args()

    payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    ledger = DecisionLedger(
        match_id=str(payload["match_id"]),
        events=tuple(parse_event(row) for row in payload.get("events", [])),
    )
    ledger.validate()

    print(
        json.dumps(
            {
                "match_id": ledger.match_id,
                "current_state": ledger.current_state.value,
                "events": len(ledger.events),
                "terminal": ledger.current_state == LifecycleState.AUDITED,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
