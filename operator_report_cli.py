from __future__ import annotations

import argparse
from datetime import datetime
import json

from gah.airtable_bridge import AirtableBridgeClient
from gah.airtable_state import load_airtable_state, load_state_export
from gah.operator_report import build_operator_report


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.13 compact forward-operations report"
    )
    ap.add_argument(
        "--state",
        help="Optional v2.11 state export; otherwise load live Airtable.",
    )
    ap.add_argument(
        "--as-of",
        help="Optional timezone-aware ISO timestamp for deterministic QA.",
    )
    ap.add_argument(
        "--stale-grace-hours",
        type=float,
        default=12.0,
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="Print one-line operator status instead of JSON.",
    )
    args = ap.parse_args()

    state = (
        load_state_export(args.state)
        if args.state
        else load_airtable_state(AirtableBridgeClient.from_env())
    )
    report = build_operator_report(
        state,
        generated_at=None if args.as_of is None else dt(args.as_of),
        stale_grace_hours=args.stale_grace_hours,
    )
    if args.compact:
        print(report.compact_text())
    else:
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
