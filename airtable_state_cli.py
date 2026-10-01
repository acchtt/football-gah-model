from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from gah.airtable_bridge import AirtableBridgeClient
from gah.airtable_state import (
    STATE_SCHEMA_VERSION,
    load_airtable_state,
    load_state_export,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.11 durable Airtable state loader/exporter"
    )
    ap.add_argument(
        "--input",
        help="Read an existing v2.11 state export instead of Airtable.",
    )
    ap.add_argument(
        "--out",
        help="Write the normalized raw state export to this JSON file.",
    )
    args = ap.parse_args()

    state = (
        load_state_export(args.input)
        if args.input
        else load_airtable_state(AirtableBridgeClient.from_env())
    )

    output = {
        "schema_version": STATE_SCHEMA_VERSION,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        **state.summary(),
        "ledgers": [
            {
                "match_id": ledger.match_id,
                "current_state": ledger.current_state.value,
                "events": len(ledger.events),
            }
            for ledger in state.ledgers
        ],
    }

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                state.to_export(),
                sort_keys=True,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        output["export_path"] = args.out

    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
