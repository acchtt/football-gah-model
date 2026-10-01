from __future__ import annotations

import argparse
import json
from pathlib import Path

from gah.airtable_bridge import (
    AirtableBridgeClient,
    BridgeBatch,
    operation_from_manifest,
    operation_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.8 deterministic Airtable operational bridge"
    )
    ap.add_argument(
        "json_file",
        help="JSON array or {operations:[...]} manifest",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply operations to Airtable using AIRTABLE_TOKEN; "
            "otherwise validate only."
        ),
    )
    args = ap.parse_args()

    payload = json.loads(
        Path(args.json_file).read_text(encoding="utf-8")
    )
    rows = (
        payload.get("operations", [])
        if isinstance(payload, dict)
        else payload
    )
    batch = BridgeBatch(
        tuple(operation_from_manifest(row) for row in rows)
    ).normalized()

    output: dict = {
        "schema_version": "v2.8",
        "operations": [
            operation_manifest(op)
            for op in batch.operations
        ],
    }
    if args.apply:
        client = AirtableBridgeClient.from_env()
        output["results"] = [
            result.__dict__
            for result in client.apply_batch(batch)
        ]

    print(
        json.dumps(
            output,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
