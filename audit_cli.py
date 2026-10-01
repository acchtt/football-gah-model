from __future__ import annotations

import argparse
import json

import pandas as pd

from gah.audit import grouped_audit, summarize_audit


def main() -> None:
    ap = argparse.ArgumentParser(description="GAH v2.4 prospective/execution audit")
    ap.add_argument("csv", help="Audit rows CSV")
    ap.add_argument(
        "--group-by",
        action="append",
        default=[],
        help="Repeatable grouping column, e.g. --group-by competition",
    )
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    summary = summarize_audit(frame)
    payload = {"overall": summary.__dict__}

    if args.group_by:
        payload["groups"] = grouped_audit(frame, args.group_by).to_dict(
            orient="records"
        )

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
