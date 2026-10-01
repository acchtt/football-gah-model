from __future__ import annotations

import argparse
import json

import pandas as pd

from gah.prospective_review import prospective_readiness, prospective_review


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.7 prospective review readiness gate"
    )
    ap.add_argument("csv", help="Normalized prospective AH CSV")
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    readiness = prospective_readiness(frame)
    payload = {"readiness": readiness.__dict__}

    if readiness.ready:
        payload["review"] = prospective_review(frame).__dict__

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
