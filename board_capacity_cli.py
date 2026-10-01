from __future__ import annotations

import argparse
import json

import pandas as pd

from gah.board_capacity import BoardCandidate, plan_board


def main() -> None:
    ap = argparse.ArgumentParser(description="GAH v2.5 board capacity planner")
    ap.add_argument("csv", help="Candidate board CSV")
    ap.add_argument("--max-matches", type=int, default=8)
    ap.add_argument("--max-concurrent", type=int, default=3)
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    required = {"match_id", "kickoff", "priority"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    candidates = []
    for _, row in frame.iterrows():
        kickoff = pd.Timestamp(row["kickoff"])
        if kickoff.tzinfo is None:
            raise SystemExit("All kickoff values must include a timezone.")
        candidates.append(
            BoardCandidate(
                match_id=str(row["match_id"]),
                kickoff=kickoff.to_pydatetime(),
                priority=int(row["priority"]),
                monitor_minutes_before=int(
                    row.get("monitor_minutes_before", 60)
                ),
                monitor_minutes_after=int(
                    row.get("monitor_minutes_after", 120)
                ),
            )
        )

    plan = plan_board(
        candidates,
        max_matches=args.max_matches,
        max_concurrent=args.max_concurrent,
    )
    payload = [
        {
            "match_id": d.candidate.match_id,
            "kickoff": d.candidate.kickoff.isoformat(),
            "priority": d.candidate.priority,
            "watch_start": d.candidate.watch_start.isoformat(),
            "watch_end": d.candidate.watch_end.isoformat(),
            "status": d.status.value,
            "reason": d.reason,
        }
        for d in plan.decisions
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
