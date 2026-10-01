from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from gah.airtable_bridge import AirtableBridgeClient, operation_manifest
from gah.board_capacity import BoardCandidate
from gah.execution import DecayStep, ExecutionPlan
from gah.live import LiveAdjustment, LiveState
from gah.prospective import ProspectiveAHOutcome, ProspectiveAHSnapshot
from gah.session import (
    BoardSessionCandidate,
    LiveSessionObservation,
    OutcomeObservation,
    orchestrate_board_session,
)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_candidate(row: dict) -> BoardSessionCandidate:
    board = BoardCandidate(
        match_id=str(row["match_id"]),
        kickoff=dt(row["kickoff"]),
        priority=int(row["priority"]),
        monitor_minutes_before=int(row.get("monitor_minutes_before", 60)),
        monitor_minutes_after=int(row.get("monitor_minutes_after", 120)),
    )
    snapshot = ProspectiveAHSnapshot(**row["snapshot"])

    plan_row = row["execution_plan"]
    plan = ExecutionPlan(
        side=plan_row["side"],
        protected_line=float(plan_row["protected_line"]),
        protected_minimum_odds=float(plan_row["protected_minimum_odds"]),
        expiry_minutes_to_kickoff=int(
            plan_row.get("expiry_minutes_to_kickoff", 0)
        ),
        decay_steps=tuple(
            DecayStep(
                int(step["activates_at_minutes_to_kickoff"]),
                float(step["minimum_line"]),
                float(step["minimum_odds"]),
            )
            for step in plan_row.get("decay_steps", [])
        ),
    )

    live = None
    if row.get("live"):
        raw = row["live"]
        adjustment = LiveAdjustment(**raw.get("adjustment", {}))
        state = LiveState(
            minute=float(raw["minute"]),
            home_score=int(raw["home_score"]),
            away_score=int(raw["away_score"]),
            pre_match_home_xg=float(raw["pre_match_home_xg"]),
            pre_match_away_xg=float(raw["pre_match_away_xg"]),
            adjustment=adjustment,
        )
        live = LiveSessionObservation(
            at=dt(raw["at"]),
            state=state,
            side=raw["side"],
            line=float(raw["line"]),
            decimal_odds=(
                None
                if raw.get("decimal_odds") is None
                else float(raw["decimal_odds"])
            ),
            change_type=raw.get("change_type"),
            reason=raw.get(
                "reason",
                "Live AH re-priced from explicit football state.",
            ),
            source=raw.get("source", row["source"]),
        )

    outcome = None
    if row.get("outcome"):
        raw = row["outcome"]
        outcome = OutcomeObservation(
            at=dt(raw["at"]),
            outcome=ProspectiveAHOutcome(
                fixture_key=row["match_id"],
                captured_side=raw["captured_side"],
                captured_line=float(raw["captured_line"]),
                closing_line=(
                    None
                    if raw.get("closing_line") is None
                    else float(raw["closing_line"])
                ),
                captured_market_probability=float(
                    raw["captured_market_probability"]
                ),
                closing_market_probability=(
                    None
                    if raw.get("closing_market_probability") is None
                    else float(raw["closing_market_probability"])
                ),
                settlement_category=raw["settlement_category"],
                realized_net_return=float(raw["realized_net_return"]),
            ),
            closing_at=raw.get("closing_at"),
            reason=raw.get("reason", "Prospective AH outcome recorded."),
            source=raw.get("source", row["source"]),
        )

    return BoardSessionCandidate(
        board=board,
        snapshot_id=str(row["snapshot_id"]),
        snapshot=snapshot,
        source=str(row["source"]),
        board_selected_at=dt(row["board_selected_at"]),
        execution_at=dt(row["execution_at"]),
        execution_plan=plan,
        current_line=float(row["current_line"]),
        current_odds=float(row["current_odds"]),
        minutes_to_kickoff=int(row["minutes_to_kickoff"]),
        live=live,
        outcome=outcome,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.9 one-board/session orchestrator"
    )
    ap.add_argument("json_file")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))

    result = orchestrate_board_session(
        [parse_candidate(row) for row in payload.get("candidates", [])],
        max_matches=int(payload.get("max_matches", 8)),
        max_concurrent=int(payload.get("max_concurrent", 3)),
    )

    output = {
        "board": [
            {
                "match_id": d.candidate.match_id,
                "status": d.status.value,
                "reason": d.reason,
            }
            for d in result.board_plan.decisions
        ],
        "operations": [
            operation_manifest(op) for op in result.operations
        ],
        "audit_rows": list(result.audit_rows),
        "states": {
            m.board_decision.candidate.match_id: m.ledger.current_state.value
            for m in result.matches
        },
    }
    if args.apply:
        client = AirtableBridgeClient.from_env()
        output["apply_results"] = [
            item.__dict__ for item in result.apply(client)
        ]

    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
