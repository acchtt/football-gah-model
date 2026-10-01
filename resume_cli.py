from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from gah.airtable_bridge import AirtableBridgeClient
from gah.airtable_state import load_airtable_state, load_state_export
from gah.execution import DecayStep, ExecutionPlan
from gah.live import LiveAdjustment, LiveState
from gah.prospective import ProspectiveAHOutcome
from gah.resume import (
    resume_audit,
    resume_execution,
    resume_live,
    resume_outcome,
)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_state(path: str | None):
    if path:
        return load_state_export(path)
    return load_airtable_state(AirtableBridgeClient.from_env())


def parse_plan(raw: dict) -> ExecutionPlan:
    return ExecutionPlan(
        side=str(raw["side"]),
        protected_line=float(raw["protected_line"]),
        protected_minimum_odds=float(raw["protected_minimum_odds"]),
        expiry_minutes_to_kickoff=int(
            raw.get("expiry_minutes_to_kickoff", 0)
        ),
        decay_steps=tuple(
            DecayStep(
                int(step["activates_at_minutes_to_kickoff"]),
                float(step["minimum_line"]),
                float(step["minimum_odds"]),
            )
            for step in raw.get("decay_steps", [])
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GAH v2.12 resumable incremental session update"
    )
    ap.add_argument("update_json")
    ap.add_argument(
        "--state",
        help="Optional v2.11 state export; otherwise load live Airtable.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Apply generated operations to live Airtable.",
    )
    args = ap.parse_args()

    state = load_state(args.state)
    raw = json.loads(Path(args.update_json).read_text(encoding="utf-8"))
    kind = str(raw["kind"]).lower()
    common = {
        "snapshot_id": str(raw["snapshot_id"]),
        "at": dt(raw["at"]),
        "source": str(raw.get("source", "resume-cli")),
    }

    if kind == "execution":
        result = resume_execution(
            state,
            **common,
            plan=parse_plan(raw["plan"]),
            current_line=float(raw["current_line"]),
            current_odds=float(raw["current_odds"]),
            minutes_to_kickoff=int(raw["minutes_to_kickoff"]),
        )
    elif kind == "live":
        adjustment = LiveAdjustment(**raw.get("adjustment", {}))
        live_state = LiveState(
            minute=float(raw["minute"]),
            home_score=int(raw["home_score"]),
            away_score=int(raw["away_score"]),
            pre_match_home_xg=float(raw["pre_match_home_xg"]),
            pre_match_away_xg=float(raw["pre_match_away_xg"]),
            adjustment=adjustment,
        )
        result = resume_live(
            state,
            **common,
            live_state=live_state,
            line=float(raw["line"]),
            decimal_odds=(
                None
                if raw.get("decimal_odds") is None
                else float(raw["decimal_odds"])
            ),
            change_type=raw.get("change_type"),
            reason=str(
                raw.get(
                    "reason",
                    "Live AH re-priced from explicit football state.",
                )
            ),
        )
    elif kind == "outcome":
        outcome = ProspectiveAHOutcome(
            fixture_key=str(raw["match_id"]),
            captured_side=str(raw["captured_side"]),
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
            settlement_category=str(raw["settlement_category"]),
            realized_net_return=float(raw["realized_net_return"]),
        )
        result = resume_outcome(
            state,
            **common,
            outcome=outcome,
            closing_at=raw.get("closing_at"),
            reason=str(
                raw.get("reason", "Prospective AH outcome recorded.")
            ),
        )
    elif kind == "audit":
        result = resume_audit(
            state,
            **common,
            reason=str(raw.get("reason", "Operational audit completed.")),
        )
    else:
        raise ValueError(
            "kind must be execution, live, outcome, or audit."
        )

    payload = result.to_dict()
    if args.apply:
        client = AirtableBridgeClient.from_env()
        payload["apply_results"] = [
            item.__dict__ for item in result.apply(client)
        ]

    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
