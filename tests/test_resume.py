from datetime import datetime, timezone

import pytest

from gah.airtable_state import state_from_records
from gah.execution import ExecutionPlan, ExecutionState
from gah.lifecycle import LifecycleState
from gah.live import LiveState
from gah.prospective import ProspectiveAHOutcome
from gah.resume import (
    resume_audit,
    resume_execution,
    resume_live,
    resume_outcome,
)


T0 = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)


def snapshot(
    *,
    execution_state="WAIT",
    outcome_status="Pending",
    last_event_id="",
    settlement_id="",
):
    fields = {
        "Snapshot ID": "s1",
        "Match ID": "m1",
        "Captured At": T0.isoformat(),
        "Kickoff": "2026-10-01T12:00:00+00:00",
        "Side": "home",
        "AH Line": -0.5,
        "Odds": 1.95,
        "Market Probability": 0.50,
        "GAH Probability": 0.57,
        "Outcome Status": outcome_status,
        "Execution State": execution_state,
    }
    if last_event_id:
        fields["Last Lifecycle Event ID"] = last_event_id
    if settlement_id:
        fields["Settlement ID"] = settlement_id
    return {"id": "rec-s1", "fields": fields}


def event(event_id, state, minute, reason, payload="{}"):
    return {
        "id": f"rec-{event_id}",
        "fields": {
            "Event ID": event_id,
            "Match ID": "m1",
            "Snapshot ID": "s1",
            "State": state,
            "Event At": datetime(
                2026,
                10,
                1,
                10,
                minute,
                tzinfo=timezone.utc,
            ).isoformat(),
            "Reason": reason,
            "Source": "unit-test",
            "Payload JSON": payload,
            "Schema Version": "v2.8",
        },
    }


def base_state(events, **snapshot_kwargs):
    return state_from_records(
        [snapshot(**snapshot_kwargs)],
        events,
    )


def plan():
    return ExecutionPlan(
        side="home",
        protected_line=-0.5,
        protected_minimum_odds=1.90,
    )


def test_resume_execution_appends_without_recapturing_snapshot():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
        ]
    )
    result = resume_execution(
        state,
        snapshot_id="s1",
        at=datetime(2026, 10, 1, 10, 5, tzinfo=timezone.utc),
        source="unit-test",
        plan=plan(),
        current_line=-0.5,
        current_odds=1.92,
        minutes_to_kickoff=90,
    )
    assert result.execution.state == ExecutionState.ENTER_PROTECTED
    assert result.current_state == LifecycleState.ENTERED_PROTECTED
    assert len(result.operations) == 2
    assert all(
        op.fields.get("Outcome Status") != "Pending"
        for op in result.operations
    )


def test_resume_wait_can_repeat_validly():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            event("e3", "WAITING_ENTRY", 2, "wait"),
        ]
    )
    result = resume_execution(
        state,
        snapshot_id="s1",
        at=datetime(2026, 10, 1, 10, 6, tzinfo=timezone.utc),
        source="unit-test",
        plan=plan(),
        current_line=-0.75,
        current_odds=1.92,
        minutes_to_kickoff=80,
    )
    assert result.execution.state == ExecutionState.WAIT
    assert result.current_state == LifecycleState.WAITING_ENTRY


def test_resume_live_after_entry():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            event("e3", "ENTERED_PROTECTED", 2, "entered"),
        ],
        execution_state="ENTER_PROTECTED",
    )
    result = resume_live(
        state,
        snapshot_id="s1",
        at=datetime(2026, 10, 1, 10, 30, tzinfo=timezone.utc),
        source="unit-test",
        live_state=LiveState(
            minute=30,
            home_score=1,
            away_score=0,
            pre_match_home_xg=1.6,
            pre_match_away_xg=1.0,
        ),
        line=-0.5,
        decimal_odds=1.90,
    )
    assert result.current_state == LifecycleState.LIVE_REPRICED
    assert result.live_quote is not None
    assert len(result.operations) == 2


def test_entered_outcome_settles_reconstructed_ledger():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            event("e3", "ENTERED_PROTECTED", 2, "entered"),
        ],
        execution_state="ENTER_PROTECTED",
    )
    result = resume_outcome(
        state,
        snapshot_id="s1",
        at=datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc),
        source="unit-test",
        outcome=ProspectiveAHOutcome(
            fixture_key="m1",
            captured_side="home",
            captured_line=-0.5,
            closing_line=-0.5,
            captured_market_probability=0.50,
            closing_market_probability=0.53,
            settlement_category="full_win",
            realized_net_return=0.95,
        ),
    )
    assert result.current_state == LifecycleState.SETTLED
    assert len(result.operations) == 2


def test_passed_outcome_updates_prospective_only():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            event("e3", "PASSED", 2, "pass"),
        ],
        execution_state="PASS",
    )
    result = resume_outcome(
        state,
        snapshot_id="s1",
        at=datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc),
        source="unit-test",
        outcome=ProspectiveAHOutcome(
            fixture_key="m1",
            captured_side="home",
            captured_line=-0.5,
            closing_line=-0.75,
            captured_market_probability=0.50,
            closing_market_probability=0.55,
            settlement_category="full_loss",
            realized_net_return=-1.0,
        ),
    )
    assert result.current_state == LifecycleState.PASSED
    assert len(result.operations) == 1
    assert result.status == "APPEND_PROSPECTIVE_OUTCOME"


def test_nonentered_outcome_without_pass_is_rejected():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            event("e3", "WAITING_ENTRY", 2, "wait"),
        ]
    )
    with pytest.raises(ValueError, match="explicit PASS"):
        resume_outcome(
            state,
            snapshot_id="s1",
            at=datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc),
            source="unit-test",
            outcome=ProspectiveAHOutcome(
                fixture_key="m1",
                captured_side="home",
                captured_line=-0.5,
                closing_line=-0.5,
                captured_market_probability=0.50,
                closing_market_probability=0.50,
                settlement_category="push",
                realized_net_return=0.0,
            ),
        )


def test_audit_transition_can_finish_settled_lifecycle():
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            event("e3", "ENTERED_PROTECTED", 2, "entered"),
            event("e4", "SETTLED", 50, "settled"),
        ],
        execution_state="ENTER_PROTECTED",
        outcome_status="Settled",
    )
    result = resume_audit(
        state,
        snapshot_id="s1",
        at=datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc),
        source="unit-test",
    )
    assert result.current_state == LifecycleState.AUDITED
    assert len(result.operations) == 2


def test_exact_retry_after_state_progressed_emits_no_old_operations():
    wait_at = datetime(2026, 10, 1, 10, 2, tzinfo=timezone.utc)
    wait_reason = "Wait: active line threshold not met."
    state = base_state(
        [
            event("e1", "BOARD_SELECTED", 0, "board"),
            event("e2", "SNAPSHOT_CAPTURED", 1, "snapshot"),
            {
                "id": "rec-e3",
                "fields": {
                    "Event ID": "e3",
                    "Match ID": "m1",
                    "Snapshot ID": "s1",
                    "State": "WAITING_ENTRY",
                    "Event At": wait_at.isoformat(),
                    "Reason": wait_reason,
                    "Source": "unit-test",
                    "Payload JSON": "{}",
                    "Schema Version": "v2.8",
                },
            },
            event("e4", "LIVE_REPRICED", 20, "live"),
        ]
    )
    # The exact same logical WAIT event is already in historical state. Even
    # if a caller retries it, v2.12 must not regress Last Lifecycle Event ID.
    result = resume_execution(
        state,
        snapshot_id="s1",
        at=wait_at,
        source="unit-test",
        plan=plan(),
        current_line=-0.75,
        current_odds=1.92,
        minutes_to_kickoff=80,
    )
    assert result.status == "ALREADY_RECORDED_PAST"
    assert result.operations == ()


def test_settled_snapshot_rejects_new_execution():
    state = base_state(
        [
            event("e1", "PASSED", 0, "pass"),
        ],
        outcome_status="Settled",
        execution_state="PASS",
    )
    with pytest.raises(ValueError, match="settled"):
        resume_execution(
            state,
            snapshot_id="s1",
            at=datetime(2026, 10, 1, 10, 5, tzinfo=timezone.utc),
            source="unit-test",
            plan=plan(),
            current_line=-0.5,
            current_odds=1.92,
            minutes_to_kickoff=90,
        )
