from datetime import datetime, timezone

import pytest

from gah.board_capacity import BoardCandidate, BoardStatus
from gah.execution import DecayStep, ExecutionPlan, ExecutionState
from gah.live import LiveState
from gah.prospective import ProspectiveAHOutcome, ProspectiveAHSnapshot
from gah.session import (
    BoardSessionCandidate,
    LiveSessionObservation,
    OutcomeObservation,
    orchestrate_board_session,
)


def dt(hour, minute=0):
    return datetime(2026, 10, 1, hour, minute, tzinfo=timezone.utc)


def candidate(
    match_id,
    kickoff,
    priority,
    *,
    line=-0.5,
    current_line=-0.5,
    current_odds=1.92,
    minutes_to_kickoff=90,
    live=None,
    outcome=None,
):
    return BoardSessionCandidate(
        board=BoardCandidate(
            match_id=match_id,
            kickoff=kickoff,
            priority=priority,
            monitor_minutes_before=30,
            monitor_minutes_after=90,
        ),
        snapshot_id=f"s-{match_id}",
        snapshot=ProspectiveAHSnapshot(
            fixture_key=match_id,
            snapshot_time_utc=dt(10, 5).isoformat(),
            competition="EPL",
            home_team="A",
            away_team="B",
            side="home",
            line=line,
            decimal_odds=1.95,
            market_probability=0.50,
            gah_probability=0.57,
        ),
        source="unit-test",
        board_selected_at=dt(10),
        execution_at=dt(10, 10),
        execution_plan=ExecutionPlan(
            side="home",
            protected_line=-0.5,
            protected_minimum_odds=1.90,
            decay_steps=(
                DecayStep(60, -0.75, 1.85),
            ),
        ),
        current_line=current_line,
        current_odds=current_odds,
        minutes_to_kickoff=minutes_to_kickoff,
        live=live,
        outcome=outcome,
    )


def test_board_uses_explicit_priority_and_preserves_rejection():
    low_edge_high_priority = candidate("high-priority", dt(12), 5)
    high_edge_low_priority = candidate("low-priority", dt(12), 1)
    # Edge is deliberately not part of board ordering.
    object.__setattr__(
        low_edge_high_priority.snapshot,
        "gah_probability",
        0.51,
    )
    object.__setattr__(
        high_edge_low_priority.snapshot,
        "gah_probability",
        0.80,
    )
    result = orchestrate_board_session(
        [high_edge_low_priority, low_edge_high_priority],
        max_matches=8,
        max_concurrent=1,
    )
    status = {
        d.candidate.match_id: d.status
        for d in result.board_plan.decisions
    }
    assert status["high-priority"] == BoardStatus.SELECTED
    assert status["low-priority"] == BoardStatus.REJECTED_CONCURRENCY
    rejected = next(
        m for m in result.matches
        if m.board_decision.candidate.match_id == "low-priority"
    )
    assert rejected.operations == ()
    assert rejected.audit_row is None


def test_protected_entry_creates_ordered_snapshot_and_execution_operations():
    result = orchestrate_board_session([candidate("m1", dt(12), 1)])
    match = result.matches[0]
    assert match.execution.state == ExecutionState.ENTER_PROTECTED
    assert match.ledger.current_state.value == "ENTERED_PROTECTED"
    assert match.audit_row["entered"] is True
    assert match.audit_row["entry_type"] == "protected"
    assert any(
        op.fields.get("Outcome Status") == "Pending"
        for op in match.operations
    )
    assert any(
        op.fields.get("State") == "ENTERED_PROTECTED"
        for op in match.operations
    )


def test_decay_entry_is_preserved_not_reselected():
    c = candidate(
        "m2",
        dt(12),
        1,
        current_line=-0.75,
        current_odds=1.86,
        minutes_to_kickoff=45,
    )
    result = orchestrate_board_session([c])
    assert result.matches[0].execution.state == ExecutionState.ENTER_DECAY
    assert result.matches[0].audit_row["entry_type"] == "decay"


def test_live_reprice_is_optional_and_linked():
    live = LiveSessionObservation(
        at=dt(11),
        state=LiveState(
            minute=30,
            home_score=1,
            away_score=0,
            pre_match_home_xg=1.6,
            pre_match_away_xg=1.1,
        ),
        side="home",
        line=-0.5,
        decimal_odds=1.90,
        source="unit-test",
    )
    result = orchestrate_board_session(
        [candidate("m3", dt(12), 1, live=live)]
    )
    match = result.matches[0]
    assert match.live_quote is not None
    assert match.ledger.current_state.value == "LIVE_REPRICED"
    assert any(
        op.fields.get("State") == "LIVE_REPRICED"
        for op in match.operations
    )


def test_entered_outcome_settles_lifecycle_and_audit_uses_entry_odds():
    outcome = OutcomeObservation(
        at=dt(14),
        outcome=ProspectiveAHOutcome(
            fixture_key="m4",
            captured_side="home",
            captured_line=-0.5,
            closing_line=-0.5,
            captured_market_probability=0.50,
            closing_market_probability=0.53,
            settlement_category="full_win",
            realized_net_return=0.95,
        ),
        source="unit-test",
    )
    result = orchestrate_board_session(
        [candidate("m4", dt(12), 1, current_odds=1.92, outcome=outcome)]
    )
    match = result.matches[0]
    assert match.ledger.current_state.value == "SETTLED"
    assert match.audit_row["net_return"] == pytest.approx(0.92)
    assert any(
        op.fields.get("Settlement ID")
        for op in match.operations
    )


def test_passed_snapshot_can_receive_prospective_outcome_without_settled_event():
    outcome = OutcomeObservation(
        at=dt(14),
        outcome=ProspectiveAHOutcome(
            fixture_key="m5",
            captured_side="home",
            captured_line=-0.5,
            closing_line=-0.75,
            captured_market_probability=0.50,
            closing_market_probability=0.55,
            settlement_category="full_loss",
            realized_net_return=-1.0,
        ),
        source="unit-test",
    )
    c = candidate(
        "m5",
        dt(12),
        1,
        current_line=-1.0,
        current_odds=1.80,
        minutes_to_kickoff=-1,
        outcome=outcome,
    )
    result = orchestrate_board_session([c])
    match = result.matches[0]
    assert match.execution.state == ExecutionState.PASS
    assert match.ledger.current_state.value == "PASSED"
    assert all(
        op.fields.get("State") != "SETTLED"
        for op in match.operations
    )
    assert match.audit_row["entered"] is False
    assert match.audit_row["settlement"] == "full_loss"


def test_waiting_snapshot_cannot_be_attached_to_outcome_without_explicit_pass():
    outcome = OutcomeObservation(
        at=dt(14),
        outcome=ProspectiveAHOutcome(
            fixture_key="m6",
            captured_side="home",
            captured_line=-0.5,
            closing_line=-0.5,
            captured_market_probability=0.50,
            closing_market_probability=0.50,
            settlement_category="push",
            realized_net_return=0.0,
        ),
        source="unit-test",
    )
    c = candidate(
        "m6",
        dt(12),
        1,
        current_line=-0.75,
        current_odds=1.92,
        minutes_to_kickoff=90,
        outcome=outcome,
    )
    with pytest.raises(ValueError, match="explicit PASS"):
        orchestrate_board_session([c])


def test_pass_cannot_be_live_repriced_afterward():
    live = LiveSessionObservation(
        at=dt(11),
        state=LiveState(
            minute=10,
            home_score=0,
            away_score=0,
            pre_match_home_xg=1.5,
            pre_match_away_xg=1.0,
        ),
        side="home",
        line=-0.5,
        source="unit-test",
    )
    c = candidate(
        "m7",
        dt(12),
        1,
        current_line=-1.0,
        current_odds=1.80,
        minutes_to_kickoff=-1,
        live=live,
    )
    with pytest.raises(ValueError, match="passed execution"):
        orchestrate_board_session([c])
