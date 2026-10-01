from datetime import datetime, timedelta, timezone

import pandas as pd

from gah.board_capacity import BoardCandidate
from gah.lifecycle import DecisionLedger, LifecycleEvent, LifecycleState
from gah.quality_control import (
    QASeverity,
    normalize_prospective_columns,
    operational_health,
)


T0 = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)


def row(**updates):
    base = {
        "snapshot_id": "s1",
        "match_id": "m1",
        "kickoff": T0 + timedelta(hours=2),
        "captured_at": T0,
        "outcome_status": "Pending",
        "settlement": None,
        "net_return": None,
        "captured_line": -0.5,
        "closing_line": None,
        "market_probability": 0.50,
        "closing_market_probability": None,
        "execution_state": "WAIT",
        "entry_type": None,
        "entry_at": None,
        "entry_line": None,
        "entry_odds": None,
        "settlement_id": None,
    }
    base.update(updates)
    return base


def codes(report, severity=None):
    return {
        issue.code
        for issue in report.issues
        if severity is None or issue.severity == severity
    }


def test_airtable_column_names_are_normalized():
    frame = pd.DataFrame(
        [
            {
                "Snapshot ID": "s1",
                "Match ID": "m1",
                "Captured At": T0.isoformat(),
                "Outcome Status": "Pending",
            }
        ]
    )
    out = normalize_prospective_columns(frame)
    assert {
        "snapshot_id",
        "match_id",
        "captured_at",
        "outcome_status",
    }.issubset(out.columns)


def test_duplicate_snapshot_ids_are_errors():
    frame = pd.DataFrame([row(), row(match_id="m2")])
    report = operational_health(
        frame,
        as_of=T0 + timedelta(hours=1),
    )
    assert "DUPLICATE_SNAPSHOT_ID" in codes(
        report, QASeverity.ERROR
    )
    assert report.status == "ERROR"


def test_entered_match_requires_entry_details():
    frame = pd.DataFrame(
        [
            row(
                execution_state="ENTER_PROTECTED",
                entry_type="protected",
            )
        ]
    )
    report = operational_health(
        frame,
        as_of=T0 + timedelta(hours=1),
    )
    assert "ENTERED_WITHOUT_ENTRY_DETAIL" in codes(
        report, QASeverity.ERROR
    )


def test_settled_missing_result_and_closing_data_are_reported():
    frame = pd.DataFrame(
        [
            row(
                outcome_status="Settled",
                settlement=None,
                net_return=None,
            )
        ]
    )
    report = operational_health(
        frame,
        as_of=T0 + timedelta(hours=3),
    )
    assert "SETTLED_WITHOUT_RESULT" in codes(
        report, QASeverity.ERROR
    )
    assert "MISSING_CLOSING_DATA" in codes(
        report, QASeverity.WARNING
    )


def test_stale_pending_uses_kickoff_when_available():
    frame = pd.DataFrame([row()])
    report = operational_health(
        frame,
        as_of=T0 + timedelta(hours=15),
        stale_grace_hours=12,
    )
    assert "STALE_PENDING_SNAPSHOT" in codes(
        report, QASeverity.WARNING
    )


def test_invalid_lifecycle_transition_is_reported():
    ledger = DecisionLedger(
        "m1",
        (
            LifecycleEvent(
                LifecycleState.PASSED,
                T0,
                "pass",
                "test",
            ),
            LifecycleEvent(
                LifecycleState.ENTERED_PROTECTED,
                T0 + timedelta(minutes=1),
                "invalid",
                "test",
            ),
        ),
    )
    report = operational_health(
        pd.DataFrame([row()]),
        ledgers=[ledger],
        as_of=T0 + timedelta(hours=1),
    )
    assert "INVALID_LIFECYCLE" in codes(
        report, QASeverity.ERROR
    )


def test_duplicate_lifecycle_event_is_reported():
    event = LifecycleEvent(
        LifecycleState.WAITING_ENTRY,
        T0 + timedelta(minutes=2),
        "wait",
        "test",
    )
    ledger = DecisionLedger(
        "m1",
        (
            LifecycleEvent(
                LifecycleState.BOARD_SELECTED,
                T0,
                "board",
                "test",
            ),
            LifecycleEvent(
                LifecycleState.SNAPSHOT_CAPTURED,
                T0 + timedelta(minutes=1),
                "snapshot",
                "test",
            ),
            event,
            event,
        ),
    )
    report = operational_health(
        pd.DataFrame([row()]),
        ledgers=[ledger],
        as_of=T0 + timedelta(hours=1),
    )
    assert "DUPLICATE_LIFECYCLE_EVENT" in codes(
        report, QASeverity.ERROR
    )


def test_lifecycle_beyond_board_without_snapshot_is_error():
    ledger = DecisionLedger(
        "missing",
        (
            LifecycleEvent(
                LifecycleState.BOARD_SELECTED,
                T0,
                "board",
                "test",
            ),
            LifecycleEvent(
                LifecycleState.SNAPSHOT_CAPTURED,
                T0 + timedelta(minutes=1),
                "snapshot",
                "test",
            ),
        ),
    )
    report = operational_health(
        pd.DataFrame([row()]),
        ledgers=[ledger],
        as_of=T0 + timedelta(hours=1),
    )
    assert "LIFECYCLE_WITHOUT_SNAPSHOT" in codes(
        report, QASeverity.ERROR
    )


def test_board_capacity_violation_is_machine_checkable():
    selected = [
        BoardCandidate(
            "a",
            T0 + timedelta(hours=2),
            2,
            monitor_minutes_before=30,
            monitor_minutes_after=90,
        ),
        BoardCandidate(
            "b",
            T0 + timedelta(hours=2),
            1,
            monitor_minutes_before=30,
            monitor_minutes_after=90,
        ),
    ]
    report = operational_health(
        pd.DataFrame([row()]),
        selected_board_candidates=selected,
        max_concurrent=1,
        as_of=T0 + timedelta(hours=1),
    )
    assert "BOARD_CAPACITY_VIOLATION" in codes(
        report, QASeverity.ERROR
    )


def test_readiness_progress_uses_frozen_v21_gate():
    rows = []
    for i in range(10):
        rows.append(
            row(
                snapshot_id=f"s{i}",
                match_id=f"m{i}",
                captured_at=T0 + timedelta(days=i),
                kickoff=T0 + timedelta(days=i, hours=2),
                outcome_status="Settled",
                settlement="full_win",
                net_return=0.9,
                closing_line=-0.5,
                closing_market_probability=0.52,
            )
        )
    report = operational_health(
        pd.DataFrame(rows),
        as_of=T0 + timedelta(days=10),
    )
    assert report.readiness is not None
    assert report.readiness.status == "WAIT"
    assert report.readiness.settled_snapshots == 10
    assert report.readiness.snapshots_remaining == 90
    assert "PROSPECTIVE_GATE_PROGRESS" in codes(
        report, QASeverity.INFO
    )


def test_clean_pending_rows_do_not_create_betting_recommendations():
    report = operational_health(
        pd.DataFrame([row()]),
        as_of=T0 + timedelta(hours=1),
    )
    payload = report.to_dict()
    serialized = str(payload).lower()
    assert "recommendation" not in serialized
    assert "bet" not in serialized
