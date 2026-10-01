from datetime import datetime, timedelta, timezone

from gah.airtable_state import state_from_records
from gah.operator_report import build_operator_report


T0 = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)


def snapshot(
    snapshot_id,
    match_id,
    *,
    outcome_status="Pending",
    execution_state="WAIT",
    entry_type=None,
    settlement=None,
    net_return=None,
    closing_line=None,
    closing_probability=None,
    captured_at=T0,
):
    fields = {
        "Snapshot ID": snapshot_id,
        "Match ID": match_id,
        "Kickoff": (captured_at + timedelta(hours=2)).isoformat(),
        "Captured At": captured_at.isoformat(),
        "Outcome Status": outcome_status,
        "Settlement": settlement,
        "Net Return": net_return,
        "AH Line": -0.5,
        "Closing AH Line": closing_line,
        "Market Probability": 0.50,
        "Closing Market Probability": closing_probability,
        "Execution State": execution_state,
        "Entry Type": entry_type,
        "Competition": "EPL",
    }
    if entry_type:
        fields.update(
            {
                "Entry At": (captured_at + timedelta(minutes=5)).isoformat(),
                "Entry Line": -0.5,
                "Entry Odds": 1.92,
            }
        )
    return {"id": f"rec-{snapshot_id}", "fields": fields}


def lifecycle(event_id, match_id, state, at):
    return {
        "id": f"rec-{event_id}",
        "fields": {
            "Event ID": event_id,
            "Match ID": match_id,
            "State": state,
            "Event At": at.isoformat(),
            "Reason": state.lower(),
            "Source": "unit-test",
        },
    }


def test_empty_forward_state_is_valid_wait_not_unavailable():
    state = state_from_records([], [])
    report = build_operator_report(state, generated_at=T0)
    assert report.data_status == "EMPTY"
    assert report.snapshot_count == 0
    assert report.health.status == "OK"
    assert report.readiness is not None
    assert report.readiness.status == "WAIT"
    assert report.readiness.settled_snapshots == 0
    assert report.readiness.snapshots_remaining == 100
    assert report.readiness.days_remaining == 30


def test_active_counts_are_operational_not_predictive():
    state = state_from_records(
        [
            snapshot("s1", "m1", execution_state="WAIT"),
            snapshot(
                "s2",
                "m2",
                execution_state="ENTER_PROTECTED",
                entry_type="protected",
            ),
            snapshot(
                "s3",
                "m3",
                execution_state="PASS",
                outcome_status="Settled",
                settlement="full_loss",
                net_return=-1.0,
                closing_line=-0.75,
                closing_probability=0.55,
            ),
        ],
        [],
    )
    report = build_operator_report(
        state,
        generated_at=T0 + timedelta(hours=3),
    )
    assert report.data_status == "ACTIVE"
    assert report.snapshot_count == 3
    assert report.pending_snapshots == 2
    assert report.settled_snapshots == 1
    assert report.entered_snapshots == 1
    assert report.passed_snapshots == 1
    assert report.waiting_snapshots == 1


def test_lifecycle_counts_use_current_state_per_match():
    state = state_from_records(
        [
            snapshot("s1", "m1"),
            snapshot(
                "s2",
                "m2",
                execution_state="ENTER_PROTECTED",
                entry_type="protected",
            ),
        ],
        [
            lifecycle("e1", "m1", "BOARD_SELECTED", T0),
            lifecycle(
                "e2",
                "m1",
                "SNAPSHOT_CAPTURED",
                T0 + timedelta(minutes=1),
            ),
            lifecycle(
                "e3",
                "m1",
                "WAITING_ENTRY",
                T0 + timedelta(minutes=2),
            ),
            lifecycle("e4", "m2", "BOARD_SELECTED", T0),
            lifecycle(
                "e5",
                "m2",
                "SNAPSHOT_CAPTURED",
                T0 + timedelta(minutes=1),
            ),
            lifecycle(
                "e6",
                "m2",
                "ENTERED_PROTECTED",
                T0 + timedelta(minutes=2),
            ),
        ],
    )
    report = build_operator_report(
        state,
        generated_at=T0 + timedelta(hours=1),
    )
    assert report.lifecycle_state_counts["WAITING_ENTRY"] == 1
    assert report.lifecycle_state_counts["ENTERED_PROTECTED"] == 1
    assert report.lifecycle_state_counts["BOARD_SELECTED"] == 0


def test_health_errors_flow_through_without_becoming_recommendations():
    bad = snapshot(
        "s1",
        "m1",
        execution_state="ENTER_PROTECTED",
        entry_type=None,
    )
    state = state_from_records([bad], [])
    report = build_operator_report(
        state,
        generated_at=T0 + timedelta(hours=1),
    )
    payload = report.to_dict()
    assert report.health.status == "ERROR"
    codes = {item["code"] for item in payload["health"]["issues"]}
    assert "ENTERED_WITHOUT_ENTRY_DETAIL" in codes
    assert "recommendation" not in str(payload).lower()


def test_readiness_progress_is_frozen_100_and_30():
    rows = []
    for i in range(10):
        captured = T0 + timedelta(days=i)
        rows.append(
            snapshot(
                f"s{i}",
                f"m{i}",
                execution_state="PASS",
                outcome_status="Settled",
                settlement="full_win",
                net_return=0.9,
                closing_line=-0.5,
                closing_probability=0.52,
                captured_at=captured,
            )
        )
    state = state_from_records(rows, [])
    report = build_operator_report(
        state,
        generated_at=T0 + timedelta(days=10),
    )
    assert report.readiness is not None
    assert report.readiness.settled_snapshots == 10
    assert report.readiness.snapshots_remaining == 90
    assert report.readiness.observation_span_days == 9
    assert report.readiness.days_remaining == 21


def test_compact_report_contains_only_operational_status():
    state = state_from_records([], [])
    text = build_operator_report(state, generated_at=T0).compact_text()
    assert "GAH v2.13" in text
    assert "data=EMPTY" in text
    assert "settled=0/100" in text
    assert "span=0/30d" in text
    assert "recommend" not in text.lower()


def test_all_pending_airtable_shape_keeps_readiness_available():
    # Airtable omits every blank settlement/closing field while the sample is
    # pending. The report must still interpret that as valid forward state.
    record = snapshot("s1", "m1", execution_state="WAIT")
    fields = record["fields"]
    for key in (
        "Settlement",
        "Net Return",
        "Closing AH Line",
        "Closing Market Probability",
        "Entry Type",
    ):
        fields.pop(key, None)

    state = state_from_records([record], [])
    report = build_operator_report(
        state,
        generated_at=T0 + timedelta(hours=1),
    )
    assert report.readiness is not None
    assert report.readiness.status == "WAIT"
    assert report.readiness.settled_snapshots == 0
    assert report.readiness.snapshots_remaining == 100
    assert report.health.status == "OK"
