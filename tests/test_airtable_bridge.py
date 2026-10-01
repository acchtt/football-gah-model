from datetime import datetime, timezone

import pytest

from gah.airtable_bridge import (
    AirtableBridgeClient,
    BridgeOperationKind,
    canonical_json,
    deduplicate_operations,
    execution_operation,
    execution_operations,
    lifecycle_event_id,
    lifecycle_operation,
    operation_from_manifest,
    operation_manifest,
    prospective_outcome_operation,
    settlement_operations,
    snapshot_operation,
    stable_record_id,
)
from gah.lifecycle import LifecycleEvent, LifecycleState


T0 = datetime(
    2026,
    10,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def event(state, reason="test"):
    return LifecycleEvent(
        LifecycleState(state),
        T0,
        reason,
        "unit-test",
    )


def test_stable_ids_and_canonical_payload_are_deterministic():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json(
        {"a": 1, "b": 2}
    )
    assert stable_record_id(
        "x",
        {"b": 2, "a": 1},
    ) == stable_record_id(
        "x",
        {"a": 1, "b": 2},
    )
    e = event("SNAPSHOT_CAPTURED")
    assert lifecycle_event_id(
        "m1",
        e,
        snapshot_id="s1",
        payload={"b": 2, "a": 1},
    ) == lifecycle_event_id(
        "m1",
        e,
        snapshot_id="s1",
        payload={"a": 1, "b": 2},
    )


def test_snapshot_operation_targets_canonical_prospective_row():
    op = snapshot_operation(
        {
            "Snapshot ID": "s1",
            "Match ID": "m1",
            "Captured At": "2026-10-01T12:00:00+00:00",
            "AH Line": -0.5,
            "Odds": 1.9,
        }
    )
    assert op.kind == BridgeOperationKind.UPSERT
    assert op.key_field == "Snapshot ID"
    assert op.fields["Outcome Status"] == "Pending"
    assert op.fields["Bridge Schema Version"] == "v2.8"


def test_lifecycle_is_append_only_with_event_primary_key():
    op = lifecycle_operation(
        "m1",
        event("SNAPSHOT_CAPTURED"),
        snapshot_id="s1",
        payload={"kind": "snapshot"},
    )
    assert op.table == "tblF4rvY4jQPgb3an"
    assert op.key_field == "Event ID"
    assert op.fields["Snapshot ID"] == "s1"
    assert op.fields["State"] == "SNAPSHOT_CAPTURED"


def test_execution_payload_is_linked_to_snapshot():
    op = execution_operation(
        match_id="m1",
        snapshot_id="s1",
        event=event("ENTERED_PROTECTED"),
        state="ENTERED_PROTECTED",
        current_line=-0.5,
        current_odds=1.92,
        active_minimum_line=-0.5,
        active_minimum_odds=1.9,
        minutes_to_kickoff=80,
    )
    assert op.fields["Snapshot ID"] == "s1"
    assert '"kind":"execution"' in op.fields["Payload JSON"]


def test_settlement_updates_existing_snapshot_and_emits_event():
    batch = settlement_operations(
        match_id="m1",
        snapshot_id="s1",
        event=event("SETTLED"),
        outcome_fields={
            "Outcome Status": "Settled",
            "Settlement": "full_win",
            "Net Return": 0.92,
            "Closing AH Line": -0.5,
            "Closing Market Probability": 0.54,
        },
    )
    update, lifecycle = batch.operations
    assert update.kind == BridgeOperationKind.UPDATE_EXISTING
    assert update.fields["Settlement ID"].startswith("v28set-")
    assert lifecycle.fields["State"] == "SETTLED"
    assert lifecycle.fields["Snapshot ID"] == "s1"


def test_identical_duplicate_operations_collapse():
    op = lifecycle_operation(
        "m1",
        event("WAITING_ENTRY"),
        snapshot_id="s1",
    )
    assert deduplicate_operations([op, op]) == [op]


def test_conflicting_duplicate_operations_are_rejected():
    op1 = snapshot_operation(
        {
            "Snapshot ID": "s1",
            "Match ID": "m1",
            "Captured At": "2026-10-01T12:00:00+00:00",
            "AH Line": -0.5,
            "Odds": 1.9,
        }
    )
    op2 = snapshot_operation(
        {
            "Snapshot ID": "s1",
            "Match ID": "m1",
            "Captured At": "2026-10-01T12:00:00+00:00",
            "AH Line": -0.75,
            "Odds": 1.9,
        }
    )
    with pytest.raises(
        ValueError,
        match="Conflicting duplicate",
    ):
        deduplicate_operations([op1, op2])


def test_naive_event_timestamp_is_rejected():
    bad = LifecycleEvent(
        LifecycleState.WAITING_ENTRY,
        datetime(2026, 10, 1, 12, 0),
        "x",
        "test",
    )
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        lifecycle_operation(
            "m1",
            bad,
            snapshot_id="s1",
        )


class FakeClient(AirtableBridgeClient):
    def __init__(self, records=None):
        super().__init__("token")
        self.records = (
            {}
            if records is None
            else records
        )
        self.calls = []

    def _find(self, op):
        return self.records.get(
            (op.table, op.key_value)
        )

    def _request(
        self,
        method,
        table,
        *,
        query=None,
        payload=None,
    ):
        self.calls.append(
            (method, table, payload)
        )
        if method == "POST":
            return {
                "id": "rec-created",
                "fields": payload["fields"],
            }
        if method == "PATCH":
            return {
                "id": "rec-updated",
                "fields": payload["fields"],
            }
        raise AssertionError(method)


def test_execution_batch_updates_prospective_and_appends_event():
    batch = execution_operations(
        match_id="m1",
        snapshot_id="s1",
        event=event("ENTERED_DECAY"),
        execution_state="ENTER_DECAY",
        current_line=-0.75,
        current_odds=1.86,
        active_minimum_line=-0.75,
        active_minimum_odds=1.85,
        minutes_to_kickoff=40,
    )
    update, evt = batch.operations
    assert update.fields["Entry Type"] == "decay"
    assert update.fields["Entry Odds"] == 1.86
    assert evt.fields["State"] == "ENTERED_DECAY"


def test_update_existing_never_creates_missing_snapshot():
    batch = execution_operations(
        match_id="m1",
        snapshot_id="s1",
        event=event("WAITING_ENTRY"),
        execution_state="WAIT",
        current_line=-0.75,
        current_odds=1.9,
        active_minimum_line=-0.5,
        active_minimum_odds=1.9,
        minutes_to_kickoff=80,
    )
    client = FakeClient()
    with pytest.raises(
        ValueError,
        match="Refusing to create",
    ):
        client.apply(batch.operations[0])


def test_immutable_upsert_conflict_is_rejected():
    op = snapshot_operation(
        {
            "Snapshot ID": "s1",
            "Match ID": "m1",
            "Captured At": "2026-10-01T12:00:00+00:00",
            "AH Line": -0.5,
            "Odds": 1.9,
        }
    )
    existing = {
        "id": "rec1",
        "fields": dict(
            op.fields,
            **{"AH Line": -0.75},
        ),
    }
    client = FakeClient(
        {(op.table, "s1"): existing}
    )
    with pytest.raises(
        ValueError,
        match="Conflicting existing immutable",
    ):
        client.apply(op)


def test_manifest_round_trip():
    op = snapshot_operation(
        {
            "Snapshot ID": "s1",
            "Match ID": "m1",
            "Captured At": "2026-10-01T12:00:00+00:00",
            "AH Line": -0.5,
            "Odds": 1.9,
        }
    )
    assert operation_from_manifest(
        operation_manifest(op)
    ) == op


def test_prospective_outcome_can_settle_without_lifecycle_event():
    op = prospective_outcome_operation(
        snapshot_id="s-pass",
        outcome_fields={
            "Outcome Status": "Settled",
            "Settlement": "full_loss",
            "Net Return": -1.0,
            "Closing AH Line": -0.5,
            "Closing Market Probability": 0.55,
        },
    )
    assert op.kind == BridgeOperationKind.UPDATE_EXISTING
    assert op.fields["Snapshot ID"] == "s-pass"
    assert op.fields["Settlement ID"].startswith("v28set-")
    assert "Last Lifecycle Event ID" not in op.fields
